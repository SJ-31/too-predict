suppressMessages({
  library(here)
  library(tidyverse)
  library(glue)
  library(edgeR)
  library(paletteer)
  library(reticulate)
  library(BiocParallel)
  use_condaenv("too-predict")
  source(here("src", "R", "utils.R"))
  source(here("src", "R", "plotting.R"))
  ExperimentHub::setExperimentHubOption("CACHE", here("data", ".ExperimentHubCache"))
})

## --- CODE BLOCK ---

tf <- import("too_predict.filter")
ut <- import("too_predict.utils")
pd <- import("pandas")
ad <- import("anndata")

gene_sets <- gs_internal(
  from_file = FALSE, min_size = 15,
  max_size = 500
) # Gene sets must have at least `min_size` genes

bp_param <- MulticoreParam(workers = multicoreWorkers())

get_tags <- function(qlf, count) {
  topTags(qlf, n = count, sort.by = "PValue") |>
    as.data.frame() |>
    as_tibble() |>
    relocate(any_of(ADDED_COLS), .after = "GENEID")
}

## * Setup
wanted_stype <- c("primary", "organoid")
outdir_o <- here("data", "output", "chula_organoid_comparison", "de_enrichment")
if (path.expand("~") != "/home/shannc") {
  wanted_ttype <- c("PAAD", "COAD_READ", "LIHC", "CHOL")
  adata_fn <- function() {
    adata <- ut$training_data_internal()
    adata
  }
  gsa_nperm <- NULL
  validate <- TRUE
  storage <- here("remote", "repos", "too-predict", "organoid_comparison")
} else {
  adata_fn <- function() {
    adata <- ad$read_h5ad(here(
      "data", "output", "normalization_comparison",
      "edgeR_median_lfc_feature_list_3000", "none-plus_one.h5ad"
    ))
    adata <- adata[, 1:2000]
    adata
  }
  gsa_nperm <- 5
  storage <- here(outdir_o, ".storage")
  outdir_o <- here(outdir_o, "test")
  wanted_ttype <- c("PAAD", "COAD_READ", "LIHC", "CHOL")
  validate <- FALSE
}
dir.create(outdir_o)
dir.create(storage)
outdir_markers <- here(outdir_o, "markers")
outdir_gs <- here(outdir_o, "gene_sets")
dir.create(outdir_markers)
dir.create(outdir_gs)

adata <- adata_fn()
adata$obs$tumor_type <- str_replace_all(adata$obs$tumor_type, "-", "_")
adata <- adata[adata$obs$tumor_type %in% wanted_ttype, ]
adata <- adata[adata$obs$Sample_Type %in% wanted_stype | grepl("CHULA", adata$obs$Project_ID), ]

## * Get DE genes

dge <- adata2dge(adata, convert_na = TRUE)

rm(adata)

dge$samples$tumor_type <- factor(dge$samples$tumor_type)
dge$samples$Sample_type <- factor(dge$samples$Sample_Type)
dge$samples$combined <- factor(paste0(dge$samples$tumor_type, ".", dge$samples$Sample_Type))

mm <- model.matrix(~ 0 + combined, dge$samples)
colnames(mm) <- str_remove(colnames(mm), "tumor_type") |>
  str_replace(":", "_") |>
  str_remove("combined")

## [1] "CHOL.organoid"      "CHOL.primary"       "COAD_READ.organoid"
## [4] "COAD_READ.primary"  "LIHC.organoid"      "LIHC.primary"
## [7] "PAAD.organoid"      "PAAD.primary"
contrasts <- list(
  sample_type = c(1, -1, 1, -1, 1, -1, 1, -1), # 1
  coad_read = makeContrasts("COAD_READ.organoid - COAD_READ.primary", levels = mm), # 2
  lihc = makeContrasts("LIHC.organoid - LIHC.primary", levels = mm), # 3
  paad = makeContrasts("PAAD.organoid - PAAD.primary", levels = mm), # 4
  chol = makeContrasts("CHOL.organoid - CHOL.primary", levels = mm) # 5
)

# Explanation
# 1 highlights the differences between primary and organoid samples,
#   while adjusting for tumor-type specific differences
# The other contrasts find DE genes between sample types within a specific tumor type

# For contrast 1, positive LFC are genes that are up in organoid over primary overall
# For contrasts 2-5, positive LFC are genes that are up in
#   organoid over primary in that ttype
ADDED_COLS <- c("logFC", "logCPM", "F", "PValue", "FDR", "unshrunk.logFC")

de_summary_file <- here(outdir_o, "de_summary.tsv")
do_de <- function(output) {
  ## dge <- filterByExpr(dge, design = mm, min.count = 10, min.prop = 0.8)
  dge <- normLibSizes(dge)
  dge <- estimateDisp(dge, design = mm, robust = TRUE)

  other_cols <- colnames(dge$genes)
  unwanted_cols <- c(
    "GENEBIOTYPE", "SEQNAME", "SEQLENGTH", "n_cells_by_counts",
    "mean_counts", "log1p_mean_counts", "pct_dropout_by_counts", "total_counts",
    "log1p_total_counts", "n_cells", "n_counts"
  )

  fit <- glmQLFit(dge, mm, robust = TRUE)
  decided <<- list()
  top_tags <<- sapply(names(contrasts), \(n) {
    qlf <- glmTreat(fit, contrast = contrasts[[n]], lfc = log2(1.2))
    message(glue("Comparison: {qlf$comparison}"))
    dec <<- summary(decideTests(qlf, p.value = 0.05))
    decided[[n]] <<- as_tibble(as.integer(dec)) |>
      rename(!!as.symbol(n) := value) |>
      mutate(direction = rownames(dec))
    get_tags(qlf, count = nrow(dge)) |> select(-any_of(unwanted_cols))
  }, simplify = FALSE, USE.NAMES = TRUE)
  saveRDS(dge, output)
  suppressMessages(lmap(top_tags, \(x) write_tsv(x[[1]], file = here(outdir_o, glue("{names(x)}_top_tags.tsv")))))
  purrr::reduce(decided, \(x, y) inner_join(x, y, by = join_by(direction))) |>
    relocate(direction, .before = everything()) |>
    write_tsv(de_summary_file)
  dge
}

dge_file <- here(storage, "dge.rds")
if (file.exists(dge_file)) {
  top_tags <- sapply(names(contrasts), \(n) {
    read_tsv(here(outdir_o, glue("{n}_top_tags.tsv")))
  }, simplify = FALSE, USE.NAMES = TRUE)
}
dge <- read_existing(dge_file, do_de, readRDS)
tagdir <- here(outdir_o, "top_tags_sorted")
dir.create(tagdir)
suppressMessages({
  lmap(top_tags, \(x) {
    filter(x[[1]], PValue <= 0.1) |>
      arrange(desc(abs(logFC))) |>
      dplyr::slice_head(n = 2000) |>
      write_tsv(here(tagdir, glue("{names(x)}_top_tags.tsv")))
  })
})


## * Plots

adj_counts <- edgeR::cpm(dge)
log_adj_counts <- edgeR::cpm(dge, log = TRUE)
rownames(adj_counts) <- dge$genes$GENEID

plot_spec <- sapply(names(top_tags), \(n) {
  if (n == "sample_type") {
    c("primary", "organoid")
  } else {
    upper <- str_to_upper(n)
    c(glue("{upper}.primary"), glue("{upper}.organoid"))
  }
}, simplify = FALSE, USE.NAMES = TRUE)


fdr_cutoff <- 0.01
lapply(names(top_tags), \(x) {
  cur_tags <- top_tags[[x]]
  vplot <- volcano_plot(cur_tags, fdr_cutoff = fdr_cutoff) + labs(title = x)
  vplot_name <- here(outdir_o, glue("{x}_volcano.png"))
  ggsave(vplot_name, plot = vplot, height = 8, width = 8)

  if (x == "sample_type") {
    lc <- "Sample_Type"
  } else {
    lc <- "combined"
  }
  lfc_spec <- plot_spec[[x]]
  lfc_plot <- plot_lfc(
    x = lfc_spec[1], y = lfc_spec[2], label_col = lc,
    cpm = adj_counts, dge = dge, tag_tb = cur_tags,
    p_value = 0.05
  ) + scale_color_paletteer_c("ggthemes::Red-Green Diverging") + labs(title = x)
  lfc_plot_name <- here(outdir_o, glue("{x}_cpm_comparison.png"))
  ggsave(lfc_plot_name, plot = lfc_plot, height = 8, width = 8)
})


## * Enrichment analyses

gs_meta <- read_tsv(gs_meta_internal()) |> mutate(set_name = paste0(source, ":", name))

# Want to know how organoid vs. primary differ on the basis of gene sets
enrichment_analyses <- list(
  ora = FALSE,
  fgsea = TRUE,
  gsa = TRUE,
  plage = TRUE
)

lfc_filter <- function(tag_tb) {
  lfc_threshold <- 0 # TODO: any way to choose a good threshold?
  alpha <- 0.05
  fdr_cutoff <- 0.05
  tag_tb |> filter(abs(logFC) >= lfc_threshold & PValue <= alpha & FDR <= fdr_cutoff)
}


# Validate gene sets by..
## - keeping sets where at least n_keep% of genes are nonzero in the data
## gene_sets
min_nonzero_percent <- 70
min_sample_percent <- 70
sample_stype_map <- rownames_to_column(dge$samples, var = "sample") |>
  select(sample, Sample_Type, Project_ID)
if (validate) {
  ## gene_sets <- filter_gene_sets(
  ##   gene_sets = gene_sets, counts = adj_counts,
  ##   min_nonzero_percent = min_nonzero_percent, min_sample_percent = min_sample_percent
  ## )
  ## [2025-04-24 Thu] Nothing passed...
  ## gs_meta |>
  ##   filter(set_name %in% names(gene_sets)) |>
  ##   write_tsv(here(outdir_o, "tested_gene_sets.tsv"))

  gs_stats <- filter_gene_sets(counts = adj_counts, gene_sets = gene_sets, stats_only = TRUE)
  agg_gene_set_fractions(gs_stats, sample_stype_map, c("Sample_Type", "Project_ID")) |>
    write_tsv(here(outdir_o, "gene_set_nonzero_percent.tsv"))
  gs_stats |> write_tsv(here(outdir_o, "gene_set_counts.tsv"))
}

## ** ORA

if (enrichment_analyses$ora) {
  go_annotations <- evoGO::loadGOAnnotation(species = "hsapiens", path = as.character(ut$get_data("")))
  universe <- dge$genes$GENEID
  ora_alpha <- 0.05
  overrepresented <- sapply(names(top_tags), \(n) {
    tb <- lfc_filter(top_tags[[n]])
    up <- tb |> filter(logFC > 0)
    down <- tb |> filter(logFC < 0)
    enrich <- list(up = up, down = down)
    lapply(names(enrich), \(x) {
      evoGO::calcGOenrichment(go_annotations,
        deGenes = enrich[[x]]$GENEID,
        universe = universe
      ) |>
        as_tibble() |>
        filter(evogo.pvalue <= ora_alpha) |>
        mutate(direction = x) |>
        select(-def)
    }) |>
      bind_rows()
  }, simplify = FALSE, USE.NAMES = TRUE)
}


## ** Gene set enrichment analysis (FGSEA)
# Operates on pre-ranked genes i.e. uses results from edgeR above

fgsea_helper <- function(gene_sets, alpha = 0.05, plotdir) {
  library(fgsea)
  fgsea_results <- sapply(names(top_tags), \(n) {
    tb <- top_tags[[n]]
    sorted <- lfc_filter(tb) |> arrange(logFC)
    warning(glue("There are {sum(is.na(sorted$GENEID))} genes with missing ENSEMBL ids!"))
    sorted <- filter(sorted, !is.na(GENEID))
    ranked <- setNames(sorted$logFC, sorted$GENEID)
    result <- fgsea(pathways = gene_sets, stats = ranked)
    result <- result[result$padj <= alpha, ]

    lapply(result$pathway, \(pwy) {
      plot <- plotEnrichment(pathway = gene_sets[[pwy]], stats = ranked) + labs(title = pwy)
      name <- str_replace_all(pwy, " ", "_") |> str_replace_all("/", "-")
      ggsave(here(plotdir, glue("{name}_enrichment.png")), plot, height = 8, width = 8)
    })
    fgsea_tb <- result |> mutate(leadingEdge = map_chr(leadingEdge, \(x) paste0(x, collapse = ";")))
    fgsea_tb
  }, simplify = FALSE, USE.NAMES = TRUE)
  fgsea_results
}

if (enrichment_analyses$fgsea && !file.exists(here(outdir_gs, "fgsea_sample_type.tsv"))) {
  dir.create(here(outdir_gs, "fgsea_plots"))
  fgsea_gene_sets <- fgsea_helper(gene_sets = gene_sets, alpha = 0.05, plotdir = here(outdir_gs, "fgsea_plots"))
  suppressMessages({
    lmap(fgsea_gene_sets, \(x)  {
      write_tsv(x[[1]], here(outdir_gs, glue("fgsea_{names(x)}.tsv")))
    })
  })
}

# Methods below use raw expression data

## ** Gene Set Analysis

# [2025-04-18 Fri] More direct way of getting the ora results
do_gsa <- function(outfile, gene_sets, metadata) {
  library(GSALightning)
  dge$genes$sd_adj <- apply(adj_counts, 1, sd)
  to_gsa <- adj_counts[dge$genes$sd_adj != 0, ]
  gsa <- GSALight(
    eset = to_gsa, fac = dge$samples$Sample_Type,
    gs = gene_sets, rmGSGenes = "gene",
    nperm = gsa_nperm
  ) |>
    as.data.frame() |>
    rownames_to_column(var = "set_name")
  write_tsv(gsa, outfile)
  gsa
}

if (enrichment_analyses$gsa) {
  gsa_out <- here(outdir_gs, "gsa.tsv")
  gsa <- read_existing(gsa_out, \(x) do_gsa(x, gene_sets, gs_meta), read_tsv)
}

## ** PLAGE

if (enrichment_analyses$plage) {
  plage <- plage_wrapper(
    counts = adj_counts,
    gene_sets = gene_sets, fc_cutoff = 1.2,
    contrasts = contrasts,
    model_matrix = mm
  )
  plage$de <- plage$de |> rownames_to_column(var = "set_name")
  write_tsv(plage$de, here(outdir_gs, "plage_decideTests.tsv"))
  suppressMessages({
    lapply(names(plage$topTreats), \(x) {
      df <- plage$topTreats[[x]] |> rownames_to_column(var = "set_name")
      write_tsv(df, here(outdir_gs, glue("plage_topTreat_{x}.tsv")))
    })
  })
}

# Value interpretation
# -1 as significantly negative in the comparison
# 0 not significant
# 1 significant


## ** Globaltest
# [2025-04-18 Fri] You've set it up correctly, but way too slow
## library(globaltest)
## gt.options(transpose = TRUE, trim = TRUE)
## # Remove the GO sets from this because of their implicit structure
## global <- gt(
##   response = dge$samples$Sample_Type,
##   alternative = adj_counts,
##   subsets = gene_sets
## )
## p.adjust(global)

## ** Final summary

summarize_gs <- list()
for (g in names(contrasts)) {
  summarize_gs[[glue("FGSEA {g}")]] <- fgsea_gene_sets[[g]] |> rename(set_name = pathway)
  summarize_gs[[glue("PLAGE {g}")]] <- plage$topTreats[[g]] |>
    rownames_to_column(var = "set_name") |>
    rename(padj = adj.P.Val)
}
summarize_gs[["GSA organoid up-regulated"]] <- select(gsa, set_name, `p-value:up-regulated in organoid`) |>
  rename(padj = `p-value:up-regulated in organoid`)
summarize_gs[["GSA primary up-regulated"]] <- select(gsa, set_name, `p-value:up-regulated in primary`) |>
  rename(padj = `p-value:up-regulated in primary`)

gs_summary <- enrichment_summary(gs_meta, summarize_gs)
write_tsv(gs_summary, here(outdir_gs, "enrichment_summary.tsv"))

## * Cross-reference with markers

min_marker_size <- 10
max_marker_size <- 1000
marker_sets <- markers_internal()
marker_lengths <- map_dbl(marker_sets, length)
marker_sets <- marker_sets[marker_lengths >= min_marker_size & marker_lengths <= max_marker_size]
marker_meta <- markers_meta_internal() |>
  mutate(
    set_name = paste0(tissue, "-", cell_type),
    set_name = case_when(from_tme ~ paste0(set_name, "-tme"), .default = set_name)
  ) |>
  select(-all_of(c("tissue", "cell_type", "ensembl", "from_tme"))) |>
  group_by(set_name) |>
  summarise(size = n(), source = dplyr::first(source))

## marker_sets <- filter_gene_sets(marker_sets,
##   counts = adj_counts, min_nonzero_percent = min_nonzero_percent,
##   min_sample_percent = min_sample_percent
## )
## marker_meta |>
##   filter(set_name %in% names(marker_sets)) |>
##   write_tsv(here(outdir_o, "tested_marker_sets.tsv"))
marker_stats <- filter_gene_sets(counts = adj_counts, gene_sets = marker_sets, stats_only = TRUE)

marker_agg <- agg_gene_set_fractions(marker_stats, sample_stype_map, c("Sample_Type", "Project_ID"))
marker_agg |> write_tsv(here(outdir_o, "marker_nonzero_percent.tsv"))
marker_stats |> write_tsv(here(outdir_o, "marker_counts.tsv"))

gsa_marker_file <- here(outdir_markers, "gsa_markers.tsv")
read_existing(gsa_marker_file, \(x) do_gsa(x, marker_sets, marker_meta), read_tsv)

if (!file.exists(here(outdir_markers, "fgsea_sample_type.tsv"))) {
  dir.create(here(outdir_markers, "fgsea_plots"))
  fgsea_markers <- fgsea_helper(marker_sets, alpha = 0.05, plotdir = here(outdir_markers, "fgsea_plots"))
  suppressMessages({
    lmap(fgsea_markers, \(x)  {
      write_tsv(x[[1]], here(outdir_markers, glue("fgsea_{names(x)}.tsv")))
    })
  })
}
