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

# TODO: Need to write all of these results out somewhere

## --- CODE BLOCK ---

tf <- import("too_predict.filter")
ut <- import("too_predict.utils")
pd <- import("pandas")
ad <- import("anndata")
go_annotations <- evoGO::loadGOAnnotation(species = "hsapiens", path = as.character(ut$get_data("")))
gene_sets <- gs_internal()
bp_param <- MulticoreParam(workers = multicoreWorkers())

get_tags <- function(qlf, count) {
  topTags(qlf, n = count, sort.by = "PValue") |>
    as.data.frame() |>
    as_tibble() |>
    relocate(all_of(added_cols), .after = "GENEID")
}

## * Setup
wanted_stype <- c("primary", "organoid")
if (path.expand("~") != "/home/shannc") {
  wanted_ttype <- c("PAAD", "COAD_READ", "LIHC", "CHOL")
  adata_fn <- function() {
    adata <- ut$training_data_internal()
    adata <- adata[, 1:2000]
    adata
  }
  gsa_nperm <- 5
} else {
  ## test_file <- here(output())
  adata_fn <- function() {
    ad$read_h5ad(here(
      "data", "output", "normalization_comparison",
      "edgeR_median_lfc_feature_list_3000", "none-plus_one.h5ad"
    ))
  }
  gsa_nperm <- NULL
  wanted_ttype <- c("PAAD", "COAD_READ", "LIHC", "CHOL")
}

adata <- adata_fn()
adata$obs$tumor_type <- str_replace_all(adata$obs$tumor_type, "-", "_")
adata <- adata[adata$obs$tumor_type %in% wanted_ttype, ]
adata <- adata[adata$obs$Sample_Type %in% wanted_stype | grepl("CHULA", adata$obs$Project_ID), ]
outdir_o <- here("data", "output", "chula_organoid_comparison")

## * Get DE genes

dge <- adata2dge(adata)

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

dge <- normLibSizes(dge)
dge <- estimateDisp(dge, design = mm, robust = TRUE)

added_cols <- c("logFC", "logCPM", "F", "PValue", "FDR")
other_cols <- colnames(dge$genes)
unwanted_cols <- c(
  "GENEBIOTYPE", "SEQNAME", "SEQLENGTH", "n_cells_by_counts",
  "mean_counts", "log1p_mean_counts", "pct_dropout_by_counts", "total_counts",
  "log1p_total_counts", "n_cells", "n_counts"
)

fit <- glmQLFit(dge, mm, robust = TRUE)

top_tags <- sapply(names(contrasts), \(n) {
  qlf <- glmQLFTest(fit, contrast = contrasts[[n]])
  ## plotMD(qlf)
  message(glue("Comparison: {qlf$comparison}"))
  get_tags(qlf, count = nrow(dge)) |>
    ## rename_with(\(cols) {
    ##   map_chr(cols, \(x) {
    ##     if (x %in% added_cols) {
    ##       glue("{n}_{x}")
    ##     } else {
    ##       x
    ##     }
    ##   })
    ## }) |>
    select(-any_of(unwanted_cols))
}, simplify = FALSE, USE.NAMES = TRUE)


## * Plots

adj_counts <- edgeR::cpm(dge)
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

# Want to know how organoid vs. primary differ on the basis of gene sets

lfc_filter <- function(tag_tb) {
  lfc_threshold <- 0 # TODO: any way to choose a good threshold?
  alpha <- 0.05
  fdr_cutoff <- 0.05
  tag_tb |> filter(abs(logFC) >= lfc_threshold & PValue <= alpha & FDR <= fdr_cutoff)
}

# Validate gene sets by..
## - keeping sets where at least n_keep% of genes are nonzero in the data
## - removing setes with less than n_genes
## gene_sets
n_keep_percent <- 90
min_genes <- 5
gene2avg <- rowMeans(adj_counts)
gene_set_mask <- bplapply(names(gene_sets), \(name) {
  set <- gene_sets[[name]]
  if (length(set) < min_genes) {
    return(FALSE)
  }
  expr_level <- map_lgl(set, \(x) gene2avg[x] > 0) |> replace_na(FALSE)
  (sum(expr_level) / length(set)) * 100 > n_keep_percent
}, BPPARAM = bp_param) |>
  unlist()
gene_sets <- gene_sets[gene_set_mask]

## ** ORA

do_ora <- FALSE
if (do_ora) {
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


## ** Gene set enrichment analysis
# Operates on pre-ranked genes i.e. uses results from edgeR above

do_fgsea <- TRUE
if (do_fgsea) {
  library(fgsea)

  fgsea_alpha <- 0.05
  fgsea_results <- sapply(names(top_tags), \(n) {
    tb <- top_tags[[n]]
    sorted <- lfc_filter(tb) |>
      arrange(logFC)
    ranked <- setNames(sorted$logFC, sorted$GENEID)
    fgsea(pathways = gene_sets, stats = ranked) |> filter(padj <= fgsea_alpha)
  }, simplify = FALSE, USE.NAMES = TRUE)
}

# Methods below use raw expression data

## ** Gene Set Analysis

# [2025-04-18 Fri] More direct way of getting the ora results
do_gsa <- TRUE
if (do_gsa) {
  library(GSALightning)
  dge$genes$sd_adj <- apply(adj_counts, 1, sd)
  to_gsa <- adj_counts[dge$genes$sd_adj != 0, ]

  gsa <- GSALight(
    eset = to_gsa, fac = dge$samples$Sample_Type,
    gs = gene_sets, rmGSGenes = "gene",
    nperm = gsa_nperm
  ) |> as_tibble()
}

## ** PLAGE

do_plage <- TRUE
if (do_plage) {
  library(GSVA)
  library(limma)
  params <- plageParam(exprData = adj_counts, geneSets = gene_sets)
  plage <- gsva(params) |> as.data.frame()

  # Use limma for comparison, not edgeR because the latter needs the raw units
  fit_gs <- lmFit(plage, design = mm)

  de_gs <- lapply(names(contrasts), \(name) {
    contrast_fit <- contrasts.fit(fit_gs, contrasts[[name]])
    contrast_fit <- eBayes(contrast_fit, robust = TRUE)
    result <- decideTests(contrast_fit,
      lfc = log2(1.5) # Judge significant when abs(log2-fc) is at least this large
    )
    df <- data.frame(row.names = rownames(result))
    df[[name]] <- as.double(result[, 1])
    df
  }) |>
    bind_cols()
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

## * Cross-reference with markers
# TODO: need to find marker sets
