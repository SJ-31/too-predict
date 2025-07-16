suppressMessages({
  library(here)
  library(BiocParallel)
  library(glue)
  library(tidyverse)
  library(zellkonverter)
  library(httr2)
  library(data.table)
  library(scRNAseq)
  library(edgeR)
})


suffix <- ""
if (sys.nframe() == 0) {
  library("optparse")
  parser <- OptionParser()
  parser <- add_option(
    parser,
    c("-t", "--test"),
    type = "logical",
    default = FALSE,
    action = "store_true"
  )
  parser <- add_option(
    parser,
    c("-c", "--cores"),
    type = "integer",
    default = 8
  )
  parser <- add_option(
    parser,
    c("-g", "--recode_go"),
    type = "logical",
    default = FALSE,
    action = "store_true"
  )
  args <- parse_args(parser)
  if (args$recode_go) {
    suffix <- "_GO"
  }
}

library(reticulate)
use_condaenv("too-predict")
source(here("src", "R", "utils.R"))

utils <- import("too_predict.utils")
outdir <- here("data", "output", "feature_selection")
if (args$test) {
  print("Using test subset")
  outdir <- here(outdir, "test")
  dir.create(outdir, recursive = TRUE)
  adata <- utils$training_data_internal_test(minimal = TRUE)
} else {
  adata <- utils$training_data_internal()
}
if (args$recode_go) {
  adata <- utils$recode_to_go(adata)
}

adata$obs$tumor_type <- str_replace_all(adata$obs$tumor_type, "-", "_")
adata$obs$Sample_Type <- str_replace_all(adata$obs$Sample_Type, "-", "_")
data <- AnnData2SCE(adata)
rm(utils)

# %%
p_threshold <- 0.05
GROUP <- "tumor_type"
## technical_factors <- c("Sample_Type", "Project_ID") TODO: need to address confounding
TECHNICAL_FACTORS <- NULL

make_counts <- function(adata_sce) {
  counts <- assays(adata_sce)$X |> as.matrix()
  mode(counts) <- "integer"
  assays(adata_sce, withDimnames = FALSE)$X <- counts
  counts
}

counts <- make_counts(data)

## * DGE with ALDEx2

get_aldex <- function(f) {
  result <- aldex_glm_wrapper(
    data,
    GROUP,
    TECHNICAL_FACTORS,
    use_parallel = TRUE
  )
  # TODO: maybe use the scale aware version by specifying gamma
  effect <- as_tibble(result$effect) # Effect size are standardized mean differences
  test <- as_tibble(result$test)

  # Take the average effects of all comparisons
  between_groups <- effect |> filter(str_starts(contrast, GROUP))
  id_col <- test$gene_id
  tb_list <- between_groups |>
    group_by(contrast) |>
    select(where(is.numeric)) |>
    nest() |>
    pluck("data")
  averaged <- (purrr::reduce(tb_list, \(x, y) x + y) / length(tb_list)) |>
    as_tibble() |>
    mutate(
      gene_id = id_col,
      abs_effect = abs(effect)
    )

  # Features with least change across conditions,
  # possible candidates for ALR
  n_lowest <- averaged |>
    arrange(abs_effect) |>
    dplyr::slice(1:n)

  # Features with most change, for machine learning
  # Must be statistically significant across all comparisons
  significant <- test |>
    select(gene_id, contains(GROUP) & contains("pval.padj")) |>
    filter(if_all(where(is.numeric), \(x) x <= p_threshold)) |>
    pluck("gene_id")

  greatest_change <- averaged |>
    filter(gene_id %in% significant) |>
    arrange(desc(abs_effect)) |>
    slice(1:1000)

  write_tsv(averaged, f)
  write_tsv(effect, here(outdir, "ALDEx2_all_effect.tsv"))
  write_tsv(test, here(outdir, "ALDEx2_test.tsv"))
}

aldex_average_file <- here(outdir, "ALDEx2_averaged_effect.tsv")
## aldex_average <- read_existing(aldex_average_file, get_aldex, read_tsv)
## <2025-02-28 Fri> OOM errors, even on the tiny test dataset...

## * With edgeR

# Goal: finding the top DEGs in each class with one-vs-rest
get_edgeR <- function(f, counts, data, group, technical_factors) {
  dge <- DGEList(
    counts = counts,
    samples = colData(data),
    genes = rowData(data)
  )
  normLibSizes(dge)
  factor_str <- paste0(c(group, technical_factors), collapse = " + ")
  mm <- model.matrix(as.formula(paste("~0+", factor_str)), data = colData(data))
  dge <- estimateDisp(dge, design = mm, robust = TRUE)
  fit <- glmQLFit(dge, mm, robust = TRUE)

  # Make contrasts to get fold changes in one class vs mean of other classes
  group_vec <- colnames(mm) |> keep(\(x) str_detect(x, group))
  mean_val <- 1 / (length(group_vec) - 1)

  contrast_str <- map_chr(group_vec, \(x) {
    mean_others <- paste(
      mean_val,
      "*",
      group_vec[group_vec != x],
      collapse = "+"
    )
    paste0(x, "-", "(", mean_others, ")")
  })

  ccs <- makeContrasts(contrasts = contrast_str, levels = mm)
  test <- glmQLFTest(fit, contrast = ccs)

  top <- topTags(test, n = nrow(dge), sort.by = "PValue") |>
    as.data.frame() |>
    as_tibble()
  fc_names <- paste0("logFC_", group_vec)
  names(fc_names) <- keep(colnames(top), \(x) str_detect(x, "logFC"))
  top <- rename(top, all_of(fc_names))
  # logFC_<x> are the logFC of x vs the mean of all other classes
  write_tsv(top, f)
}

edgeR_top_file <- here(outdir, glue("edgeR_top_types{suffix}.tsv"))
edgeR_top <- read_existing(
  edgeR_top_file,
  \(f) {
    get_edgeR(
      f,
      counts = counts,
      data = data,
      group = GROUP,
      technical_factors = TECHNICAL_FACTORS
    )
  },
  read_tsv
)

## ** Sample-type specific analyses

stypes <- c("primary", "organoid")
for (type in stypes) {
  print(glue("Running lfc analysis for {type}..."))
  mask <- replace_na(colData(data)$Sample_Type == type, FALSE)
  filtered <- data[, mask]
  edgeR_top_organoid <- read_existing(
    here(outdir, glue("edgeR_top_types_{type}")),
    \(f) {
      get_edgeR(
        f = f,
        counts = assays(filtered)$X,
        data = filtered,
        group = GROUP,
        technical_factors = TECHNICAL_FACTORS
      )
    },
    read_tsv
  )
}

## * Calculate aurocs

gene_auroc <- read_existing(
  here(outdir, "gene_auROC_scores.csv"),
  \(f) {
    tmeta <- import("too_predict.meta_markers")
    MM <- tmeta$MetaMarkers(
      datasets = list(main = adata),
      label_col = "tumor_type",
      marker_col = "GENEID"
    )
    MM$add_markers(adata$var$GENEID)
    aurocs <- show_reticulate_error(MM$calc_auroc("main"))
    write_csv(aurocs, f)
  },
  read_csv
)

gene_auroc_organoid <- read_existing(
  here(outdir, "gene_auROC_scores_chula_organoid.csv"),
  \(f) {
    tmeta <- import("too_predict.meta_markers")
    organoids <- adata[str_detect(adata$obs$Project_ID, "CHULA"), ]
    MM <- tmeta$MetaMarkers(
      datasets = list(main = organoids),
      label_col = "tumor_type",
      marker_col = "GENEID"
    )
    MM$add_markers(adata$var$GENEID)
    aurocs <- show_reticulate_error(MM$calc_auroc("main"))
    write_csv(aurocs, f)
  },
  read_csv
)

## * Marker-based
# %%
cell_markers <- markers_meta_internal(grouped = FALSE)
tissues <- unique(cell_markers$tissue)

# Clearly need to consider organoid vs primary
ovp_tb <- read_tsv(here(
  "data",
  "output",
  "chula_organoid_comparison",
  "de_enrichment",
  "sample_type_top_tags.tsv"
))
ovp_blacklist <- ovp_tb |>
  filter(PValue >= 0.01) |>
  pull(GENEID)

tmp <- cell_markers |>
  inner_join(edgeR_top, by = join_by(x$ensembl == y$GENEID)) |>
  filter(!ensembl %in% ovp_blacklist)

hpa_wanted_cols <- c(
  "eg", # Ensembl
  "evih", # HPA evidence
  "rnats", # RNA tissue specificity
  "rnatd", # RNA tissue distribution
  "rnatss", # RNA tissue specificity score
  "rnatsm", # RNA tissue specific nTPM
  "rnacas", # RNA cancer specificity
  "rnacad", # RNA cancer distribution
  "rnacass", # RNA cancer specificity score
  "rnacasm" # RNA cancer specific FPKM
)

# Key to elevated expression levels
## Tissue enriched: At least four-fold higher mRNA level in heart compared to any other tissues.
## Group enriched: At least four-fold higher average mRNA level in a group of 2-5 tissues compared to any other tissue.
## Tissue enhanced: At least four-fold higher mRNA level in heart compared to the average level in all other tissues.

# Prefer group enriched or tissue enhanced

query <- "tissue_category_rna:Bone marrow;Tissue enriched"

tissues_to_get <- c(
  "tongue",
  "stomach",
  "skeletal muscle",
  "heart muscle",
  "intestine",
  "liver",
  "kidney",
  "prostate",
  "breast",
  "adrenal gland",
  "retina",
  "lymphoid tissue",
  "salivary gland",
  "urinary bladder",
  "bone marrow",
  "pancreas",
  "brain",
  "skin",
  "esophagus",
  "testis"
)

tissue2primary_site <- c(
  "tongue" = "mouth_tongue",
  "skeletal muscle" = "bones_joints_articular_cartilage",
  "heart muscle" = "heart_mediastinum_and_pleura",
  "retina" = "eye",
  "urinary bladder" = "bladder",
  "bone marrow" = "hematopoietic_and_reticuloendothelial_systems",
  "salivary gland" = "mouth_tongue",
  "lymphoid tissue" = "lymph_nodes",
  "adrenal gland" = "adrenal gland"
)

hpa_lookup <- slowly(
  \(qstring) {
    hpa_query <- get_hpa_query(
      qstring,
      format = "tsv",
      columns = hpa_wanted_cols,
      compress = "no"
    )
    tryCatch(
      expr = {
        request(hpa_query) |>
          req_perform() |>
          resp_body_string() |>
          fread() |>
          as_tibble()
      },
      error = \(cnd) NULL
    )
  },
  rate_delay(pause = 1)
)


hpa_tissue_file <- here(
  "data",
  "reference",
  "hpa_tissue_enriched_2025-5-20.csv"
)
hpa_tissues <- read_existing(
  hpa_tissue_file,
  \(f) {
    hpa_tissues <- lapply(tissues_to_get, \(t) {
      query_str <- glue(
        "tissue_category_rna:{t};Group enriched,Tissue enhanced"
      )
      tb <- hpa_lookup(query_str)
      if (!is.null(tb)) {
        tb |> mutate(tissue = t)
      }
    }) |>
      bind_rows() |>
      mutate(
        primary_site = tissue2primary_site[tissue],
        primary_site = case_match(
          primary_site,
          NA ~ tissue,
          .default = primary_site
        )
      )
    write_csv(hpa_tissues, f)
    hpa_tissues
  },
  read_csv
)
