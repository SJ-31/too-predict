suppressMessages({
  library(here)
  library(tidyverse)
  library(glue)
  library(edgeR)
  library(paletteer)
  library(reticulate)
  use_condaenv("too-predict")
  source(here("src", "R", "utils.R"))
})

tf <- import("too_predict.filter")
ut <- import("too_predict.utils")
ad <- import("anndata")
go_annotations <- evoGO::loadGOAnnotation(species = "hsapiens", path = as.character(ut$get_data("")))
pathways <- pathways_internal()

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
  adata_fn <- function() ut$training_data_internal()
} else {
  ## test_file <- here(output())
  adata_fn <- function() {
    ad$read_h5ad(here(
      "data", "output", "normalization_comparison",
      "edgeR_median_lfc_feature_list_3000", "none-plus_one.h5ad"
    ))
  }
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
  chol = makeContrasts("CHOL.organoid - CHOL.primary", levels = mm)
)

# Explanation
# 1 highlights the differences between primary and organoid samples,
#   while adjusting for tumor-type specific differences
# The other contrasts find DE genes between sample types within a specific tumor type

dge <- normLibSizes(dge)
dge <- estimateDisp(dge, design = mm, robust = TRUE)

added_cols <- c("logFC", "logCPM", "F", "PValue", "FDR")
other_cols <- colnames(dge$genes)

fit <- glmQLFit(dge, mm)

top_tags <- sapply(contrasts, \(n) {
  qlf <- glmQLFTest(fit, contrast = contrasts[[n]])
  message(glue("Comparison: {qlf$comparison}"))
  get_tags(qlf, count = nrow(dge)) |> rename_with(\(cols) {
    map_chr(cols, \(x) {
      if (x %in% added_cols) {
        glue("{n}_{x}")
      } else {
        x
      }
    })
  })
}, simplify = FALSE, USE.NAMES = TRUE)

# TODO: use plots to make sure that you interpret the lfc direction for the
# `sample_type` contrast correctly

## * Plots

## * Enrichment analyses


## * Cross-reference with markers
# TODO: need to find marker sets
