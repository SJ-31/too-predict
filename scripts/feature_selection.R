library(BiocParallel)
library(ALDEx2)
library(tidyverse)
library(zellkonverter)
library(scRNAseq)
library(edgeR)
library(here)

register(MulticoreParam(workers = 2))
source(here("src", "R", "utils.R"))

data <- readH5AD(here("data", "tests", "TCGA_CESC-DLBC-ESCA-GBM.h5ad"))
data <- data[1:50, ]
# TODO: replace this with the complete dataset
outdir <- here("data", "output", "feature_selection")

# TODO: can include the sequencing tech and the tumor type as factors to account
# for their effects
## --- CODE BLOCK ---
p_threshold <- 0.05
group <- "Project_ID"
technical_factors <- c("Sample_Type")

colData(data)$Project_ID <- str_replace(colData(data)$Project_ID, "-", ".")
colData(data)$Sample_Type <- str_replace(colData(data)$Sample_Type, " ", "_")

counts <- assays(data)$X |> `rownames<-`(rowData(data)[, 1])

## * DGE with ALDEx2

get_aldex <- function(f) {
  result <- aldex_glm_wrapper(data, group, technical_factors, use_parallel = TRUE)
  # TODO: maybe use the scale aware version by specifying gamma
  effect <- as_tibble(result$effect) # Effect size are standardized mean differences
  test <- as_tibble(result$test)

  # Take the average effects of all comparisons
  between_groups <- effect |> filter(str_starts(contrast, group))
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
    select(gene_id, contains(group) & contains("pval.padj")) |>
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

## * With edgeR

# Goal: finding the top DEGs in each class
get_edgeR <- function(f) {
  dge <- DGEList(counts = assays(data)$X, samples = colData(data), genes = rowData(data))
  normLibSizes(dge)
  factor_str <- paste0(c(group, technical_factors), collapse = " + ")
  mm <- model.matrix(as.formula(paste("~0+", factor_str)), data = colData(data))
  dge <- estimateDisp(dge, mm, robust = TRUE)
  fit <- glmQLFit(dge, mm, robust = TRUE)
  # Fit glm to account for batch effect specified above

  # Make contrasts to get fold changes in one class vs mean of other classes
  group_vec <- colnames(mm) |> keep(\(x) str_detect(x, group))
  mean_val <- 1 / (length(group_vec) - 1)

  contrast_str <- map_chr(group_vec, \(x) {
    mean_others <- paste(mean_val, "*", group_vec[group_vec != x], collapse = "+")
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

edgeR_top_file <- here(outdir, "edgeR_top_types.tsv")
## edgeR_top <- read_existing(edgeR_top_file, get_edgeR, read_tsv)

## * With Seurat
# <2025-02-21 Fri> isn't too useful since it doesn't give you the values used to
# determine top features
## obj <- SeuratObject::CreateSeuratObject(assays(data)$X, meta.data = colData(data))
## obj <- Seurat::AddMetaData(obj[["RNA"]], rowData(data))
## obj <- Seurat::NormalizeData(obj)
## obj <- Seurat::FindVariableFeatures(obj)
## top <- Seurat::VariableFeatures(obj)
## Seurat::VariableFeaturePlot(obj)
## seurat_top_file <- here(outdir, )
