suppressMessages({
  library(BiocParallel)
  library(edgeR)
  library(here)
  library(reticulate)
  use_condaenv("too-predict")
  library(tidyverse)
  library(Seurat)
  library(DESeq2)
  library(zellkonverter)
  library(scRNAseq)
  source(here("src", "R", "utils.R"))
  source(here("src", "R", "plotting.R"))
})

tdir <- here("data", "tests")
scrna <- here(tdir, "scr_ref")
ut <- import("too_predict.utils")
ad <- import("anndata")

adata <- ut$training_data_internal_test(minimal = TRUE)


brca <- adata[adata$obs$tumor_type == "BRCA", ]

n_groups <- nlevels(adata$obs$tumor_type)

obj <- as.matrix(t(adata$X))
colnames(obj) <- rownames(adata$obs)
rownames(obj) <- rownames(adata$var)

mm <- model.matrix(~ 0 + tumor_type, data = adata$obs)


sample <- here(
  "data",
  "output",
  "tests",
  "deep",
  "holdout",
  "2025-08-22-",
  "holdout_summary.csv"
)

task <- "tumor_type"

holdout <- read_csv(sample)

results <- holdout

plot_metric_variation <- function(tb) {
  tb |>
    ggplot(aes(x = model, y = value, color = metric)) +
    geom_boxplot() +
    facet_wrap(~task)
}

unique_metrics <- results

holdout |>
  filter(metric == "acc", task == "tumor_type") |>
  friedman_test_wrapper("metric", "split")
