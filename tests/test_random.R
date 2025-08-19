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

adata$X <- as.matrix(adata$X)
sce <- zellkonverter::AnnData2SCE(adata)

library(splatter)

params <- splatEstimate(sce)
sim <- splatSimulate(params, batchCells = 100)
