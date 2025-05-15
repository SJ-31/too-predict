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


adata <- ut$training_data_internal_test()
adata <- adata[, 1:200]

stype <- adata$obs$Sample_Type
stype <- replace(stype, is.na(stype), "organoid")
