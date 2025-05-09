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


## adata <- ut$training_data_internal_test()
## adata <- adata[, 1:200]

obj <- readRDS(here(scrna, "htca_2025-4-21", "HTCA_ADULT_ADRENAL_GLAND.rds"))

## scgen <- import("scgen")

## adata <- ad$AnnData(X = t(LayerData(obj)), var = obj[["RNA"]][[]], obs = obj[[]])

## scgen$SCGEN$setup_anndata(adata, batch_key = "Sample_ID", labels_key = "Cell_Type")

## model <- scgen$SCGEN(adata)
## model$train(
##   max_epochs = 3,
##   batch_size = 32,
##   early_stopping = TRUE,
##   early_stopping_patience = 25,
## )
