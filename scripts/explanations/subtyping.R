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

ut <- import("too_predict.utils")
tu <- import("too_predict._train_utils")
sc <- import("scanpy")
ad <- import("anndata")
skc <- import("sklearn.cluster")

if (path.expand("~") != "/home/shannc") {
  adata_fn <- function() ut$training_data_internal()
} else {
  adata_fn <- function() {
    adata <- ut$training_data_internal_test()
    adata[, 1:100]
  }
}
outdir <- here("data", "output", "explanations", "subtypes")
dir.create(outdir)

if (sys.nframe() == 0) {
  library("optparse")
  parser <- OptionParser()
  args <- parse_args(parser)
}

get_clusters <- function(adata, method = "HDBSCAN", ...) {
  if (method == "HDBSCAN") {
    hdb <- skc$HDBSCAN(...)
    hdb$fit_predict(adata$X)
  } else if (method == "leiden") {
    sc$pp$neighbors(adata)
    sc$tl$leiden(adata, ...)
    adata$obs$leiden
  } else if (method == "test") {
    sample(c(1, 2, 3, 4), size = adata$shape[[1]], replace = TRUE)
  }
}


main <- function(adata, transformer, target, label_col = "tumor_type") {
  adata <- transformer$fit_transform(adata)
  filtered <- adata[adata$obs[[label_col]] == target, ]
  counts <- filtered$X
  clusters <- get_clusters(counts, "HDBSCAN")
  if (length(unique(clusters)) < 2) {
    print("No clusters detected")
    q()
  }
}

target <- "BRCA"
label_col <- "tumor_type"
SPEC <- tu$read_model_spec(tu$MODELS$clr_xgb3_edger)

adata <- adata_fn()
transformed <- SPEC[[3]]$fit_transform(adata)
filt <- transformed[transformed$obs[[label_col]] == target, ]

# Get clusters
clusters <- get_clusters(filt, "test") %>% paste0("cls", .)
filt$obs$cluster <- clusters
unique_clusters <- unique(clusters)

# Want the genes that are DE in each cluster
de_results <- edgeR_mean_all(adata2dge(filt, "counts"), group = "cluster")

alpha <- 0.001
sig_genes <- sapply(unique_clusters, \(c) {
  de_results[[c]] |>
    filter(PValue <= alpha) |>
    pluck("id")
}, simplify = FALSE, USE.NAMES = TRUE)
