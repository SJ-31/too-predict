suppressMessages({
  library(here)
  library(tidyverse)
  library(glue)
  library(edgeR)
  library(paletteer)
  library(reticulate)
  library(fgsea)
  library(evoGO)
  use_condaenv("too-predict")
  source(here("src", "R", "utils.R"))
})

ut <- import("too_predict.utils")
tu <- import("too_predict._train_utils")
plotting <- import("too_predict.plotting")
sc <- import("scanpy")
skc <- import("sklearn.cluster")

GO_ANNOTATIONS <- evoGO::loadGOAnnotation(species = "hsapiens", path = as.character(ut$get_data("")))
PATHWAYS <- pathways_internal()


if (path.expand("~") != "/home/shannc") {
  adata_fn <- function() {
    adata <- ut$training_data_internal()
    adata[grepl("CHULA", adata$obs$Project_ID), ]
  }
} else {
  adata_fn <- function() {
    adata <- ut$training_data_internal_test()
    adata[, 1:100]
  }
}

main <- function(adata, target, outdir, cluster_method, label_col = "tumor_type") {
  target_outdir <- here(outdir, target)
  dir.create(target_outdir)
  filt <- adata[adata$obs[[label_col]] == target, ]
  if (filt$shape[[1]] == 0) {
    print(glue("Filtering for {target} left no samples"))
    return()
  }
  clusters <- show_reticulate_error(paste0("cls", get_clusters(filt, cluster_method)))
  if (length(unique(clusters)) < 2) {
    print("No clusters detected")
    return()
  }
  # Get clusters
  filt$obs$cluster <- clusters
  unique_clusters <- unique(clusters)
  pca_fig <- plotting$plot_adata(filt, y = "cluster", plot_mode = "pca")
  pca_fig$save_fig(here(target_outdir, "cluster_pca.png"))

  umap_fig <- plotting$plot_adata(filt, y = "cluster", plot_mode = "umap")
  umap_fig$save_fig(here(target_outdir, "cluster_umap.png"))

  # Get genes that are DE in each cluster
  alpha <- 0.05
  de_results <- edgeR_mean_all(adata2dge(filt, "counts"), group = "cluster")
  de_results <- lapply(de_results, \(tb) {
    mutate(tb, is_significant = PValue <= alpha)
  }, simplify = FALSE, USE.NAMES = TRUE)

  sig_genes <- sapply(unique_clusters, \(c) {
    de_results[[c]] |>
      filter(PValue <= alpha) |>
      pluck("id") |>
      as.character()
  }, simplify = FALSE, USE.NAMES = TRUE)

  # Identify overrepresented GO terms in all the DE genes for each cluster
  universe <- rownames(filt$var)
  ora_alpha <- 0.05
  overrepresented <- sapply(sig_genes, \(gl) {
    evoGO::calcGOenrichment(GO_ANNOTATIONS,
      deGenes = gl,
      universe = universe
    ) |>
      as_tibble() |>
      filter(evogo.pvalue <= ora_alpha)
  }, simplify = FALSE, USE.NAMES = TRUE)
  lapply(names(overrepresented), \(n) {
    write_tsv(de_results[[n]], here(target_outdir, glue("{n}_edgeR.tsv")))
  })

  # Identify significant downregulated/upregulated reactome pathways and GO BP using fgsea
  fgsea_alpha <- 0.05
  fgsea_results <- sapply(de_results, \(tb) {
    sorted <- tb |>
      filter(PValue <= fgsea_alpha) |>
      arrange(logFC)
    ranked <- setNames(sorted$logFC, sorted$id)
    fgsea(pathways = PATHWAYS, stats = ranked) |> filter(padj <= fgsea_alpha)
  }, simplify = FALSE, USE.NAMES = TRUE)

  lapply(unique_clusters, \(n) {
    write_tsv(de_results[[n]], here(target_outdir, glue("{n}_edgeR.tsv")))
    write_tsv(overrepresented[[n]], here(target_outdir, glue("{n}_evoGO.tsv")))
    write_tsv(fgsea_results[[n]], here(target_outdir, glue("{n}_fgsea.tsv")))
  })
}


get_clusters <- function(adata, layer = NULL, method = "HDBSCAN", ...) {
  if (method == "HDBSCAN") {
    hdb <- skc$HDBSCAN(...)
    hdb$fit_predict(adata$X)
  } else if (method == "leiden") {
    sc$pp$pca(adata)
    sc$pp$neighbors(adata)
    sc$tl$umap(adata)
    sc$tl$leiden(adata, ...)
    adata$obs$leiden
  } else if (method == "test") {
    sample(c(1, 2, 3), size = adata$shape[[1]], replace = TRUE)
  }
}


if (sys.nframe() == 0) {
  library("optparse")
  parser <- OptionParser()
  parser <- add_option(parser, c("-l", "--label_col"), type = "character", default = "tumor_type")
  parser <- add_option(parser, c("-g", "--targets"), type = "character", default = NULL)
  parser <- add_option(parser, c("-c", "--cluster_method"), type = "character", default = "HDBSCAN")
  args <- parse_args(parser)
  adata <- adata_fn()
  adata$obs$tumor_type <- str_replace_all(adata$obs$tumor_type, "-", "_")

  SPEC <- tu$read_model_spec(tu$MODELS[["clr_xgb3_edger"]])
  transformer <- SPEC[[3]]
  transformed <- transformer$fit_transform(adata)
  outdir <- here("data", "output", "explanations", "subtypes")
  dir.create(outdir)

  targets <- str_split_1(args$targets, ",")
  for (t in targets) {
    filtered <- adata[adata$obs[[args$label_col]] == t, ]

    main(transformed,
      target = t, outdir = outdir, label_col = args$label_col,
      cluster_method = args$cluster_method
    )
  }
}
