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
sc <- import("scanpy")
ad <- import("anndata")
skc <- import("sklearn.cluster")

GO_ANNOTATIONS <- evoGO::loadGOAnnotation(species = "hsapiens", path = as.character(ut$get_data("")))

PATHWAYS <- local({
  go_data <- as.character(ut$get_data("ensembl_go_map_2025-3-20.tsv")) |> read_tsv()
  bp_mapping <- local({
    tb <- go_data |>
      filter(`GO domain` == "biological_process" & !is.na(`GO term name`)) |>
      mutate(`GO term name` = str_replace_all(paste0("GO_BP:", `GO term name`), " ", "_"))
    group_by_into_list(tb, "GO term name", "Gene stable ID")
  })
  ensembl2reactome <- as.character(ut$get_data("ensembl2reactome_2025-4-11.tsv")) |> read_tsv()
  reactome <- as.character(ut$get_data("ReactomePathways_2025-4-11.txt")) |>
    read_tsv(col_names = c("Reactome ID", "name", "species"))
  reactome_mapping <- ensembl2reactome |>
    inner_join(reactome, by = join_by(`Reactome ID`)) |>
    mutate(name = str_replace_all(paste0("Reactome:", name), " ", "_")) |>
    group_by_into_list("name", "Gene stable ID")
  c(bp_mapping, reactome_mapping)
})


if (path.expand("~") != "/home/shannc") {
  adata_fn <- function() ut$training_data_internal()
} else {
  adata_fn <- function() {
    adata <- ut$training_data_internal_test()
    adata[, 1:100]
  }
}

main <- function(adata, transformer, target, outdir, label_col = "tumor_type") {
  target_outdir <- here(outdir, target)
  dir.create(target_outdir)
  adata <- transformer$fit_transform(adata)
  filtered <- adata[adata$obs[[label_col]] == target, ]
  counts <- filtered$X
  clusters <- get_clusters(counts, "HDBSCAN") %>% paste0("cls", .)
  if (length(unique(clusters)) < 2) {
    print("No clusters detected")
    q()
  }
  # Get clusters
  filt$obs$cluster <- clusters
  unique_clusters <- unique(clusters)

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
      filter(PValue <= alpha) |>
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


get_clusters <- function(adata, method = "HDBSCAN", ...) {
  if (method == "HDBSCAN") {
    hdb <- skc$HDBSCAN(...)
    hdb$fit_predict(adata$X)
  } else if (method == "leiden") {
    sc$pp$neighbors(adata)
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
  parser <- add_option(parser, c("-t", "--target"), type = "character", default = NULL)
  parser <- add_option(parser, c("-p", "--plot_only"), type = "character", default = FALSE, action = "store_true")
  args <- parse_args(parser)
  adata <- adata_fn()
  SPEC <- tu$read_model_spec(tu$MODELS[["clr_xgb3_edger"]])
  transformer <- SPEC[[3]]
  outdir <- here("data", "output", "explanations", "subtypes")
  dir.create(outdir)
  if (!args$plot_only) {
    main(adata,
      transformer = transformer, target = args$target, outdir = outdir,
      label_col = args$label_col
    )
  }
}
