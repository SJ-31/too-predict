lfc_filter <- function(tag_tb) {
  lfc_threshold <- 0 # TODO: any way to choose a good threshold?
  alpha <- 0.05
  fdr_cutoff <- 0.05
  tag_tb |> filter(abs(logFC) >= lfc_threshold & PValue <= alpha & FDR <= fdr_cutoff)
}

fgsea_helper <- function(gene_sets, alpha = 0.05, plotdir, meta = NULL) {
  library(fgsea)
  fgsea_results <- sapply(names(top_tags), \(n) {
    tb <- top_tags[[n]]
    sorted <- lfc_filter(tb) |> arrange(logFC)
    warning(glue("There are {sum(is.na(sorted$GENEID))} genes with missing ENSEMBL ids!"))
    sorted <- filter(sorted, !is.na(GENEID))
    ranked <- setNames(sorted$logFC, sorted$GENEID)
    result <- fgsea(pathways = gene_sets, stats = ranked)
    result <- result[result$padj <= alpha, ]

    ndir <- here(plotdir, n)
    dir.create(ndir)
    plot_fgsea_gseavis(result, ranks = ranked, gene_sets = gene_sets, plotdir = ndir)

    avg_lfc <- gene_set_average(gene_sets = gene_sets, reference = tb) |> rename(mean_lfc = average)
    fgsea_tb <- result |>
      mutate(leadingEdge = map_chr(leadingEdge, \(x) paste0(x, collapse = ";"))) |>
      inner_join(avg_lfc, by = join_by(x$pathway == y$set_name))
    if (!is.null(meta)) {
      fgsea_tb |> left_join(select(meta, set_name, category), by = join_by(x$pathway == y$set_name))
    } else {
      fgsea_tb
    }
  }, simplify = FALSE, USE.NAMES = TRUE)
  fgsea_results
}
