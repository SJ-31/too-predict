library(ggplot2)
library(pheatmap)
library(paletteer)

plot_roc <- function(df,
                     file = "",
                     tpr_col = "tpr", fpr_col = "fpr",
                     threshold_col = "thresholds",
                     class_col = "class") {
  plot <- ggplot(df, aes(
    x = !!as.symbol(fpr_col),
    y = !!as.symbol(tpr_col),
    color = !!as.symbol(class_col),
    alpha = !!as.symbol(threshold_col)
  )) +
    geom_line() +
    xlab("FPR") +
    ylab("TPR")
  ggsave(file, plot)
}

dist_index <- function(dist, i, j) {
  labels <- attr(dist, "Labels")
  if (is.character(i)) {
    i <- match(i, labels)
  }
  if (is.character(j)) {
    j <- match(j, labels)
  }
  n <- attr(dist, "Size")
  if (j == 1) {
    NA
  } else if (j < i) {
    dist[(2 * n - j) * (j - 1) / 2 + (i - j)]
  } else {
    dist[n * (i - 1) - i * (i - 1) / 2 + j - i]
  }
}

#' TODO: this doesn't work
#' Convert `dist` object into a long-form tibble as lower-triangular
#'
dist2tb <- function(dist, colname = "name", value = "value") {
  n <- seq_len(attr(dist, "Size"))
  labels <- attr(dist, "Labels")
  vals <- list()
  vals[[colname]] <- c()
  vals[["y"]] <- c()
  vals[[value]] <- c()
  if (is.null(labels)) {
    labels <- n
  }
  lapply(n, \(i) {
    lapply(n, \(j) {
      if (i < j) {
        j_l <- labels[j]
        i_l <- labels[i]
        # <2025-03-03 Mon> You need to assign it to the correct column somehow
        ## if (i_l %in% vals[[colname]]) {
        ##   vals[[colname]] <<- c(vals[[colname]], j_l)
        ##   vals[["y"]] <<- c(vals[["y"]], i_l)
        ## } else {
        vals[[colname]] <<- c(vals[[colname]], i_l)
        vals[["y"]] <<- c(vals[["y"]], j_l)
        ## }
        vals[[value]] <<- c(vals[[value]], dist_index(dist, i, j))
      }
    })
  })
  as_tibble(vals)
}

get_color_lists <- function(vector, colormap, rep = TRUE) {
  uniques <- unique(vector)
  not_enough <- length(colormap) < length(uniques)
  if (length(colormap) > length(uniques)) {
    colormap <- colormap[seq_along(uniques)]
  }
  if (not_enough && !rep) {
    errorCondition("Number of levels exceeds number of provided colors")
  } else if (not_enough) {
    warning("Number of levels exceeds number of provided colors")
  }
  if (not_enough && rep) {
    colormap <- c(colormap, sample(colormap, length(uniques) - length(colormap), replace = TRUE))
  }
  setNames(colormap, uniques)
}

pheatmap_helper <- function(obs = NULL,
                            counts = NULL,
                            sce = NULL,
                            sample_annotations = list(),
                            color_default = NULL,
                            count_assay = "X",
                            order_on = "",
                            pheatmap_kwargs = list()) {
  if (is.null(counts) && is.null(sce)) {
    errorCondition("One of counts or sce must be provided!")
  }
  if (is.null(counts) && !is.null(sce)) {
    counts <- assays(sce)[[count_assay]]
  }
  if (is.null(color_default)) {
    default_cmap <- grDevices::colors()[grep("gr(a|e)y", grDevices::colors(), invert = TRUE)]
  } else if (is.character(color_default)) {
    default_cmap <- paletteer_d(color_default)
  }
  if (is.null(obs) && !is.null(sce)) {
    obs <- colData(sce)
  }
  anno_colors <- sapply(names(sample_annotations), \(x) {
    if (!is.null(sample_annotations[[x]])) {
      if (is.character(sample_annotations[[x]])) {
        cmap <- paletteer_d(sample_annotations[[x]])
      } else {
        cmap <- sample_annotations[[x]]
      }
      get_color_lists(obs[[x]], cmap)
    } else {
      get_color_lists(obs[[x]], default_cmap)
    }
  }, simplify = FALSE)

  sample_anno <- obs[, colnames(obs) %in% names(sample_annotations)] |> as.data.frame()
  if (nchar(order_on) > 0) {
    vec <- obs[[order_on]]
    uniques <- unique(vec)
    sample_anno[[order_on]] <- factor(vec, levels = uniques)
    counts <- counts[, order(sample_anno[[order_on]])]
    sample_anno <- sample_anno[order(sample_anno[[order_on]]), ]
  }

  do.call(\(...) {
    pheatmap(counts,
      cluster_rows = FALSE, cluster_cols = FALSE,
      annotation_col = sample_anno, annotation_colors = anno_colors, ...
    )
  }, pheatmap_kwargs)
}
