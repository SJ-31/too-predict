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
  if (length(sample_annotations) > 0) {
    sample_anno <- obs[, colnames(obs) %in% names(sample_annotations)] |> as.data.frame()
    if (nchar(order_on) > 0) {
      vec <- obs[[order_on]]
      uniques <- unique(vec)
      sample_anno[[order_on]] <- factor(vec, levels = uniques)
      counts <- counts[, order(sample_anno[[order_on]])]
      sample_anno <- sample_anno[order(sample_anno[[order_on]]), ]
    }
  } else {
    sample_anno <- NA
  }

  do.call(\(...) {
    pheatmap(counts,
      cluster_rows = FALSE, cluster_cols = FALSE,
      annotation_col = sample_anno, annotation_colors = anno_colors, ...
    )
  }, pheatmap_kwargs)
}

# BUG: this isn't working properly [2025-05-22 Thu]
plot_confusion_matrix <- function(cm, x = "x", y = "y", v = "value",
                                  diagonal = TRUE, null_zeros = TRUE,
                                  x_label = NULL, y_label = NULL,
                                  show_counts = TRUE,
                                  only_misses = TRUE,
                                  fill_label = NULL,
                                  na_color = "grey",
                                  palette = "ggthemes::Green") {
  if (diagonal) {
    cm <- distinct_orderings(cm, c(x, y))
  }
  if (only_misses) {
    uniques <- unique(cm[[x]])
    no_misses <- uniques |> discard(\(u) {
      filtered <- filter(cm, !!as.symbol(x) == u | !!as.symbol(y) == u) |>
        filter(!!as.symbol(v) > 0)
      nrow(filtered) > 1
    })
    cm <- filter(cm, !(!!as.symbol(x) %in% no_misses | !!as.symbol(y) %in% no_misses))
  }
  if (null_zeros) {
    replaced <- case_match(cm[[v]], 0 ~ NA, .default = cm[[v]])
    cm[[v]] <- replaced
  }
  plot <- ggplot(cm, aes(
    x = as.factor(!!as.symbol(x)), y = as.factor(!!as.symbol(y)),
    fill = !!as.symbol(v)
  )) +
    geom_tile() +
    theme_minimal() +
    theme(
      panel.grid = element_blank(),
      axis.text.x = element_text(angle = 90, vjust = 0.5)
    )
  if (show_counts) {
    plot <- plot + geom_text(aes(label = !!as.symbol(v)))
  }
  if (!is.null(x_label)) {
    plot <- plot + xlab(x_label)
  }
  if (!is.null(y_label)) {
    plot <- plot + ylab(y_label)
  }
  if (!is.null(fill_label)) {
    plot <- plot + guides(fill = guide_legend(fill_label))
  }
  if (diagonal) {
    plot <- plot + scale_x_discrete(position = "top")
  }
  plot + scale_fill_paletteer_c(palette = palette, na.value = na_color)
}


plot_lfc <- function(x, y, label_col, tag_tb, dge, cpm = NULL,
                     p_value = 0.05, subset = NULL, subset_col = "tumor_type") {
  gene_mask <- tag_tb$PValue <= p_value
  tag_tb <- tag_tb[gene_mask, ]
  if (!is.null(cpm)) {
    stopifnot("Dimensions of cpm and dge must be equal" = dim(cpm) == dim(dge))
  } else {
    cpm <- edgeR::cpm(dge)
  }
  if (!is.null(subset)) {
    dge <- dge[, dge$samples[[subset_col]] %in% subset]
  }
  dge <- dge[gene_mask, ]
  cpm <- cpm[gene_mask, ]
  x_vec <- cpm[, dge$samples[[label_col]] == x] |> rowMeans()
  y_vec <- cpm[, dge$samples[[label_col]] == y] |> rowMeans()
  count_tb <- tibble(
    x = log(x_vec), y = log(y_vec), lfc = tag_tb$logFC,
    pvalue = tag_tb$PValue
  )
  ggplot(count_tb, aes(x = x, y = y, color = lfc)) +
    geom_point() +
    xlab(glue("mean log cpm {x}")) +
    ylab(glue("mean log cpm {y}")) +
    geom_abline(slope = 1, intercept = 0, alpha = 0.5)
}


volcano_plot <- function(tag_tb, label_col = "GENENAME", fdr_cutoff = 0.001) {
  tag_tb |>
    mutate(
      `Significant FDR` = case_when(
        FDR < fdr_cutoff ~ "Yes",
        .default = "No"
      ),
      !!as.symbol(label_col) := case_when(!is.na(!!as.symbol(label_col)) ~ !!as.symbol(label_col),
        .default = ""
      ),
      delabel = case_when(FDR < fdr_cutoff ~ !!as.symbol(label_col),
        .default = NA
      )
    ) |>
    ggplot(aes(x = logFC, -log10(FDR), color = `Significant FDR`, label = delabel)) +
    geom_point(size = 1) +
    ggrepel::geom_text_repel(size = 1.5) +
    labs(x = "log fold change", y = "-log10(adjusted p-value)") +
    theme_bw() +
    guides(color = "none") +
    scale_color_manual(values = c("black", "red"))
}


#' Helper for getting enrichment plots of all gene sets in
#'    `fgsea_df`, placing the images in plotdir
#'
plot_fgsea_gseavis <- function(fgsea_df, ranks, gene_sets, plotdir,
                               suffix = "_enrichment.png") {
  fgsea_df$id <- fgsea_df$pathway
  converted <- fgsea_result2gseGO(fgsea_df, id_col = "id")

  en_res <- GseaVis::dfGO2gseaResult(
    enrich.df = converted, geneList = sort(ranks, decreasing = TRUE),
    own_termSet = gene_sets, setType = "ALL"
  )
  tmp <- lapply(fgsea_df$pathway, \(pwy) {
    name <- str_replace_all(pwy, " ", "_") |> str_replace_all("/", "-")
    plot <- enrichplot::gseaplot2(en_res,
      geneSetID = pwy, title = pwy,
      pvalue_table = TRUE
    )
    ggsave(here(plotdir, glue("{name}{suffix}")), plot, height = 8, width = 12)
  })
}
