read_existing <- function(filename, expr, read_fn = identity()) {
  if (file.exists(filename)) {
    read_fn(filename)
  } else {
    expr(filename)
  }
}

get_obs <- function() {
  read_csv(here("data", "training_data_obs.csv"))
}

aldex_glm_wrapper <- function(data, group, technical_factors = c(),
                              clr_args = list(
                                mc.samples = 16, denom = "all",
                                verbose = TRUE
                              ),
                              glm_args = list(fdr.method = "holm"),
                              gene_col = "gene_id",
                              count_slot = "X",
                              use_parallel = FALSE) {
  library(ALDEx2)
  library(glue)

  if (is.null(rowData(data))) {
    warning("No rowdata available")
    vars <- NA
  }
  if (!is.null(rowData(data)) && !(gene_col %in% colnames(rowData(data)))) {
    warning(glue("Column for gene ids {gene_col} not found! Taking first column"))
    vars <- rowData(data)[, 1]
  } else {
    vars <- rowData(data)[, gene_col]
  }
  clr_args$useMC <- use_parallel
  counts <- assays(data)[[count_slot]]
  var_col <- ifelse(!is.null(gene_col), gene_col, "var")
  rownames(counts) <- vars
  mm <- make_mm(group, technical_factors, colData(data))
  clr <- do.call(\(...) aldex.clr(counts, mm, ...), clr_args)
  test <- do.call(\(...) aldex.glm(clr, mm, ...), glm_args)
  effect <- aldex.glm.effect(clr, useMC = use_parallel)
  combined_effect <- lapply(names(effect), \(x) {
    df <- effect[[x]]
    df[[var_col]] <- rownames(df)
    df$contrast <- x
    df
  }) |> dplyr::bind_rows()
  test[[var_col]] <- rownames(test)
  rownames(test) <- NULL
  rownames(combined_effect) <- NULL
  return(list(effect = as.data.frame(combined_effect), test = as.data.frame(test)))
}

make_mm <- function(group, other_factors = c(), data) {
  factor_str <- paste0(c(group, other_factors), collapse = " + ")
  mm <- model.matrix(as.formula(paste("~0+", factor_str)), data)
  mm
}

basename_no_ext <- function(file) {
  bname <- basename(file)
  helper <- function(b) {
    splits <- b |> str_split_1("\\.")
    if (length(splits) > 1) {
      paste0(head(splits, n = -1), collapse = ".")
    } else {
      b
    }
  }
  map_chr(bname, helper)
}

#' Wrapper for the Friedman test
#'
#' @description
#' In addition to the original test statistic Friedman's chi-square,
#' also calculates an alternative, less conservative statistic provided by [1]
#' and its p-value
friedman_test_wrapper <- function(tb, metric_col, with_class = TRUE, var = "fold") {
  j <- length(unique(tb[[var]]))
  k <- length(unique(tb$model))
  test <- friedman.test(tb[[metric_col]], groups = tb$model, blocks = tb[[var]])
  # Ranks are arranged with the blocks
  # `groups` are the treatments we want to compare
  f_chi <- test$statistic
  f_alt <- ((j - 1) * f_chi) / j * (k - 1) - f_chi # Alternative test statistic
  df1 <- k - 1
  df2 <- (k - 1) * (j - 1)
  test$statistic_alt <- f_alt
  # Two-sided p-value
  test$p_value_alt <- pf(f_alt, df1, df2, lower.tail = FALSE) + pf(-f_alt, df1, df2) |> `names<-`(NULL)
  test
}

empty_tibble <- function(headers, init = "") {
  tmp <- matrix(init, nrow = 1, ncol = length(headers))
  colnames(tmp) <- headers
  tib <- as_tibble(tmp)
  dplyr::slice(tib, 2)
}

table2tb <- function(table, row_header) {
  table |>
    as.data.frame() |>
    `colnames<-`(c(row_header, "Var2", "Freq")) |>
    pivot_wider(names_from = Var2, values_from = Freq)
}

#' Helper function for performing a series of pairwise tests and collecting
#' the results in a tidy format
#'
#' @param groups vector of groups to test against
#' @param values vector of values, the same length as groups
#' @param test_fn a hypothesis testing function f(x, y)
#' @param adjust_fn function to correct p-values, takes a vector of unadjusted p-values
#'    as input
tidy_pairwise <- function(groups, values, test_fn, adjust_fn) {
  pairs <- unique(groups) |> combn(2)
  tb <- tibble(group = groups, v = values)
  apply(pairs, 2, \(p) {
    x <- filter(tb, group == p[1]) |> pluck("v")
    y <- filter(tb, group == p[2]) |> pluck("v")
    test_fn(x, y) |>
      tidy() |>
      mutate(x = p[1], y = p[2])
  }) |>
    bind_rows() |>
    mutate(p_adjust = adjust_fn(p.value))
}

#' Filter out distinct combinations of values in `cols` from
#' long-form tibble `tb`
#'
distinct_orderings <- function(tb, cols) {
  lists <- apply(tb, 1, \(x) {
    sort(sapply(cols, \(c) x[c]))
  }) |> t()
  tb[["tmp"]] <- lists
  tb |>
    distinct(tmp, .keep_all = TRUE) |>
    select(-tmp)
}

adata2eset <- function(adata, layer = NULL, convert_na = FALSE) {
  if (!is.null(layer)) {
    counts <- t(as.matrix(adata$layers[[layer]]))
  } else {
    counts <- t(as.matrix(adata$X))
    if (convert_na) {
      counts <- replace(counts, is.na(counts), 0)
      stopifnot("Removing any counts failed" = !any(is.na(counts)))
    }
  }
  rownames(counts) <- rownames(adata$var)
  colnames(counts) <- rownames(adata$obs)
  pdata <- Biobase::AnnotatedDataFrame(adata$obs)
  fdata <- Biobase::AnnotatedDataFrame(adata$var)
  Biobase::ExpressionSet(assayData = counts, phenoData = pdata, featureData = fdata)
}


adata2dge <- function(adata, layer = NULL, convert_na = FALSE) {
  if (!is.null(layer)) {
    counts <- t(as.matrix(adata$layers[[layer]]))
  } else {
    counts <- t(as.matrix(adata$X))
    if (convert_na) {
      counts <- replace(counts, is.na(counts), 0)
      stopifnot("Removing any counts failed" = !any(is.na(counts)))
    }
  }
  DGEList(counts = counts, samples = adata$obs, genes = adata$var)
}

sce2dge <- function(sce, count_assay = "X", as_integer = FALSE) {
  counts <- assays(sce)[[count_assay]] |> as.matrix()
  if (as_integer) {
    mode(counts) <- "integer"
  }
  DGEList(counts = counts, samples = colData(sce), genes = rowData(sce))
}

#' Helper function to determine highest-ranked member of a column
#' according to multiple metrics
#'
#' @param tb wide-form tibble in terms of metrics
#' @param group_col column of `tb` containing the different groups to compare
#' @param var_col column of `tb` containing representing that group's performance
#'    under different scenarios e.g. different folds of cross validation
#' @param metric_defs A list mapping the name of a metrics in `tb` to logicals
#'    Are TRUE if higher values are better for the given metric
rank_by_metrics <- function(group_col, var_col = NULL, tb, metric_defs) {
  make_score_tb <- function(winner) {
    tmp <- sapply(unique(tb[[group_col]]), \(x) 0, simplify = FALSE)
    tmp[[winner]] <- 1
    as_tibble(tmp)
  }
  if (is.null(var_col)) {
    var_col <- "run"
    tb[[var_col]] <- "1"
  }
  score_tracker <- empty_tibble(c("winner", "metric", var_col))
  rank_scores <- lapply(unique(tb[[var_col]]), \(f) {
    current <- filter(tb, !!as.symbol(var_col) == f)
    lapply(names(metric_defs), \(m) {
      if (metric_defs[[m]]) {
        sorted <- arrange(current, desc(!!as.symbol(m)))
      } else {
        sorted <- arrange(current)
      }
      winner <- head(sorted, n = 1) |> pluck(group_col)
      score_tracker <<- add_row(score_tracker,
        winner = winner, metric = m, !!as.symbol(var_col) := f
      )
      make_score_tb(winner)
    }) |>
      bind_rows() |>
      colSums()
  }) |>
    bind_rows() |>
    colSums()
  score_tracker_table <- table(score_tracker$winner, score_tracker$metric) |>
    table2tb(group_col)
  list(top = rank_scores, table = score_tracker_table)
}

split_path <- function(x) if (dirname(x) == x) x else c(basename(x), split_path(dirname(x)))

#' Transpose a dataframe or tibble while explicitly specifying column names
#'
#' @param colnames either a vector of column names, or the index/name
#' of the new column names in the df
transpose <- function(df, colnames = 1) {
  if (length(colnames) != ncol(df) && is.numeric(colnames)) {
    tmp <- colnames
    colnames <- df[, colnames]
    df <- df[, -tmp]
  } else if (length(colnames) != ncol(df)) {
    tmp <- colnames
    colnames <- df[[colnames]]
    df[[tmp]] <- NULL
  }
  t(df) |> `colnames<-`(colnames)
}

#' Determine a Bonferroni-adjusted cutoff for the proportionality coefficient
#'
#' @description
#' @param N number of samples
#' @param alpha desired significance level
#' @param D number of features
find_prop_cutoff <- function(N, alpha, D, one_tailed = TRUE) {
  sd <- 1 / sqrt(N - 3)
  z_alpha <- qnorm(alpha / (D * (D - 1)), lower.tail = !one_tailed) # one-tailed test if you only
  # consider positive correlations
  z_cutoff <- sd * z_alpha
  tanh(z_cutoff)
}


#' Helper function for identifying DE genes between train vs. test splits
#' for the `CompareLFC` class
#'
#' @description
#' The form is an additive model, made to compare the effect of the split between
#' samples of the same label e.g. tumor type
#' The fewer DE genes between splits, the better because it indicates that the split
#' variable/method does not induce variation between samples within the label
edgeR_lfc_train_test <- function(counts, obs_meta, var_meta, label) {
  library(edgeR)
  dge <- DGEList(counts = counts, samples = obs_meta, genes = var_meta)
  dge$samples[[label]] <- as.factor(dge$samples[[label]])
  mm <- model.matrix(as.formula(paste("~0+", label, "+usage")), data = dge$samples)
  dge <- normLibSizes(dge)
  dge <- estimateDisp(dge, design = mm)
  fit <- glmQLFit(dge, design = mm)
  test <- glmQLFTest(fit)
  topTags(test, n = nrow(dge), sort.by = "PValue") |>
    as.data.frame()
}

get_hallmark_set <- function() {
  library(msigdb)
  library(ExperimentHub)
  library(GSEABase)
  file <- as.character(ut$get_data("msigdb_hallmark-7.4.rds"))

  eh <- ExperimentHub()
  msigdb_hs <- getMsigdb(org = "hs", id = "SYM", version = "7.4")
  hallmark_symbol <- subsetCollection(msigdb_hs, "h")
  mapping <- read_tsv(as.character(ut$get_data("ensembl_113_id_mapping.tsv"))) |>
    filter(!is.na(symbol))

  symbol2ensembl <- setNames(mapping$ensembl, mapping$symbol)

  hallmark <- sapply(names(hallmark_symbol), \(n) {
    gene_set <- hallmark_symbol[[n]]
    map_chr(gene_set, \(symbol) {
      symbol2ensembl[symbol]
    }) |> discard(is.na)
  }, USE.NAMES = TRUE, simplify = FALSE)
  saveRDS(hallmark, file)
  hallmark
}


#' Helper function to run DE with edgeR
#'
#' @description
#' The form of the analysis is to compare a given level
#' of factor `group` against the mean expression of all other levels in `group`
edgeR_mean_all <- function(dge, group, id_col = "GENEID") {
  normLibSizes(dge)
  var_ids <- dge$genes[[id_col]]
  var_cols <- colnames(dge$genes)
  mm <- model.matrix(as.formula(paste("~0+", group)), data = dge$samples)
  dge <- estimateDisp(dge, design = mm, robust = TRUE)
  fit <- glmQLFit(dge, mm, robust = TRUE)
  group_vec <- colnames(mm) |> keep(\(x) str_detect(x, group))
  group_levels <- unique(dge$samples[[group]])

  mean_val <- 1 / (length(group_vec) - 1)
  contrast_str <- map_chr(group_vec, \(x) {
    mean_others <- paste(mean_val, "*", group_vec[group_vec != x], collapse = "+")
    paste0(x, "-", "(", mean_others, ")")
  })
  ccs <- makeContrasts(contrasts = contrast_str, levels = mm)

  lapply(seq_along(contrast_str), \(i) {
    test <- glmQLFTest(fit, contrast = ccs[i, ])
    g <- group_vec[i]
    top <- topTags(test, n = nrow(dge), sort.by = "PValue") |>
      as.data.frame() |>
      as_tibble() |>
      select(-all_of(var_cols)) |>
      mutate(id = var_ids) |>
      relocate(id, .before = everything())
    top
    ## rename(!!as.symbol(glue("logFC_{g}")) := contrast_str[i])
  }) |>
    `names<-`(group_levels)
}

#' Return a named list mapping items in `name_col` to list of items in `elem_col`
#'
#' @description
#' @param name_col column to group on that will become the keys
group_by_into_list <- function(tb, name_col, elem_col, min_size = NULL,
                               with_counts = FALSE, count_col = "size",
                               keep_first = NULL,
                               unique = TRUE) {
  grouped <- group_by(tb, !!as.symbol(name_col)) |>
    summarize(
      items = list(!!as.symbol(elem_col)), !!as.symbol(count_col) := n(),
      across(any_of(keep_first), dplyr::first)
    )
  if (unique) {
    grouped$items <- lapply(grouped$items, unique)
    grouped$size <- map_dbl(grouped$items, length)
  }
  if (!is.null(min_size)) {
    grouped <- grouped[grouped[[count_col]] >= min_size, ]
  }
  lst <- grouped$items |> `names<-`(grouped[[name_col]])
  if (with_counts) {
    list(lst = lst, counts = select(grouped, -items))
  } else {
    lst
  }
}


show_reticulate_error <- function(expr) {
  captured <- substitute(expr)
  result <- tryCatch(
    expr = eval(captured, envir = parent.frame()),
    error = \(cnd) {
      last_error <- reticulate::py_last_error()
      message("Python error: ", last_error$type, "\n", last_error$value, "\n", last_error$traceback)
    }
  )
  result
}

lget <- function(lst, key, default = NULL) {
  val <- lst[[key]]
  if (is.null(val)) {
    default
  } else {
    val
  }
}

plage_wrapper <- function(counts, gene_sets, contrasts = NULL,
                          fc_cutoff = 1.5,
                          fit_method = "treat",
                          max_genes = NULL,
                          model_matrix = NULL) {
  library(GSVA)
  library(limma)
  dupes <- duplicated(rownames(counts))
  if (any(dupes)) {
    ndupes <- sum(dupes)
    warning(glue("{ndupes} duplicated features in counts detected! Keeping only first...\n
nrow before: {nrow(counts)}\n
nrow after: {nrow(counts) - ndupes}
"))
    counts <- counts[!duplicated(rownames(counts)), ]
  }
  params <- plageParam(exprData = counts, geneSets = gene_sets)
  result <- list()
  plage <- gsva(params) |> as.data.frame()
  result$values <- plage
  if (!is.null(contrasts) && !is.null(model_matrix)) {
    max_genes <- if (is.null(max_genes)) nrow(plage) else max_genes
    fit_gs <- lmFit(plage, design = model_matrix)
    tables <- list()
    de_gs <- lapply(names(contrasts), \(name) {
      # WARNING: there are better ways to handle testing on multiple contrasts
      # keeping it separate here because you test only few contrasts [2025-04-24 Thu]
      contrast_fit <- contrasts.fit(fit_gs, contrasts[[name]])
      if (fit_method == "eBayes") {
        contrast_fit <- eBayes(contrast_fit, robust = TRUE)
        tables[[name]] <<- topTable(contrast_fit, number = max_genes)
      } else if (fit_method == "treat") {
        contrast_fit <- treat(fit_gs, fc = fc_cutoff, robust = TRUE)
        tables[[name]] <<- topTreat(contrast_fit)
      }
      result <- decideTests(contrast_fit,
        lfc = log2(fc_cutoff)
        # Judge significant when abs(log2-fc) is at least this large
      )
      df <- data.frame(row.names = rownames(result))
      df[[name]] <- as.double(result[, 1])
      df
    }) |>
      bind_cols()
    result$fit <- fit_gs
    result$de <- de_gs
    tname <- if (fit_method == "treat") "topTreats" else "topTables"
    result[[tname]] <- tables
  }
  result
}


markers_internal <- function() yaml::read_yaml(as.character(ut$get_data("reference/cell_markers_custom.yaml")))
markers_meta_internal <- function() read_tsv(as.character(ut$get_data("reference/cell_markers_custom_meta.tsv")))

gs_meta_internal <- function() as.character(ut$get_data("reference/gene_sets_custom_meta.tsv", FALSE))

gs_internal <- function(from_file = FALSE, sets = c("go", "reactome", "hallmark"),
                        min_size = NULL,
                        max_size = 500) {
  ut <<- import("too_predict.utils")
  outfile <- as.character(ut$get_data("reference/gene_sets_custom.yaml", FALSE))
  meta_out <- as.character(ut$get_data("reference/gene_sets_custom_meta.tsv", FALSE))
  if (from_file) {
    result <- yaml::read_yaml(outfile)
    meta <- read_tsv(meta_out)
  } else {
    sets <- str_to_lower(sets)
    result <- c()
    meta <- empty_tibble(headers = c("id", "name", "source", "category", "size"))
    mode(meta$size) <- "double"
    if ("go" %in% sets) {
      not_allowed_evidence <- c( # Exclude evidence that isn't manually reviewed
        "IEA", "ND", "NAS", "ISS", "ISO", "ISA", "ISM", "IGC"
      )
      go_data <- as.character(ut$get_data("mappings/ensembl_go_map_2025-3-20.tsv")) |> read_tsv()
      go_top_level <- as.character(ut$get_data("mappings/go_term2to_level_3.csv")) |>
        read_csv() |>
        filter(!is.na(accession))
      go_tb <- go_data |> filter(`GO domain` == "biological_process" &
        !is.na(`GO term name`) &
        (!`GO term evidence code` %in% not_allowed_evidence) &
        (`GO term accession` %in% go_top_level$accession))

      bp_grouped <- go_tb |>
        mutate(set_name = paste0("GO_BP:", `GO term name`)) |>
        group_by_into_list("set_name", "Gene stable ID",
          with_counts = TRUE,
          keep_first = c("GO term accession", "GO term name")
        )

      go_meta <- bp_grouped$counts |>
        dplyr::rename(id = `GO term accession`, name = `GO term name`) |>
        select(id, name, size) |>
        mutate(source = "GO_BP") |>
        left_join(select(go_top_level, accession, top_level_name),
          by = join_by(x$id == y$accession)
        ) |>
        dplyr::rename(category = top_level_name)

      meta <- bind_rows(meta, go_meta)
      result <- c(result, bp_grouped$lst)
    }
    if ("reactome" %in% sets) {
      ensembl2reactome <- as.character(ut$get_data("mappings/ensembl2reactome_2025-4-11.tsv")) |> read_tsv()

      reactome <- as.character(ut$get_data("ReactomePathways_2025-4-11.txt")) |>
        read_tsv(col_names = c("Reactome ID", "name", "species")) |>
        filter(species == "Homo sapiens") |>
        inner_join(ensembl2reactome, by = join_by(`Reactome ID`))

      hierarchy <- as.character(ut$get_data("Reactome_Complex_2_Pathway_human-2025-4-23.txt")) |>
        read_tsv() |>
        inner_join(distinct(reactome, `Reactome ID`, .keep_all = TRUE),
          by = join_by(x$top_level_pathway == y$`Reactome ID`)
        ) |>
        dplyr::rename(category = name) |>
        distinct(pathway, .keep_all = TRUE)

      reactome_grouped <- reactome |>
        mutate(set_name = paste0("Reactome:", name)) |>
        group_by_into_list("set_name", "Gene stable ID",
          with_counts = TRUE,
          keep_first = c("name", "Reactome ID")
        )

      reactome_meta <- reactome_grouped$counts |>
        dplyr::rename(id = `Reactome ID`) |>
        select(size, name, id) |>
        left_join(select(hierarchy, pathway, category), by = join_by(x$id == y$pathway)) |>
        mutate(source = "Reactome")

      meta <- bind_rows(meta, reactome_meta)
      result <- c(result, reactome_grouped$lst)
    }
    if ("hallmark" %in% sets) {
      hfile <- as.character(ut$get_data("msigdb_hallmark-7.4.rds"))
      if (!file.exists(hfile)) {
        hallmark <- get_hallmark_set()
      } else {
        hallmark <- readRDS(hfile)
      }
      names(hallmark) <- str_replace(names(hallmark), "^HALLMARK_", "Hallmark:")

      h_meta <- tibble(
        id = NA, name = str_remove(names(hallmark), "Hallmark:"),
        size = map_dbl(hallmark, length),
        category = NA,
        source = "Hallmark"
      )

      meta <- bind_rows(meta, h_meta)
      result <- c(result, hallmark)
    }
    if (nrow(meta > 0)) {
      meta$set_name <- paste0(meta$source, ":", meta$name)
      write_tsv(meta, meta_out)
    }
    yaml::write_yaml(result, outfile)
  }
  lengths <- map_dbl(result, length)
  mask <- TRUE
  if (!is.null(min_size)) {
    mask <- lengths >= min_size & mask
  }
  if (!is.null(max_size)) {
    mask <- lengths <= max_size & mask
  }
  result[mask]
}


#' Keep only gene sets that have >= `min_nonzero_percent` of nonzero genes
#' in >= `min_sample_percent` of samples
#' if stats_only, returns a sample x gene_set dataframe containing the fraction
#'    of nonzero genes in that gene set for that sample
filter_gene_sets <- function(gene_sets, counts, min_nonzero_percent = 50,
                             min_sample_percent = 70, stats_only = FALSE) {
  set_tb <- tibble(name = names(gene_sets), id = gene_sets) |>
    unnest(cols = c(id)) |>
    mutate(val = 1) |>
    pivot_wider(id_cols = name, names_from = id, values_from = val) |>
    column_to_rownames(var = "name") |>
    t() %>%
    replace(is.na(.), 0)
  nonzero <- counts > 0
  mode(nonzero) <- "integer"
  set_mask <- rownames(set_tb) %in% rownames(counts)
  n_missing <- colSums(set_tb[!set_mask, ])
  set_tb <- set_tb[set_mask, ]
  nonzero <- t(nonzero[rownames(nonzero) %in% rownames(set_tb), ])
  set_sums <- nonzero %*% set_tb
  # sample x pathway matrix where values are the count of nonzero genes for that pathway
  # in that sample
  totals <- colSums(set_tb) + n_missing
  set_percent <- apply(set_sums, 1, \(r) r / totals) |>
    t() %>%
    replace(is.na(.), 0)
  pass_nonzero_thresh <- set_percent >= min_nonzero_percent
  pass_sample_thresh <- (colSums(pass_nonzero_thresh) / nrow(set_percent)) >= min_sample_percent
  stopifnot("This shouldn't exceed 1!" = max(set_percent) <= 1)
  print(glue("N sets before: {length(gene_sets)}"))
  print(glue("N sets passed: {sum(pass_sample_thresh)}"))
  if (!stats_only) {
    gene_sets[pass_sample_thresh]
  } else {
    set_percent |>
      as.data.frame() |>
      rownames_to_column(var = "sample")
  }
}


#' Helper function for aggregating gene set fractions from the
#'  `filter_gene_sets` stats mode
#'
#' @description
#' returns a gene_set x agg_cols dataframe. The columns are all the unique elements
#'    in each of agg_cols
#' @param agg_cols Columns of metadata with which to aggregate the samples by
agg_gene_set_fractions <- function(set_stats, metadata, agg_cols, join_on = "sample",
                                   agg_fn = mean) {
  w_meta <- inner_join(set_stats, metadata, by = join_by(!!as.symbol(join_on)))
  lapply(agg_cols, \(col) {
    other_cols <- colnames(metadata)
    other_cols <- other_cols[other_cols != col]
    w_meta |>
      select(-all_of(other_cols)) |>
      group_by(!!as.symbol(col)) |>
      summarise(across(where(is.numeric), agg_fn)) |>
      pivot_longer(cols = -!!as.symbol(col)) |>
      pivot_wider(names_from = !!as.symbol(col), values_from = value) |>
      dplyr::rename_with(\(c) paste0(col, "-", c), .cols = -name)
  }) |>
    purrr::reduce(\(x, y) inner_join(x, y, by = join_by(name)))
}


bisque_marker_wrapper <- function(counts, markers) {
  library(tidyverse)
  if (is.character(markers)) {
    markers <- yaml::read_yaml(markers)
  }
  marker_spec <- tibble(gene = markers, cluster = names(markers)) |>
    tidyr::unnest(cols = c(gene)) |>
    as.data.frame()
  mode(counts) <- "integer"
  rownames(counts) <- var_names
  colnames(counts) <- sample_names
  eset <- Biobase::ExpressionSet(assayData = counts)
  res <- BisqueRNA::MarkerBasedDecomposition(eset, marker_spec)
}
