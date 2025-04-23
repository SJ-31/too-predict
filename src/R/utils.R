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

adata2dge <- function(adata, layer = NULL) {
  if (!is.null(layer)) {
    counts <- t(as.matrix(adata$layers[[layer]]))
  } else {
    counts <- t(as.matrix(adata$X))
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
group_by_into_list <- function(tb, name_col, elem_col) {
  tb <- group_by(tb, !!as.symbol(name_col)) |> summarize(items = list(!!as.symbol(elem_col)))
  tb$items |> `names<-`(tb[[name_col]])
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

markers_internal <- function() yaml::read_yaml(as.character(ut$get_data("reference/cell_markers_custom.yaml")))

gs_internal <- function(sets = c("go", "reactome", "hallmark")) {
  ut <- import("too_predict.utils")

  sets <- str_to_lower(sets)
  result <- c()
  meta <- empty_tibble(headers = c("id", "name", "source", "category", "size"))
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

    bp_mapping <- go_tb |>
      mutate(`GO term name` = str_replace_all(paste0("GO_BP:", `GO term name`), " ", "_")) |>
      group_by_into_list("GO term name", "Gene stable ID")

    go_meta <- tibble(
      name = str_replace_all(str_remove(names(bp_mapping), "GO_BP:"), "_", " "),
      size = map_dbl(bp_mapping, length),
      source = "GO_BP"
    ) |>
      inner_join(select(go_tb, `GO term accession`, `GO term name`),
        by = join_by(x$name == y$`GO term name`)
      ) |>
      rename(id = `GO term accession`) |>
      inner_join(select(go_top_level, accession, top_level_name),
        by = join_by(x$id == y$accession)
      ) |>
      rename(category = top_level_name)

    meta <- bind_rows(meta, go_meta)
    result <- c(result, bp_mapping)
  }
  if ("reactome" %in% sets) {
    ensembl2reactome <- as.character(ut$get_data("mappings/ensembl2reactome_2025-4-11.tsv")) |> read_tsv()
    reactome <- as.character(ut$get_data("ReactomePathways_2025-4-11.txt")) |>
      read_tsv(col_names = c("Reactome ID", "name", "species")) |>
      filter(species == "Homo sapiens")
    reactome_mapping <- ensembl2reactome |>
      inner_join(reactome, by = join_by(`Reactome ID`)) |>
      mutate(name = str_replace_all(paste0("Reactome:", name), " ", "_")) |>
      group_by_into_list("name", "Gene stable ID")

    # TODO: include the top level reactome terms in the hierarchy
    result <- c(result, reactome_mapping)
  }
  if ("hallmark" %in% sets) {
    hfile <- as.character(ut$get_data("msigdb_hallmark-7.4.rds"))
    if (!file.exists(hfile)) {
      hallmark <- get_hallmark_set()
    } else {
      hallmark <- readRDS(hfile)
    }
    names(hallmark) <- str_replace(names(hallmark), "^HALLMARK_", "Hallmark:")
    result <- c(result, hallmark)
  }
  result
}
