read_existing <- function(filename, expr, read_fn = identity()) {
  if (file.exists(filename)) {
    read_fn(filename)
  } else {
    expr(filename)
  }
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
friedman_test_wrapper <- function(tb, metric_col, with_class = TRUE) {
  j <- length(unique(tb$fold))
  k <- length(unique(tb$model))
  test <- friedman.test(tb[[metric_col]], groups = tb$model, blocks = tb$fold)
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
    rename(!!as.symbol(row_header) := Var1) |>
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
