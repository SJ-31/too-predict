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
  factor_str <- paste0(c(group, technical_factors), collapse = " + ")
  clr_args$useMC <- use_parallel
  counts <- assays(data)[[count_slot]]
  var_col <- ifelse(!is.null(gene_col), gene_col, "var")
  rownames(counts) <- vars
  mm <- model.matrix(as.formula(paste("~0+", factor_str)), data = colData(data))
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
