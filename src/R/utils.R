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

adata2seurat <- function(adata, layer = NULL, assay = "RNA") {
  if (is.character(adata)) {
    adata <- ad$read_h5ad(adata)
  }
  if (is.null(layer)) {
    obj <- SeuratObject::CreateSeuratObject(counts = t(adata$X), assay = assay, meta.data = adata$obs)
  } else {
    obj <- SeuratObject::CreateSeuratObject(counts = t(adata$layers[[layer]]), assay = assay, meta.data = adata$obs)
  }
  rownames(obj) <- rownames(adata$var)
  colnames(obj) <- rownames(adata$obs)
  obj[[assay]][[]] <- adata$var
  obj
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
  rownames(counts) <- rownames(adata$var)
  colnames(counts) <- rownames(adata$obs)
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
markers_meta_internal <- function(grouped = TRUE) {
  tb <- read_tsv(as.character(ut$get_data("reference/cell_markers_custom_meta.tsv")))
  if (grouped) {
    tb |>
      mutate(
        set_name = paste0(tissue, "-", cell_type),
        set_name = case_when(from_tme ~ paste0(set_name, "-tme"), .default = set_name)
      ) |>
      select(-all_of(c("tissue", "cell_type", "ensembl", "from_tme"))) |>
      group_by(set_name) |>
      summarise(size = n(), source = dplyr::first(source))
  } else {
    tb
  }
}

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
      h_meta$category <- case_match(
        h_meta$name,
        "TNFA_SIGNALING_VIA_NFKB" ~ "Immune System",
        "HYPOXIA" ~ "Cellular responses to stimuli",
        "CHOLESTEROL_HOMEOSTASIS" ~ "Metabolism",
        "MITOTIC_SPINDLE" ~ "Cell Cycle",
        "WNT_BETA_CATENIN_SIGNALING" ~ "Signal Transduction",
        "TGF_BETA_SIGNALING" ~ "Signal Transduction",
        "IL6_JAK_STAT3_SIGNALING" ~ "Signal Transduction",
        "DNA_REPAIR" ~ "DNA Repair",
        "G2M_CHECKPOINT" ~ "Cell Cycle",
        "APOPTOSIS" ~ "Programmed Cell Death",
        "NOTCH_SIGNALING" ~ "Signal Transduction",
        "ADIPOGENESIS" ~ "Metabolism",
        "ESTROGEN_RESPONSE_EARLY" ~ "response to hormone",
        "ESTROGEN_RESPONSE_LATE" ~ "response to hormone",
        "ANDROGEN_RESPONSE" ~ "response to hormone",
        "MYOGENESIS" ~ "Developmental Biology",
        "PROTEIN_SECRETION" ~ "Metabolism of proteins",
        "INTERFERON_ALPHA_RESPONSE" ~ "Immune System",
        "INTERFERON_GAMMA_RESPONSE" ~ "Immune System",
        "APICAL_JUNCTION" ~ "cellular component organization",
        "APICAL_SURFACE" ~ "cellular component organization",
        "HEDGEHOG_SIGNALING" ~ "Signal Transduction",
        "COMPLEMENT" ~ "Immune System",
        "UNFOLDED_PROTEIN_RESPONSE" ~ "regulation of protein stability",
        "PI3K_AKT_MTOR_SIGNALING" ~ "Signal Transduction",
        "MTORC1_SIGNALING" ~ "Signal Transduction",
        "E2F_TARGETS" ~ "Cell Cycle",
        "MYC_TARGETS_V1" ~ "Programmed Cell Death",
        "MYC_TARGETS_V2" ~ "Programmed Cell Death",
        "EPITHELIAL_MESENCHYMAL_TRANSITION" ~ "cell differentiation",
        "INFLAMMATORY_RESPONSE" ~ "Immune System",
        "XENOBIOTIC_METABOLISM" ~ "Metabolism",
        "FATTY_ACID_METABOLISM" ~ "Metabolism",
        "OXIDATIVE_PHOSPHORYLATION" ~ "Metabolism",
        "GLYCOLYSIS" ~ "Metabolism",
        "REACTIVE_OXYGEN_SPECIES_PATHWAY" ~ "response to oxidative stress",
        "P53_PATHWAY" ~ "Gene expression (Transcription)",
        "UV_RESPONSE_UP" ~ "response to radiation",
        "UV_RESPONSE_DN" ~ "response to radiation",
        "ANGIOGENESIS" ~ "angiogenesis",
        "HEME_METABOLISM" ~ "tetrapyrrole metabolic process",
        "COAGULATION" ~ "regulation of body fluid levels",
        "IL2_STAT5_SIGNALING" ~ "Signal Transduction",
        "BILE_ACID_METABOLISM" ~ "Metabolism",
        "PEROXISOME" ~ "cellular component organization",
        "ALLOGRAFT_REJECTION" ~ "Immune System",
        "SPERMATOGENESIS" ~ "regulation of reproductive process",
        "KRAS_SIGNALING_UP" ~ "Signal Transduction",
        "KRAS_SIGNALING_DN" ~ "Signal Transduction",
        "PANCREAS_BETA_CELLS" ~ "Metabolism"
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
    purrr::reduce(\(x, y) inner_join(x, y, by = join_by(name))) |>
    mutate(across(where(is.numeric), \(x) round(x, 3)))
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

bisque_reference_wrapper <- function(counts, reference, ref_obs, markers = NULL,
                                     cell_type_col = "cell_type",
                                     subject_col = "subject",
                                     use_overlap = FALSE) {
  mode(counts) <- "integer"
  mode(reference) <- "integer"
  shared_genes <- intersect(rownames(counts), rownames(ref))
  if (!is.null(markers)) shared_genes <- intersect(shared_genes, markers)
  counts <- counts[rownames(counts) %in% shared_genes, ]
  reference <- reference[rownames(reference) %in% shared_genes, ]
  mask <- colSums(reference) != 0
  reference <- reference[, mask]
  ref_obs <- ref_obs[mask, ]

  eset <- Biobase::ExpressionSet(assayData = counts)
  ref <- Biobase::ExpressionSet(
    assayData = reference,
    phenoData = Biobase::AnnotatedDataFrame(ref_obs)
  )
  BisqueRNA::ReferenceBasedDecomposition(
    bulk.eset = eset, sc.eset = ref, cell.types = cell_type_col,
    subject.names = subject_col, use.overlap = use_overlap
  )
}

#' Group pathways by `category_col` and count the number of statistically
#'  significant pathways for each
#'
#' @param enrichment_results a list mapping enrichment analyses to their results
#'    each value of this list must have `join_by` and `p_value_col`
enrichment_summary <- function(metadata, enrichment_results,
                               cutoff = 0.05,
                               p_value_col = "padj",
                               category_col = "category",
                               join_by = "set_name") {
  final <- lmap(enrichment_results, \(x) {
    tb <- left_join(x[[1]], metadata, by = join_by(!!as.symbol(join_by))) |>
      filter(!!as.symbol(p_value_col) <= cutoff) |>
      group_by(!!as.symbol(category_col)) |>
      summarise(!!as.symbol(names(x)) := n())
    tb
  }) |>
    purrr::reduce(\(x, y) full_join(x, y, by = join_by(!!as.symbol(category_col)))) |>
    mutate(across(where(is.numeric), \(x) replace(x, is.na(x), 0)))
  sorted_cols <- sort(colnames(final))
  final |>
    select(all_of(sorted_cols)) |>
    relocate(!!as.symbol(category_col), .before = dplyr::everything())
}

#' Convert fgsea result object into the format of gseGO as produced by clusterProfiler
#'
fgsea_result2gseGO <- function(fgsea, id_col = NULL, delim = "-") {
  if (!is.null(id_col)) {
    expanded <- fgsea |> rename(ID := !!as.symbol(id_col), Description = pathway)
  } else {
    expanded <- fgsea |>
      tidyr::separate_wider_delim(pathway, delim = delim, names = c("ID", "Description"), too_many = "merge")
  }
  expanded <- rowwise(expanded) |> mutate(leadingEdge = paste0(unlist(leadingEdge), collapse = "/"))
  enrich_df <- data.frame(
    ID = expanded$ID,
    Description = expanded$Description,
    setSize = expanded$size,
    enrichmentScore = expanded$ES,
    NES = expanded$NES,
    pvalue = expanded$pval,
    p.adjust = expanded$padj,
    core_enrichment = expanded$leadingEdge
  )
  rownames(enrich_df) <- enrich_df$ID
  enrich_df
}

#' Compute the average value e.g. for a given gene set
#'
#' The average is taken with respect to the genes present in `reference`
#' @param reference a tibble/dataframe with a column containing the gene information
gene_set_average <- function(gene_sets, reference, ref_gene_col = "GENEID", ref_val_col = "logFC") {
  eff_sizes <- map_dbl(gene_sets, \(x) length(intersect(x, reference[[ref_gene_col]])))
  df <- tibble(set = names(gene_sets), gene = gene_sets, value = 1) |>
    unnest(cols = c(gene)) |>
    pivot_wider(id_cols = set, names_from = gene, values_from = value) |>
    column_to_rownames(var = "set")
  df[is.na(df)] <- 0
  reference <- left_join(tibble(!!ref_gene_col := colnames(df)), reference, by = join_by(!!ref_gene_col)) |>
    mutate(across(where(is.numeric), \(x) replace(x, is.na(x), 0)))
  as.matrix(df) %*% reference[[ref_val_col]] |>
    as.data.frame() |>
    rownames_to_column(var = "set_name") |>
    rename(average = V1) |>
    as_tibble() |>
    mutate(average = average / eff_sizes)
}

#' Replace a column, rowname or colnames in obj with a mapping
#'
#' @description
#' @param `drop_missing` whether or not to drop elements in `obj` that aren't found
#'    in the mapping
#' @param mapping a list of old->new
mapping_replace <- function(obj, what, mapping, drop_missing = TRUE) {
  mask_on <- "rows"
  if (what == "rownames" && !is.null(rownames(obj))) {
    query <- mapping[rownames(obj)]
  } else if (what == "colnames" && !is.null(colnames(obj)) && drop_missing) {
    query <- mapping[colnames(obj)]
    mask_on <- "columns"
  } else { # `what` is a column name
    query <- mapping[obj[[what]]]
  }
  nas <- is.na(query)
  if (drop_missing && mask_on == "rows") {
    obj <- obj[!nas, ]
  } else if (drop_missing) {
    obj <- obj[, !nas]
  } else if (!drop_missing && any(nas)) {
    stop("Not all old names could be mapped!")
  }
  if (drop_missing) {
    print("Dropping {glue(sum(nas))} entries with non-mapping names...")
  }
  query <- query[!nas]
  if (what == "rownames") {
    rownames(obj) <- query
  } else if (what == "colnames") {
    colnames(obj) <- query
  } else {
    obj[[what]] <- query
  }
  obj
}


counts2tpm <- function(data, lengths = NULL) {
  if (is.character(lengths) && "DGEList" %in% class(data)) {
    lengths <- data$genes[[lengths]]
    counts <- data$counts
  } else if (is.character(lengths) && "anndata._core.anndata.AnnData" %in% class(data)) {
    lengths <- data$var[[lengths]]
    counts <- t(data$X)
  }
  stopifnot(length(lengths) == nrow(counts) && is.numeric(lengths))
  numer <- log(counts) - log(lengths)
  denom <- log(exp(colSums(numer)))
  tpm <- exp(numer - denom + log(1e6))
  tpm
}

# Must create a new object because this is not supported...
rename_seurat_features <- function(obj, new_names, mapping = FALSE) {
  assays <- Seurat::Assays(obj)
  layers <- SeuratObject::Layers(obj[[assays[1]]])
  if (mapping) {
    new_names <- new_names[rownames(obj)]
    obj <- obj[!is.na(new_names), ]
    new_names <- new_names[!is.na(new_names)]
  }
  new <- SeuratObject::CreateSeuratObject(
    counts = SeuratObject::LayerData(obj[[assays[1]]], layer = layers[1]) |>
      `rownames<-`(NULL),
    assay = assays[1],
    meta.data = obj[[]]
  )
  rownames(new) <- new_names
  colnames(new) <- colnames(obj)
  if (length(layers) > 1) {
    for (l in layers[2:length(layers)]) {
      SeuratObject::LayerData(new, assay = assays[1], layer = l) <- SeuratObject::LayerData(obj, assay = assays[1], layer = l) |> `rownames<-`(NULL)
    }
  }
  if (length(assays) > 1) {
    for (a in assays[2:length(assays)]) {
      SeuratObject::LayerData(new, assay = a)
      new[[a]][[]] <- obj[[a]][[]]
    }
  }
  new
}



#' Subcluster cells
#'
#' @description
#' Partition the seurat object by cell type, then cluster independently for each
#' obj is assumed to have undergone normalization and is ready for `FindClusters`
seurat_subcluster_cells <- function(obj, cell_col, subcluster_col = "cell_subclusters") {
  sub_clustered <- lapply(unique(obj[[]][[cell_col]]), \(type) {
    mask <- obj[[]][[cell_col]] == type
    cur <- obj[, mask]
    cur <- FindClusters(cur)
    cur[[]][[subcluster_col]] <- paste0(cur[[]][[cell_col]], ".", cur[[]]$seurat_clusters)
    cur
  })
  final <- merge(
    x = sub_clustered[[1]], y = sub_clustered[2:length(sub_clustered)],
    merge.data = TRUE, merge.dr = TRUE
  )
  final[[]]$seurat_clusters <- NULL
  final
}


gene_set_analysis <- function(method = c("fgsea"), data,
                              gene_sets,
                              partition_col = "direction", p_threshold = 0.05, ...) {
  # TODO: write more methods for this
  if (class(data) == "list" || class(data) == "numeric") preranked <- TRUE
  dnames <- c("pos", "neg")
  if (class(data) == "list") {
    if (length(intersect(dnames, names(data))) != 2) {
      stop("Names for one-tailed test list should be c('pos', 'neg')")
    }
    if (length(intersect(data$pos, data$neg)) > 0) {
      stop("One-tailed test lists should be disjoint!")
    }
  }

  gsa_internal <- function() {
    ## TODO: not implemented
  }
  result <- list()

  if (method == "fgsea") {
    if (class(data) == "list") {
      all_results <- lapply(dnames, \(n) {
        cur <- fgsea::fgsea(pathways = gene_sets, stats = data[[n]], scoreType = n, ...)
        cur[cur$padj <= p_threshold, ]
      }) |> `names<-`(dnames)
      together <- lapply(dnames, \(n) {
        tib <- as_tibble(all_results[[n]])
        tib[[partition_col]] <- n
        tib
      }) |> bind_rows()
      result$tb <- together
      result$raw <- all_results
    } else {
      raw <- fgsea::fgsea(pathways = gene_sets, stats = data, ...)
      raw <- raw[raw$padj <= p_threshold, ]
      result$raw <- raw
      result$tb <- as_tibble(result$raw) |> mutate(padj <= p_threshold)
    }
  }
  result
}
