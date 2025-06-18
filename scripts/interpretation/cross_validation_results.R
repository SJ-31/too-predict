library(here)
library(tidyverse)
library(paletteer)
library(broom)
library(glue)
source(here("src", "R", "utils.R"))
source(here("src", "R", "plotting.R"))

OUTDIR <- here("data", "output", "cross_validation")
SUBDIRECTORY <- ""
VAR <- "fold" # Name of the column denoting different evaluation sets
LABEL <- "tumor_type"

if (sys.nframe() == 0) {
  library("optparse")
  parser <- OptionParser()
  parser <- add_option(
    parser,
    c("-s", "--subdirectory"),
    type = "character",
    help = "subdirectory within the model result directory to pull data from",
    default = "" # e.g. [2025-03-11 Tue] use this to get stuff for "organoid_test_split"
  )
  parser <- add_option(
    parser,
    c("-v", "--var"),
    type = "character",
    help = "Name of column denoting different evaluation sets",
    default = "fold"
  )
  parser <- add_option(
    parser,
    c("-l", "--label"),
    type = "character",
    help = "Label that was predicted in cross validation",
    default = "tumor_type"
  )
  args <- parse_args(parser)
  OUTDIR <- here(OUTDIR, args$subdirectory)
  dir.create(OUTDIR)
  VAR <- as.character(args$var)
  SUBDIRECTORY <- args$subdirectory
}


DIRS <- list.files(
  here("data", "output", "cross_validation"),
  full.names = TRUE
) |>
  keep(\(x) dir.exists(x) & (length(list.files(x)) > 0)) |>
  discard(
    \(x)
      basename_no_ext(x) %in%
        c("test", "confusion_matrices", "additional_splits")
  )


## * Data retrieval

#' Aggregate the cross validation results across all folds
#'
summarize_sets <- function(tb, how = "mean") {
  tb |>
    select(-!!as.symbol(VAR)) |>
    group_by(model, class, step) |>
    summarize(
      var_auc = var(auc),
      min_auc = min(auc),
      max_auc = max(auc),
      across(where(is.numeric), mean),
      across(where(is.character), unique)
    ) |>
    ungroup()
}


getter_fn <- function(label, suffix, format_fn) {
  lapply(DIRS, \(x) {
    model_name <- basename_no_ext(x)
    file <- here(x, SUBDIRECTORY, glue("{label}{suffix}.csv"))
    if (file.exists(file)) {
      suppressMessages(read_csv(file)) |>
        format_fn() |>
        mutate(model = model_name)
    } else {
      NULL
    }
  }) |>
    bind_rows() |>
    mutate(!!as.symbol(VAR) := as.character(!!as.symbol(VAR)))
}

get_rocs <- function(label) {
  getter_fn(label, "-roc", \(x) {
    group_by(x, class, !!as.symbol(VAR)) |>
      mutate(step = seq_len(n())) |>
      ungroup()
  })
}

get_prec_recall <- function(label) {
  tryCatch(
    expr = {
      getter_fn(label, "-prec_recall", \(x) {
        group_by(x, class, !!as.symbol(VAR)) |>
          mutate(
            step = seq_len(n()),
            class_avg_precision = average_precision,
            average_precision = as.numeric(str_remove(
              average_precision,
              ".*: "
            ))
          ) |>
          ungroup()
      })
    },
    error = function(cnd) {
      warning("Precision recall not available")
      NULL
    }
  )
}

get_misc <- function(label) {
  getter_fn(label, "-misc", \(x) x)
}

get_report <- function(label) {
  getter_fn(label, "-report", \(x) x)
}

roc <- get_rocs(LABEL)
pr <- get_prec_recall(LABEL)
misc <- get_misc(LABEL)
report <- get_report(LABEL) |> mutate(class = str_replace_all(class, "-", "_"))
if (!is.null(pr)) {
  pr_auc <- pr |>
    mutate(class = str_replace_all(class, "-", "_")) |>
    group_by(class, model, !!as.symbol(VAR)) |>
    summarise(prc_auc = unique(auc)) |>
    ungroup()
} else {
  pr_auc <- NULL
}


## * Hypothesis testing

# %%

# Metrics to test on (suitable for imbalanced data)
# - Kappa
# - MCC
# - PRC AUC
# - F1 score
metrics <- list(
  kappa = misc,
  mcc = misc,
  `f1-score` = filter(report, !grepl("avg", class)) |>
    mutate(
      !!as.symbol(VAR) := paste0(class, !!as.symbol(VAR))
    )
)
if (!is.null(pr_auc)) {
  metrics[["prc_auc"]] <- mutate(
    pr_auc,
    !!as.symbol(VAR) := paste0(class, !!as.symbol(VAR))
  )
}


get_tests <- function() {
  friedman_tt <- lapply(names(metrics), \(x) {
    result <- friedman_test_wrapper(metrics[[x]], x, var = VAR) |>
      tidy() |>
      mutate(metric = x)
    print(glue("{x} test success"))
    result
  }) |>
    bind_rows()

  write_csv(friedman_tt, here(OUTDIR, glue("friedman_test_{LABEL}.csv")))

  significant_tt <- friedman_tt |>
    filter(p.value <= 0.01) |>
    pluck("metric")

  wilcox_tt <- lapply(significant_tt, \(m) {
    tb <- metrics[[m]]
    tidy_pairwise(
      tb$model,
      tb[[m]],
      \(x, y) {
        wilcox.test(x, y)
      },
      \(x) p.adjust(x, method = "bonferroni")
    ) |>
      mutate(metric = m)
  }) |>
    bind_rows()
  # TODO: is there a better post-hoc test to use?
  write_csv(wilcox_tt, here(OUTDIR, glue("wilcox_{LABEL}.csv")))
}

try(get_tests())

# %%

# [2025-03-10 Mon] We probably want to maximize TPR
# should do this with weights

# List mapping desired metrics to logicals which are TRUE if higher values are better for the
# given metric
# %%
metric_rankings <- list(
  kappa = TRUE,
  `f1-score` = TRUE,
  prc_auc = TRUE,
  mcc = TRUE
)
if (is.null(pr_auc)) {
  metric_rankings$prc_auc <- NULL
}

## ** Ranking measure
combined <- local({
  rep <- report |>
    group_by(model, !!as.symbol(VAR)) |>
    summarise(across(where(is.numeric), mean)) |>
    select(`f1-score`, !!as.symbol(VAR), model) |>
    ungroup()
  to_reduce <-
    list(rep, misc)
  if (!is.null(pr_auc)) {
    prc <- pr_auc |>
      group_by(model, !!as.symbol(VAR)) |>
      summarise(across(where(is.numeric), mean)) |>
      select(prc_auc, !!as.symbol(VAR), model) |>
      ungroup()
    to_reduce <- list(rep, prc, misc)
  }
  reduce(to_reduce, \(x, y) {
    inner_join(x, y, by = c("model", VAR))
  })
})

write_csv(combined, here(OUTDIR, glue("combined_metrics_{LABEL}.csv")))
combined |>
  group_by(model) |>
  select(-fold) |>
  summarize(across(where(is.numeric), mean)) |>
  write_csv(here(OUTDIR, glue("combined_metrics_{LABEL}_folded.csv")))


max_score <- length(metric_rankings) * length(unique(combined[[VAR]]))
models <- unique(combined$model)


get_top <- rank_by_metrics("model", VAR, combined, metric_rankings)
write_csv(get_top$table, here(OUTDIR, glue("rank_score_tracker_{LABEL}.csv")))
write_csv(
  as_tibble(as.list(get_top$top)),
  here(OUTDIR, glue("model_ranks_{LABEL}.csv"))
)

## * Plots

if (!is.null(pr_auc)) {
  pr_auc_plot <- ggplot(pr_auc, aes(x = class, y = auc, fill = class)) +
    geom_boxplot() +
    facet_wrap(~model) +
    ylab("Area under the precision recall curve") +
    xlab("Tumor type")

  prc_plot <- local({
    s <- pr |> summarize_sets()
    ggplot(s, aes(x = recall, y = precision, color = class)) +
      facet_wrap(~model) +
      geom_step() +
      ylab("Precision") +
      xlab("Recall")
  })
}

roc_plot <- local({
  s <- roc |> summarize_sets()
  ggplot(s, aes(x = fpr, y = tpr, color = class)) +
    facet_wrap(~model) +
    geom_step() +
    ylab("True positive rate (TPR)") +
    xlab("False positive rate (FPR)")
})

dir <- "/home/shannc/Bio_SDD/too-predict/data/output/cross_validation/alr_random_forest_edger_lfc/additional_splits"
m_files <- list.files(
  dir,
  pattern = glue("{LABEL}.*cm.*csv"),
  full.names = TRUE
)

## * Confusion matrices

cm_outdir <- here(OUTDIR, "confusion_matrices")
dir.create(cm_outdir)
summarize_cm <- function(directory, label, pattern = NULL, outname = NULL) {
  if (is.null(pattern)) {
    m_files <- list.files(
      directory,
      pattern = glue("{label}.*cm.*csv"),
      full.names = TRUE
    )
  } else {
    m_files <- list.files(directory, pattern = pattern, full.names = TRUE)
  }
  if (length(m_files) > 0) {
    matrices <- lapply(m_files, \(m) {
      suppressMessages(read_csv(m)) |>
        rename(x = "...1") |>
        pivot_longer(-x, names_to = "y")
    })
    average_m <- bind_rows(matrices) |>
      group_by(x, y) |>
      summarise(value = mean(value)) |>
      ungroup()
    plot <- plot_confusion_matrix(
      average_m,
      x_label = "True",
      y_label = "Prediction",
      fill_label = "Average count"
    )
    if (nchar(SUBDIRECTORY) == 0) {
      model <- basename_no_ext(directory)
    } else {
      model <- split_path(directory)[2]
    }
    if (is.null(outname)) {
      outfile <- glue("{here(cm_outdir, model)}_{label}_avg_cm.png")
    } else {
      outfile <- glue("{here(cm_outdir, model)}_{outname}.png")
      print(outfile)
    }
    ggsave(outfile, plot, height = 12, width = 12)
    average_m |> mutate(model = model)
  }
}

all_cm <- lapply(DIRS, \(x) summarize_cm(here(x, SUBDIRECTORY), LABEL))

# Look at chula organoid CM
if (SUBDIRECTORY == "additional_splits") {
  lapply(DIRS, \(x) {
    summarize_cm(
      here(x, SUBDIRECTORY),
      pattern = glue("{LABEL}.*cm.*CHULA.csv"),
      outname = glue("{LABEL}-chula_avg_cm")
    )
  })
  lapply(DIRS, \(x) {
    summarize_cm(
      here(x, SUBDIRECTORY),
      pattern = glue("{LABEL}.*cm.*GEO.csv"),
      outname = glue("{LABEL}-geo_avg_cm")
    )
  })
}
