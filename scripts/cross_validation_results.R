library(here)
library(tidyverse)
library(broom)
library(glue)
source(here("src", "R", "utils.R"))

CV <- TRUE
outdir <- here("data", "output", "cross_validation")

if (sys.nframe() == 0) {
  library("optparse")
  parser <- OptionParser()
  parser <- add_option(parser, c("-v", "--verbose"),
    action = "store_true",
    default = TRUE, help = "Print extra output [default]"
  )
  parser <- add_option(parser, c("-s", "--subdirectory"),
    type = "character",
    help = "subdirectory within the model result directory to pull data from",
    default = NULL # e.g. [2025-03-11 Tue] use this to get stuff for "organoid_test_split"
  )
  parser <- add_option(parser, c("-n", "--no_cv"),
    type = "logical",
    help = "Whether or not the training to parse used cross validation", default = FALSE
  )
  args <- parse_args(parser)
  CV <- !args$no_cv
}

# TODO: [2025-03-11 Tue] gonna need a separate set of analysis fns for analyzing results
# without cross validation

## targets <- c("tumor_type", "primary_site")
targets <- "tumor_type"

DIRS <- list.files(outdir, full.names = TRUE) |>
  keep(\(x) dir.exists(x) & (length(list.files(x)) > 0)) |>
  discard(\(x) str_detect(x, "test"))

## * Data retrieval

#' Aggregate the cross validation results across all folds
#'
summarize_folds <- function(tb, how = "mean") {
  tb |>
    select(-fold) |>
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
    read_csv(here(x, glue("{label}{suffix}.csv"))) |>
      format_fn() |>
      mutate(model = model_name)
  }) |> bind_rows()
}

get_rocs <- function(label) {
  getter_fn(label, "-roc", \(x) {
    group_by(x, class, fold) |>
      mutate(step = seq_len(n())) |>
      ungroup()
  })
}

get_prec_recall <- function(label) {
  getter_fn(label, "-prec_recall", \(x) {
    group_by(x, class, fold) |>
      mutate(
        step = seq_len(n()),
        class_avg_precision = average_precision,
        average_precision = as.numeric(str_remove(average_precision, ".*: "))
      ) |>
      ungroup()
  })
}

get_misc <- function(label) {
  getter_fn(label, "-misc", \(x) x)
}

get_report <- function(label) {
  getter_fn(label, "-report", \(x) x)
}


roc_tumor_type <- get_rocs("tumor_type")
pr_tumor_type <- get_prec_recall("tumor_type")
misc_tumor_type <- get_misc("tumor_type")
report_tumor_type <- get_report("tumor_type")
pr_auc_tumor_type <- pr_tumor_type |>
  group_by(class, model, fold) |>
  summarise(prc_auc = unique(auc)) |>
  ungroup()

## * Hypothesis testing

## --- CODE BLOCK ---

# Metrics to test on (suitable for imbalanced data)
# - Kappa
# - MCC
# - PRC AUC
# - F1 score
metrics <- list(
  kappa = misc_tumor_type, mcc = misc_tumor_type,
  `f1-score` = mutate(report_tumor_type, fold = paste0(class, fold)),
  prc_auc = mutate(pr_auc_tumor_type, fold = paste0(class, fold))
)

friedman <- lapply(names(metrics), \(x) {
  friedman_test_wrapper(metrics[[x]], x) |>
    tidy() |>
    mutate(metric = x)
}) |>
  bind_rows()

significant <- friedman |>
  filter(p.value <= 0.01) |>
  pluck("metric")

lapply(significant, \(x) {

})

## --- CODE BLOCK ---

# [2025-03-10 Mon] We probably want to maximize TPR


# should do this with weights

# List mapping desired metrics to logicals which are TRUE if higher values are better for the
# given metric
## --- CODE BLOCK ---
metric_rankings <- list(kappa = TRUE, `f1-score` = TRUE, prc_auc = TRUE, mcc = TRUE)

## ** Ranking measure
combined <- local({
  rep <- report_tumor_type |>
    group_by(model, fold) |>
    summarise(across(where(is.numeric), mean)) |>
    select(`f1-score`, fold, model) |>
    ungroup()
  prc <- pr_auc_tumor_type |>
    rename(prc_auc = auc) |>
    group_by(model, fold) |>
    summarise(across(where(is.numeric), mean)) |>
    select(prc_auc, fold, model) |>
    ungroup()
  reduce(list(rep, prc, misc_tumor_type), \(x, y) {
    inner_join(x, y, by = c("model", "fold"))
  })
})

max_score <- length(metric_rankings) * length(unique(combined$fold))
models <- unique(combined$model)
make_score_tb <- function(winner) {
  tmp <- sapply(models, \(x) 0, simplify = FALSE)
  tmp[[winner]] <- 1
  as_tibble(tmp)
}

score_tracker <- empty_tibble(c("winner", "metric", "fold")) |> mutate(fold = as.double(fold))
rank_scores <- lapply(unique(combined$fold), \(f) {
  current <- filter(combined, fold == f)
  lapply(names(metric_rankings), \(m) {
    if (metric_rankings[[m]]) {
      sorted <- arrange(current, desc(!!as.symbol(m)))
    } else {
      sorted <- arrange(current)
    }
    winner <- head(sorted, n = 1) |> pluck("model")
    score_tracker <<- add_row(score_tracker, winner = winner, metric = m, fold = f)
    make_score_tb(winner)
  }) |>
    bind_rows() |>
    colSums()
}) |>
  bind_rows() |>
  colSums()

score_tracker_table <- table(score_tracker$winner, score_tracker$metric) |> table2tb("model")
write_csv(score_tracker_table, here(outdir, "rank_score_tracker.csv"))
write_csv(rank_scores, here(outdir, "model_ranks.csv"))

## * Plots

pr_auc_tumor_type_plot <- ggplot(pr_auc_tumor_type, aes(x = class, y = auc, fill = class)) +
  geom_boxplot() +
  facet_wrap(~model) +
  ylab("Area under the precision recall curve") +
  xlab("Tumor type")

pr_auc_plot <-
  roc_plot <- local({
    s <- roc_tumor_type |> summarize_folds()
    ggplot(s, aes(x = fpr, y = tpr, color = class)) +
      facet_wrap(~model) +
      geom_step() +
      ylab("True positive rate (TPR)") +
      xlab("False positive rate (FPR)")
  })

prc_plot <- local({
  s <- pr_tumor_type |> summarize_folds()
  ggplot(s, aes(x = recall, y = precision, color = class)) +
    facet_wrap(~model) +
    geom_step() +
    ylab("Precision") +
    xlab("Recall")
})
