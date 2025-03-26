suppressMessages({
  library(here)
  library(tidyverse)
  library(glue)
  library(edgeR)
  library(here)
  library(ggridges)
  Sys.setenv("RETICULATE_PYTHON" = here(".venv", "bin", "python"))
  library(reticulate)
  source(here("src", "R", "utils.R"))
})

## * Setup

OUTDIR <- here("data", "output", "imbalance")
CV <- here("data", "output", "cross_validation")
CV_PREFIX <- "additional_splits"
OBJECTIVE_NAME <- "Kappa"
## BASELINE <- # The objective value to beat
# [2025-03-26 Wed]
bfile <- here("data", "output", "balancing_results_2025-3-24.csv")

bresults <- read_csv(bfile) |>
  mutate(
    sampling_name = coalesce(undersample_name, oversample_name),
    strategy = coalesce(undersampling_strategy, oversampling_strategy)
  ) |>
  select(-undersample_name, -oversample_name, -undersampling_strategy, -oversampling_strategy) |>
  filter(!is.na(classifier))


## * Helper functions
compare_misses <- function(dirs, label_col = "tumor_type",
                           wanted_meta = c("Project_ID", "Case_ID", "test_set")) {
  lapply(dirs, \(x) {
    miss_file <- here(CV, x, CV_PREFIX, glue("{label_col}-misses.csv"))
    if (file.exists(miss_file)) {
      read_csv(miss_file) |>
        select(all_of(c(wanted_meta, label_col))) |>
        mutate(model = x)
    }
  }) |> bind_rows()
}

test <- read_csv("/home/shannc/Bio_SDD/too-predict/data/output/cross_validation/clr_xgboost_variance_GO/additional_splits/tumor_type-cm_cm-CHULA.csv")

get_totals <- function(tb, model, test_set, label_col = "tumor_type") {
  cm_file <- here(CV, model, CV_PREFIX, glue("{label_col}-cm_cm-{test_set}.csv"))
  if (file.exists(cm_file)) {
    total_counts <- read_csv(cm_file) |>
      column_to_rownames(var = "...1") |>
      rowSums() |>
      discard(\(x) x == 0)
    tb |> mutate(total = map_dbl(!!as.symbol(label_col), \(x) total_counts[x]))
  } else {
    tb |> mutate(total = NA)
  }
}

compare_strategies <- function(name, objective_col, tb = bresults) {
  summarized <- tb |>
    filter(sampling_name == name) |>
    group_by(strategy) |>
    summarise(objective = mean(!!as.symbol(objective_col)))
  baseline <- summarized |> filter(min_rank(objective) == 1)
  message(glue("Using `{baseline$strategy}` value ({round(baseline$objective, 2)}) as baseline"))
  summarized$objective <- summarized$objective - baseline$objective
  # Use the smallest objective as a base reference
  summarized
}

## * Main

comparison_plot <- bresults %>%
  group_by(sampling_type) %>%
  ggplot(aes(
    y = `objective_value-kappa`, x = sampling_name, fill = strategy
  )) +
  geom_bar(position = "dodge", stat = "identity") +
  facet_wrap(~classifier) +
  ylab(OBJECTIVE_NAME)
comparison_plot
ggsave(here(OUTDIR, glue("{OBJECTIVE}_comparison.png")), comparison_plot)

compare_strategies(
  "RandomUnderSampler", "objective_value-kappa",
  filter(bresults, classifier == "XGB")
)

tt_misses <- compare_misses(
  dirs = c(
    "clr_xgboost_edger_1000_undersample", "clr_xgboost_edger_smote",
    "alr_xgboost_low_variance_1000"
  ),
  label_col = "tumor_type"
)


tt_miss_counts <- tt_misses |>
  group_by(model, test_set, tumor_type) |>
  summarise(count = n()) |>
  nest() |>
  apply(1, \(x) {
    get_totals(
      model = x$model, test_set = x$test_set, tb = x$data,
      label_col = "tumor_type"
    ) |> mutate(model = x$model, test_set = x$test_set, freq = round(count / total, 2))
  }) |>
  bind_rows()


tt_miss_plot <- tt_miss_counts |>
  filter(test_set == "CHULA") |>
  ggplot(aes(x = model, y = freq, fill = tumor_type)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~test_set) +
  ylab("Miss frequency")
tt_miss_plot

ggsave(here(OUTDIR, "tumor_type_misses.png"), tt_miss_plot)
