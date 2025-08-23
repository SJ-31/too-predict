library(tidyverse)
library(glue)

if (sys.nframe() == 0) {
  library("optparse")
  parser <- OptionParser()
  parser <- add_option(
    parser,
    c("-v", "--var"),
    help = "Name of variable denoting different runs of the same model"
  )
  parser <- add_option(
    parser,
    c("-i", "--input"),
    type = "character",
    help = "Input aggregated results file",
    default = NULL
  )
  parser <- add_option(
    parser,
    c("-s", "--src"),
    type = "character",
    help = "Path to R src"
  )
  parser <- add_option(
    parser,
    c("-o", "--omnibus"),
    type = "character",
  )
  parser <- add_option(
    parser,
    c("-p", "--post_hoc"),
    type = "character",
    help = "Output file for post_hoc results",
    default = NULL
  )
  parser <- add_option(
    parser,
    c("-l", "--plot"),
    type = "character",
    help = "Output file for variation plot",
    default = NULL
  )
  args <- parse_args(parser)
}

source(glue("{args$src}/utils.R"))
source(glue("{args$src}/plotting.R"))

result <- read_csv(args$input)
var <- args$var

if ("repeat" %in% colnames(result)) {
  result[[var]] <- paste0(result[[var]], "_", result[["repeat"]])
  result <- select(result, -`repeat`)
}


unique_metrics <- unique(result$metric)
unique_tasks <- unique(result$task)


# Omnibus tests with Friedman
omnibus_result <- lapply(unique_metrics, \(m) {
  lapply(unique_tasks, \(t) {
    current <- filter(result, metric == m, task == t)
    if ("context" %in% colnames(current)) {
      current <- filter(current, context == "test")
    }
    test <- friedman_test_wrapper(current, "metric", var)
    tibble(metric = m, task = t, p_value = test$p_value_alt)
  }) |>
    bind_rows()
}) |>
  bind_rows() |>
  mutate(p_adjust = p.adjust(p_value))

significant <- omnibus_result |> filter(p_adjust <= 0.01)

# Follow-up post-hoc tests with Wilcoxon
if (nrow(significant) > 1) {
  post_hoc_results <- apply(significant, 1, \(row) {
    print(row)
    cur_metric <- row[["metric"]]
    cur_task <- row[["task"]]
    filtered <- result |> filter(metric == cur_metric & task == cur_task)
    tidy_pairwise(
      filtered$model,
      filtered$value,
      \(x, y) {
        wilcox.test(x, y)
      },
      \(x) p.adjust(x, method = "bonferroni")
    ) |>
      mutate(metric = cur_metric)
  }) |>
    bind_rows()
} else {
  post_hoc_results <- tibble()
}


## * Variation plot

plot_metric_variation <- function(tb) {
  tb |>
    ggplot(aes(x = model, y = value, color = metric)) +
    geom_boxplot() +
    facet_wrap(~task)
}

plot <- plot_metric_variation(result) +
  labs(title = glue("Metric variation across `{var}`"))


write_csv(omnibus_result, args$omnibus)
write_csv(post_hoc_results, args$post_hoc)
ggsave(args$plot, plot)
