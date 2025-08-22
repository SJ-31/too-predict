library(tidyverse)
library(glue)
if (exists("snakemake")) {
  source(paste0(snakemake@config$src$R, "/", "utils.R"))
  source(paste0(snakemake@config$src$R, "/", "plotting.R"))
  result <- read_csv(snakemake@input[[1]])
  var <- snakemake@params$var
}

unique_metrics <- unique(result$metric)
unique_tasks <- unique(result$task)


# Omnibus tests with Friedman
omnibus_result <- lapply(unique_metrics, \(m) {
  lapply(unique_tasks, \(t) {
    current <- filter(result, metric == m, task == t)
    test <- friedman_test_wrapper(current, "metric", var)
    tibble(metric = m, task = t, p_value = test$p_value_alt)
  }) |>
    bind_rows()
}) |>
  bind_rows() |>
  mutate(p_adjust = p.adjust(p_value))

significant <- omnibus_result |> filter(p_adjust <= 0.01)

# Follow-up post-hoc tests with Wilcoxon
post_hoc_results <- apply(significant, 1, \(row) {
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

## * Variation plot

plot_metric_variation <- function(tb) {
  tb |>
    ggplot(aes(x = model, y = value, color = metric)) +
    geom_boxplot() +
    facet_wrap(~task)
}

plot <- plot_metric_variation(result) +
  labs(title = glue("Metric variation across `{var}`"))


if (exists("snakemake")) {
  write_csv(omnibus_result, snakemake@output$omnibus)
  write_csv(post_hoc_results, snakemake@output$post_hoc)
  ggsave(snakemake@output$metric_plot, plot)
}
