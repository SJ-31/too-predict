library(tidyverse)
library(glue)
library(here)
library(paletteer)
source(here("src", "R", "utils.R"))
source(here("src", "R", "plotting.R"))

outdir <- here("data", "output", "cross_validation")
dirs <- list.files(here("data", "output", "cross_validation"), full.names = TRUE) |>
  keep(\(x) dir.exists(x) & (length(list.files(x)) > 0)) |>
  discard(\(x) basename_no_ext(x) %in% c("test", "confusion_matrices", "additional_splits"))
label <- "tumor_type"

combined <- lapply(dirs, \(dir) {
  model <- basename(dir)
  file <- glue("{dir}/additional_splits/{label}-misc.csv")
  if (file.exists(file)) {
    read_csv(file) |> mutate(model = model)
  } else {
    NULL
  }
}) |>
  bind_rows()
write_csv(combined, here(outdir, "additional_splits", glue("{label}_combined.csv")))

best_spec <- list(CHULA = 4, CPTAC = 3, GEO = 3, CGCI = 3)
bests <- lapply(names(best_spec), \(x) {
  count <- best_spec[[x]]
  combined |>
    filter(test_set == x & !grepl("combat_", model)) |>
    slice_max(acc, n = count) |>
    pull(model)
}) |>
  unlist() |>
  unique()

acc_comparison <- combined |>
  filter(model %in% bests) |>
  ggplot(aes(x = factor(model, levels = bests), y = acc, fill = test_set)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold")
  ) +
  xlab("Model") +
  ylab("Accuracy") +
  scale_fill_paletteer_d("awtools::a_palette")
ggsave(acc_comparison, filename = here(outdir, "additional_splits_compare.png"), height = 9, width = 12, units = "in")


## * Confusion matrix

# Going across the rows shows the true positives
# Down columns are predictions
chosen_model_dir <- here(outdir, "clr_xgboost_edger_per_type_ovp", "additional_splits")
mfile <- here(chosen_model_dir, "tumor_type-cm_cm-CHULA_NO_CPTAC.csv")
mat <- local({
  m <- read_csv(mfile) |> column_to_rownames(var = "...1")
  predictions <- colSums(m)
  m[predictions != 0, predictions != 0]
})

library(gt)
rownames_to_column(mat, var = "Truth") |>
  gt(rowname_col = "Truth")

cm_plot <- mat |>
  rownames_to_column(var = "True") |>
  as_tibble() |>
  pivot_longer(cols = -True, names_to = "Prediction", values_to = "Count") |>
  mutate(Count = replace(Count, Count == 0, NA)) |>
  ggplot(aes(x = Prediction, y = True, fill = Count)) +
  theme_minimal() +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(angle = 90, vjust = 0.5),
    axis.title.x = element_text(face = "bold", size = 10),
    axis.title.y = element_text(face = "bold", size = 10)
  ) +
  geom_tile() +
  geom_text(aes(label = Count, size = 5)) +
  scale_fill_paletteer_c("ggthemes::Green") +
  guides(size = "none")
ggsave(cm_plot,
  filename = here(chosen_model_dir, "additional_splits_cm.png"),
  height = 12, width = 12
)

## * Sample origin contingency table

obs_data <- read_csv(here("data", "training_data_obs.csv")) |>
  mutate(Project = case_when(
    grepl("^TARGET", Project_ID) ~ "TARGET",
    grepl("^TCGA", Project_ID) ~ "TCGA",
    grepl("^CPTAC", Project_ID) ~ "CPTAC",
    grepl("^CGCI", Project_ID) ~ "CGCI",
    grepl("^CHULA", Project_ID) ~ "CHULA",
    grepl("^GSE", Project_ID) ~ "GEO study",
  ))
