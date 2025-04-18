library(tidyverse)
library(here)
library(glue)
library(httr2)
library(data.table)

markers <- read_csv(here("data", "reference", "CellMarker2.csv"))
wanted_cells <- c()

# Map of old = New
rename <- c(
  "Regulatory T(Treg) cell" = "Regulatory T (Treg) cell",
  "Natural killer T(NKT) cell" = "Natural killer T (NKT) cell",
  "Fat cell (adipocyte)" = "Adipocyte",
  "White adipocyte" = "Adipocyte",
  "Brown adipocyte" = "Adipocyte",
  "Beige adipocyte" = "Adipocyte",
  "lymphatic endothelial cell" = "Lymphatic endothelial cell"
)

# Find tissue-specific
tissues2cells <- sapply(unique(markers$tissue_class), \(t) {
  markers |>
    filter(tissue_class == t) |>
    pluck("cell_type") |>
    unique()
}, simplify = FALSE)

# [2025-04-18 Fri] As expected, got nothing from this
tissue_specific <- sapply(names(tissues2cells), \(n) {
  other_tissues <- tissues2cells[names(tissues2cells) != n]
  the_rest <- unlist(other_tissues)
  setdiff(tissues2cells[[n]], the_rest)
}, simplify = FALSE, USE.NAMES = TRUE)

## * Enriched from hpa
## https://www.proteinatlas.org/humanproteome/single+cell/tissue+cell+type
## --- CODE BLOCK ---
get_hpa_query <- function(query, ...) {
  base_url <- "www.proteinatlas.org/api/search_download.php?"
  string <- url_query_build(list(search = query, ...),
    .multi = "comma"
  )
  paste0(base_url, string)
}

hpa_wanted_cols <- c(
  "eg", # Ensembl
  "evih", # HPA evidence
  "rnats", # RNA tissue specificity
  "rnatd", # RNA tissue distribution
  "rnatss", # RNA tissue specificity score
  "rnatsm", # RNA tissue specific nTPM
  "rnacas", # RNA cancer specificity
  "rnacad", # RNA cancer distribution
  "rnacass", # RNA cancer specificity score
  "rnacasm" # RNA cancer specific FPKM
)

hpa_tissues <- list(
  Breast = c("Adipocytes (Breast)", "Endothelial cells", "Plasma cells"),
  Colon = c("Colon enterocytes", "Colon enteroendocrine cells", "Enteric glia cells"),
  Skin = c("Keratinocyte (other)", "Keratinocyte (granular)", "Sebaceous gland cells"),
  `Heart muscle` = c("Cardiomyocytes", "Fibroblasts", "Endothelial cells"),
  Kidney = c("Proximal tubular cells", "Podocytes", "Ascending Loop of Henle cells"),
  Liver = c("Hepatocytes", "Hepatic stellate cells", "Erythroid cells"),
  Lung = c("Respiratory ciliated cells", "Plasma cells", "Mitotic cells (Lung)"),
  Pancreas = c("Alpha cells", "Beta cells", "Exocrine glandular cells"),
  Prostate = c("Prostate glandular cells", "Smooth muscle cells", "Endothelial cells"),
  `Skeletal muscle` = c("Skeletal myocytes", "Fibroblasts", "Macrophages"),
  Testis = c("Early spermatids", "Late spermatids", "Spermatocytes"),
  `Thyroid gland` = c("Thyroid glandular cells", "Parafollicular cells", "Plasma cells")
)
names(hpa_tissues) <- names(hpa_tissues) |> str_to_lower()

hpa_specificity_map <- list(
  # "High" for heart because few "Very high"
  `Heart muscle Cardiomyocytes` = "High",
  `Heart muscle Fibroblasts` = "High",
  `Heart muscle Endothelial cells` = "High",
  `Kidney Ascending Loop of Henle cells` = "High",
  `Skeletal muscle Macrophages` = "High"
)

## TODO: you should check if the above cell types for each tissue make sense in the
# context of TME

hpa_all <- lapply(names(hpa_tissues), \(t) {
  cell_list <- hpa_tissues[[t]]
  lapply(cell_list, \(c) {
    specificity_level <- lget(hpa_specificity_map, glue("{t} {c}"), "Very high")
    url <- get_hpa_query(glue("ce_enriched:{t};{c};{specificity_level}"),
      format = "tsv",
      columns = hpa_wanted_cols,
      compress = "no"
    )
    Sys.sleep()
    resp <- request(url) |> req_perform()
    tb <- resp_body_string(resp) |>
      fread() |>
      as_tibble() |>
      mutate(
        query_tissue = t,
        cell_type = str_replace_all(glue("{t}_{c}"), "[() ]", "_")
      )
    tryCatch(
      expr = {
        stopifnot(nrow(tb) != 0)
        tb
      },
      error = function(cnd) {
        print(glue("Request for {t} {c} has an empty tibble"))
        tibble()
      }
    )
  }) |>
    bind_rows()
}) |>
  bind_rows()

## debug <- "www.proteinatlas.org/api/search_download.php?search=ce_enriched%3Abreast%3BAdipocytes%20%28Breast%29%3BVery%20high&format=tsv&columns=eg,evih,rnats,rnatd,rnatss,rnatsm,rnacas,rnacad,rnacass,rnacasm&compress=no"

## --- CODE BLOCK ---

## * Quantify overlap

# TODO: quantify overlap between cell marker sets
