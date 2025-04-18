library(tidyverse)
library(here)
library(glue)
library(httr2)
library(data.table)

source(here("src", "R", "utils.R"))

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

## ** Setup

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
  breast = c("Adipocytes (Breast)", "Endothelial cells", "Plasma cells"),
  colon = c("Colon enterocytes", "Colon enteroendocrine cells", "Enteric glia cells"),
  skin = c("Keratinocyte (other)", "Keratinocyte (granular)", "Sebaceous gland cells"),
  `heart muscle` = c("Cardiomyocytes", "Fibroblasts", "Endothelial cells"),
  kidney = c("Proximal tubular cells", "Podocytes", "Ascending Loop of Henle cells"),
  liver = c("Hepatocytes", "Hepatic stellate cells", "Erythroid cells"),
  lung = c("Respiratory ciliated cells", "Plasma cells", "Mitotic cells (Lung)"),
  pancreas = c("Alpha cells", "Beta cells", "Exocrine glandular cells"),
  prostate = c("Prostate glandular cells", "Smooth muscle cells", "Endothelial cells"),
  `skeletal muscle` = c("Myocytes", "fibroblasts", "Macrophages"),
  testis = c("Early spermatids", "Late spermatids", "Spermatocytes"),
  `thyroid gland` = c("Glandular cells", "Parafollicular cells", "Plasma cells")
)

hpa_specificity_map <- list(
  # "High" for heart because few "Very high"
  `heart muscle Cardiomyocytes` = "High",
  `heart muscle Fibroblasts` = "High",
  `heart muscle Endothelial cells` = "High",
  `kidney Ascending Loop of Henle cells` = "High",
  `skeletal muscle Macrophages` = "High",
  `skeletal muscle fibroblasts` = "High"
)


hpa_aliases <- list(
  `skeletal muscle Myocytes` = "Skeletal myocytes_1,Skeletal myocytes_2,Skeletal myocytes_3",
  `thyroid gland Glandular cells` = "Thyroid glandular cells_1,Thyroid glandular cells_2",
  `kidney Proximal tubular cells` = "Proximal tubular cells_1,Proximal tubular cells_2",
  `liver Hepatocytes` = "Hepatocyte_1,Hepatocyte_2"
)

## TODO: you should check if the above cell types for each tissue make sense in the
# context of TME

## ** Retrieve data

hpa_all_file <- here("data", "reference", "hpa_tissue_cell_resource.csv")
get_hpa <- function(file) {
  hpa_all <- lapply(names(hpa_tissues), \(t) {
    cell_list <- hpa_tissues[[t]]
    lapply(cell_list, \(c) {
      specificity_level <- lget(hpa_specificity_map, glue("{t} {c}"), "Very high")
      cname <- lget(hpa_aliases, glue("{t} {c}"), c)
      q <- glue("ce_enriched:{t};{cname};{specificity_level}")
      url <- get_hpa_query(q, format = "tsv", columns = hpa_wanted_cols, compress = "no")
      Sys.sleep(1)
      resp <- request(url) |> req_perform()
      tb <- as_tibble(fread(resp_body_string(resp))) |> mutate(query_tissue = t, cell_type = c)
      tryCatch(
        expr = {
          stopifnot(nrow(tb) != 0)
          tb
        },
        error = function(cnd) {
          print(glue("Request for {t} {c} has an empty tibble"))
          print(url)
          tibble()
        }
      )
    }) |>
      bind_rows()
  }) |>
    bind_rows()
  write_csv(hpa_all, file)
  hpa_all
}
hpa_tb <- read_existing(hpa_all_file, get_hpa, read_csv)

# Reference transcripts used in authors' analysis
hpa_custom_ref_transcripts <- read_csv(here("data", "reference", "hpa_ref_transcripts.csv"))


## --- CODE BLOCK ---

## * Quantify overlap

# TODO: quantify overlap between cell marker sets
