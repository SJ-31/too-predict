library(tidyverse)
library(here)
library(glue)
library(httr2)
library(BiocParallel)
library(data.table)
source(here("src", "R", "utils.R"))

bp_param <- MulticoreParam(workers = multicoreWorkers())

refdir <- here("data", "reference")

# Final output of this script is a list mapping cell types to vectors of
# ensembl gene ids
# there should be an associated metadata tb with at least the following columns:
# - ensembl_gene_id
# - tissue
# - cell_type
# - from_tme (whether or not the associated cell type was found in a tumor sample)
wanted_cols <- c("ensembl", "tissue", "cell_type", "from_tme")

source_tbs <- list()

symbol2ensembl <- local({
  grch38_ref <- read_tsv(here("data", "reference", "Homo_sapiens.GRCh38.113.gene_id_mapping.tsv")) |>
    filter(!is.na(gene_name)) |>
    rename(symbol = gene_name, ensembl = gene_id)
  hgnc_ref <- read_tsv(here("data", "hgnc_complete_set_2025-3-19.tsv")) |>
    rename(ensembl = ensembl_gene_id) |>
    filter(!is.na(symbol) & !is.na(ensembl) & !(symbol %in% grch38_ref$symbol))
  bind_rows(hgnc_ref, grch38_ref) |>
    select(symbol, ensembl) |>
    distinct(symbol, .keep_all = TRUE)
})

# TODO: only do this at the end when unifying all source data
singularize_cells <- function(cells) {
  mapping <- list(
    "neutrophils" = "neutrophil",
    "hepatocytes" = "hepatocyte",
    "myocytes" = "myocyte",
    "podocytes" = "podocyte",
    "spermatocytes" = "spermatocyte",
    "myocytes" = "myocytes",
    "cardiomyocytes" = "cardiomyocyte",
    "fibroblasts" = "fibroblast",
    "enterocytes" = "enterocyte",
    "spermatids" = "spermatid",
    "adipocytes" = "adipocyte",
    "fibroblasts" = "fibroblast",
    "macrophages" = "macrophage",
    "cells" = "cell"
  )
  bplapply(cells, \(cell) {
    for (n in names(mapping)) {
      if (str_detect(cell, n)) {
        return(str_replace(cell, n, mapping[[n]]))
      }
      title <- str_to_title(n)
      if (str_detect(cell, title)) {
        return(str_replace(cell, title, mapping[[n]]))
      }
    }
    cell
  }, BPPARAM = bp_param)
}


## ** Files

files <- list(
  cellmarker2 = here(refdir, "CellMarker2_2025-4-18.csv"),
  panglaodb = here(refdir, "PanglaoDB_markers_27_Mar_2020.tsv"),
  bcscdb = here(refdir, "CSC_Biomarker_2022_All.csv"),
  hpa_all = here(refdir, "hpa_tissue_cell_resource_2025-4-18.csv"),
  hpa_ref = here(refdir, "hpa_ref_transcripts.csv")
)

wanted_cells <- here(refdir, "cellmarker2_wanted.yaml") |> yaml::read_yaml()

## * Markers from CellMarker2

# It's not ideal that you have fold specific cell subsets into one another, but
# this is for maximum compatibility with other sources
markers <- read_csv(files$cellmarker2) |>
  mutate(
    cell_name = case_match(cell_name,
      "Regulatory T(Treg) cell" ~ "Regulatory T (Treg) cell",
      "Natural killer T(NKT) cell" ~ "Natural killer T (NKT) cell",
      "Fat cell (adipocyte)" ~ "Adipocyte",
      "White adipocyte" ~ "Adipocyte",
      "Brown adipocyte" ~ "Adipocyte",
      "Beige adipocyte" ~ "Adipocyte",
      "lymphatic endothelial cell" ~ "Lymphatic endothelial cell",
      .default = cell_name
    ),
    tissue = str_replace(str_to_lower(tissue_class), " ", "_"),
    from_tme = cancer_type != "Normal"
  ) |>
  left_join(symbol2ensembl, by = join_by(x$marker == y$symbol))



source_tbs$cellmarker2 <- markers |>
  filter(cell_name %in% wanted_cells$common | tissue_class %in% names(wanted_cells) | from_tme) |>
  filter(!is.na(ensembl)) |>
  mutate(
    cell_type = cell_name,
    tissue = case_when(cell_type %in% wanted_cells$common ~ "common",
      .default = tissue
    )
  ) |>
  select(all_of(wanted_cols))


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

## * PanglaoDB

panglaodb <- read_tsv(files$panglaodb) |>
  filter(str_detect(species, "Hs") & !is.na(organ) & !is.na(`canonical marker`)) |>
  inner_join(symbol2ensembl, by = join_by(x$`official gene symbol` == y$symbol)) |>
  mutate(organ_ct = paste0(organ, `cell type`)) |>
  group_by(organ_ct) |>
  mutate(agg_sensitivity_hs = median(sensitivity_human, na.rm = TRUE))

# `sensitivity` refers to how frequently the marker is expressed in this particular cell type

# Select top 3 cell types from each organ that have the highest median sensitivity for
# their marker genes
source_tbs$panglaodb <- panglaodb |>
  mutate(tissue = str_to_lower(case_match(
    organ,
    "Immune system" ~ "common", "Thyroid" ~ "thyroid_gland",
    "Skeletal muscle" ~ "skeletal muscle",
    "Adrenal glands" ~ "adrenal_glands",
    .default = organ
  ))) |>
  filter(tissue %in% names(wanted_cells)) |>
  group_by(tissue) |>
  nest() |>
  mutate(data = lapply(data, \(tb) {
    slice_max(tb, order_by = agg_sensitivity_hs, n = 3, with_ties = FALSE)
  }), from_tme = FALSE) |>
  unnest(cols = c(data)) |>
  ungroup() |>
  rename(cell_type = "cell type") |>
  select(all_of(wanted_cols))


## * HTC atlas

## TODO: check out the files in htc atlas

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
      tb <- as_tibble(fread(resp_body_string(resp))) |> mutate(
        query_tissue = t,
        cell_type = c
      )
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

hpa_tb <- read_existing(files$hpa_all, get_hpa, read_csv) |> mutate(cell_type = str_replace(cell_type, "_[213]\\*", ""))



source_tbs$hpa <- hpa_tb |>
  filter(`RNA cancer specificity` %in% c("Low cancer specificity", "Not detected")) |>
  mutate(tissue = str_replace(query_tissue, " ", "_"), from_tme = FALSE) |>
  rename(ensembl = Ensembl) |>
  select(all_of(wanted_cols))
# NOTE: [2025-04-21 Mon] Ideally you would  also use this resource to
# find marker genes for TME-associated
# cells, but the relevant columns ("RNA cancer distribution", "RNA cancer specificity" etc.)
# are confusing and the docs don't explain how they were calculated

# Reference transcripts used in authors' analysis to identify other enriched transcripts
hpa_custom_ref_transcripts <- read_csv(files$hpa_ref) |>
  mutate(cell_type = str_replace(cell_type, "_[213]\\*", "")) |>
  inner_join(symbol2ensembl, by = join_by(symbol))

source_tbs$hpa_ref <- hpa_custom_ref_transcripts |>
  mutate(tissue = case_match(
    tissue,
    "Breast (female)" ~ "breast",
    "Kidney (cortex)" ~ "kidney",
    "Skin (non-sun exposed)" ~ "skin",
    .default = str_to_lower(tissue) |> str_replace(" ", "_")
  ), from_tme = FALSE) |>
  select(all_of(wanted_cols))

## * Cancer stem cells
# Data from https://academic.oup.com/database/article/doi/10.1093/database/baac082/6725752?login=true
# How reliable is this?
bcscdb <- read_csv(files$bcscdb,
  col_names = c(
    "symbol", "marker_type", "expression_level",
    "hgnc_id", "cancer_type",
    "histological_type", "cell_line", "csc_enrichment", "method", "confidence_scoring",
    "global_scoring", "pubmed"
  )
)


## * Collate resources

## --- CODE BLOCK ---

## * Quantify overlap

# TODO: quantify overlap between cell marker sets
# TODO: add M1 and M2 to TAMs
# TODO: for immune cells, get markers for both the normal and tumor-associated versions
