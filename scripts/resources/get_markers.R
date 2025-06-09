library(tidyverse)
library(here)
library(glue)
library(httr2)
library(data.table)
source(here("src", "R", "utils.R"))
source(here("src", "R", "plotting.R"))

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
  grch38_ref <- read_tsv(here(
    "data",
    "reference",
    "Homo_sapiens.GRCh38.113.gene_id_mapping.tsv"
  )) |>
    filter(!is.na(gene_name)) |>
    dplyr::rename(symbol = gene_name, ensembl = gene_id)
  hgnc_ref <- read_tsv(here("data", "hgnc_complete_set_2025-3-19.tsv")) |>
    dplyr::rename(ensembl = ensembl_gene_id) |>
    filter(!is.na(symbol) & !is.na(ensembl) & !(symbol %in% grch38_ref$symbol))
  bind_rows(hgnc_ref, grch38_ref) |>
    select(symbol, ensembl) |>
    distinct(symbol, .keep_all = TRUE)
})

## ** Files

files <- list(
  cellmarker2 = here(refdir, "CellMarker2_2025-4-18.csv"),
  panglaodb = here(refdir, "PanglaoDB_markers_27_Mar_2020.tsv"),
  bcscdb = here(refdir, "CSC_Biomarker_2022_All.csv"),
  hpa_all = here(refdir, "hpa_tissue_cell_resource_2025-4-18.csv"),
  hpa_ref = here(refdir, "hpa_ref_transcripts.csv")
)
final_file <- here(refdir, "cell_markers_custom.yaml")
final_file_metrics <- here(refdir, "cell_markers_custom_metrics.tsv")
final_file_meta <- here(refdir, "cell_markers_custom_meta.tsv")

wanted_cells <- here(refdir, "cellmarker2_wanted.yaml") |> yaml::read_yaml()

## * Markers from CellMarker2

# Fold cell subsets into parent cell types & reconcile aliases
#   Not ideal, but for maximum compatibility with other sources
#   Tumor variants of cells will be renamed to the normal cell type here (the label "tme" will be added later)
markers <- read_csv(files$cellmarker2) |>
  mutate(
    cell_name = case_match(
      cell_name,
      "Regulatory T(Treg) cell" ~ "Treg cell",
      "Natural killer T(NKT) cell" ~ "Natural killer T (NKT) cell",
      "Activated dendritic cell" ~ "Dendritic cell",
      "Plasmacytoid dendritic cell" ~ "Dendritic cell",
      "Activated dendritic cell" ~ "Dendritic cell",
      "Monocyte derived dendritic cell" ~ "Dendritic cell",
      "Migrating dendritic cell" ~ "Dendritic cell",
      "Plasmacytoid dendritic cell(pDC)" ~ "Dendritic cell",
      "Cross-presenting dendritic cell" ~ "Dendritic cell",
      "Pre-dendritic cell(pre-DC)" ~ "Dendritic cell",
      "Dermal dendritic cell" ~ "Dendritic cell",
      "Myeloid dendritic cell" ~ "Dendritic cell",
      "Immature dendritic cell" ~ "Dendritic cell",
      "Mature dendritic cell" ~ "Dendritic cell",
      "Conventional dendritic cell 2(cDC2)" ~ "Dendritic cell",
      "Conventional dendritic cell 1(cDC1)" ~ "Dendritic cell",
      "Conventional dendritic cell(cDC)" ~ "Dendritic cell",
      "Conventional dendritic cell 2b(cDC2b)" ~ "Dendritic cell",
      "Conventional dendritic cell 2a(cDC2a)" ~ "Dendritic cell",
      # NK cells
      "Natural Killer CD56+ dim cell" ~ "Natural killer cell",
      "Natural killer CD56 bright cell" ~ "Natural killer cell",
      "Natural killer CD56 dim cell" ~ "Natural killer cell",
      "Natural killer CD4+ T cell" ~ "Natural killer cell",
      "Circulating natural killer cell" ~ "Natural killer cell",
      "KLRF+ natural killer cell" ~ "Natural killer cell",
      # Adipocytes
      "Fat cell (adipocyte)" ~ "Adipocyte",
      "White adipocyte" ~ "Adipocyte",
      "Brown adipocyte" ~ "Adipocyte",
      "Beige adipocyte" ~ "Adipocyte",
      # CD8 cells
      "Exhausted CD8 + T cell" ~ "Exhausted CD8+ T cell",
      "Transitional exhausted CD8+ T cell" ~ "Exhausted CD8+ T cell",
      "Pre-exhausted CD8+ T cell" ~ "Exhausted CD8+ T cell",
      "Effector CD8 T cell" ~ "Effector CD8+ T cell",
      "Memory CD8 T Cell" ~ "Memory CD8+ T cell",
      "Memory CD8 + T cell" ~ "Memory CD8+ T cell",
      "Effector memory CD8+ T cell" ~ "Memory CD8+ T cell",
      "Central memory CD8+ T cell" ~ "Memory CD8+ T cell",
      "CD8 T cell" ~ "CD8+ T cell",
      "Terminally differentiated CD8+ cell" ~ "CD8+ T cell",
      "Tumor-reactive CD8+ infiltrating T cell" ~ "CD8+ T cell",
      "CD8+ intraepithelial cell" ~ "CD8+ T cell",
      "CD8+ tumor antigen-specific T (Tas) cell" ~ "CD8+ T cell",
      "Memory CD8+ T Cell" ~ "Memory CD8+ T cell",
      # CD4 cells
      # CD4+ t cell == T helper cell (but not of any specific subset)
      "CD4+ T cell" ~ "T helper cell",
      "IL7R T helper cell" ~ "T helper cell",
      "Exhausted CD4+ T cell" ~ "T helper cell",
      "Central memory CD4+ T cell" ~ "T helper cell",
      "Conventional CD4+ T cell" ~ "T helper cell",
      "Conventional CD4 T cell" ~ "T helper cell",
      "Mucosa-associated invariant CD4+ T cell" ~ "T helper cell",
      "Effector CD4+ T cell" ~ "T helper cell",
      "CD40LG+ T helper cell" ~ "T helper cell",
      "CD4+ T helper cell" ~ "T helper cell",
      # Tregs
      "Regulatory T (Treg) cell" ~ "Treg cell",
      "Foxp3+ regulatory T cell" ~ "Treg cell",
      "Tumor regulatory T cell" ~ "Treg cell",
      "Resting regulatory T cell" ~ "Treg cell",
      "Tumor-infiltrating regulatory T cell" ~ "Treg cell",
      "Eomesodermin homolog(EOMES)+ regulatory T cell type 1" ~ "Treg cell",
      "Suppressive regulatory T cell" ~ "Treg cell",
      # T memory
      "Memory T(Tm) cell" ~ "Tm cell",
      "Memory T cell" ~ "Tm cell",
      "Memory double-positive T cell" ~ "Tm cell",
      "Memory T helper 1 cell" ~ "Tm cell",
      "Memory T(Tm) cell" ~ "Tm cell",
      "Naïve or central memory  T cell" ~ "Central memory T cell",
      # CAFs
      "Cancer associated fibroblast(CAF)" ~
        "Cancer associated fibroblast (CAF)",
      "Classical cancer associated fibroblast" ~
        "Cancer associated fibroblast (CAF)",
      "Pan-cancer associated fibroblast" ~ "Cancer associated fibroblast (CAF)",
      "Complement-secreting cancer associated fibroblast" ~
        "Cancer associated fibroblast (CAF)",
      "Myofibroblastic cancer‐associated fibroblast (myCAF)" ~
        "Cancer associated fibroblast (CAF)",
      "Inflammatory cancer‐associated fibroblast (iCAF)" ~
        "Cancer associated fibroblast (CAF)",
      "Cancer-associated fibroblast" ~ "Cancer associated fibroblast (CAF)",
      "Antigen presentation cancer-associated fibroblast" ~
        "Cancer associated fibroblast (CAF)",
      "Myofibroblastic cancer-associated fibroblast" ~
        "Cancer associated fibroblast (CAF)",
      "Inflammatory cancer-associated fibroblast" ~
        "Cancer associated fibroblast (CAF)",
      # B cells
      "Regulatory B(Breg) cell" ~ "Breg cell",
      "B10 Regulatory B cell" ~ "Breg cell",
      "GrB+ Regulatory B cell" ~ "Breg cell",
      "IgA+ Regulatory B cell" ~ "Breg cell",
      "TIM-1+ Regulatory B cell" ~ "Breg cell",
      "PD-1hi Regulatory B cell" ~ "Breg cell",
      # NOTE: Misc.
      "lymphatic endothelial cell" ~ "Lymphatic endothelial cell",
      "Pan-macrophage" ~ "Macrophage",
      "Tissue-resident macrophage" ~ "Macrophage",
      "Infiltrating macrophage" ~ "Macrophage",
      "Classical monocyte" ~ "Monocyte",
      "Malignant cell" ~ "Tumor cell",
      "Foxp3+IL-17+ T cell" ~ "T cell",
      "T-cell lineage" ~ "T cell",
      "Tumor-infiltrating lymphocyte(TIL)" ~ "Lymphocyte",
      "Lymphoid-primed multipotent progenitor cell(LMPP)" ~
        "Multipotent progenitor cell",
      "Pro-tumor type-2 pericyte" ~ "Pericyte",
      "Low-density neutrophil" ~ "Neutrophil",
      "Pan-B cell" ~ "B cell",
      "B cell lineage" ~ "B cell",
      "Primitive stromal cell" ~ "Stromal cell",
      "Tumor-associated microglia cell" ~ "Microglial cell",
      "Microglia-derived tumor-associated macrophage(Mg-TAM)" ~
        "Tumor-associated macrophage (TAM)",
      "Epithelial-mesenchymal transition cancer stem cell" ~
        "EMT cancer stem cell",
      "Pan–T-cell" ~ "T cell",
      "Superpotent cancer stem cell" ~ "Cancer stem cell",
      "pit mucous cell (PMC)" ~ "Pit mucous cell",
      "Proliferating T cell" ~ "T cell",
      "Non-malignant epithelial cell" ~ "Epithelial cell",
      "Tumor epithelial cell" ~ "Epithelial cell",
      "Tumor-infiltrating T cell" ~ "T cell",
      .default = cell_name
    ),
    tissue = str_replace(str_to_lower(tissue_class), " ", "_"),
    from_tme = cancer_type != "Normal"
  ) |>
  left_join(symbol2ensembl, by = join_by(x$marker == y$symbol))

source_tbs$cellmarker2 <- markers |>
  filter(
    cell_name %in%
      wanted_cells$common |
      tissue_class %in% names(wanted_cells) |
      from_tme
  ) |>
  filter(!is.na(Genetype) & !is.na(ensembl) & !is.na(cellontology_id)) |>
  filter(!is.na(ensembl)) |>
  mutate(
    cell_type = cell_name,
    tissue = case_when(
      cell_type %in% wanted_cells$common ~ "common",
      .default = tissue
    )
  ) |>
  select(all_of(wanted_cols))

# Find tissue-specific
tissues2cells <- sapply(
  unique(markers$tissue_class),
  \(t) {
    markers |>
      filter(tissue_class == t) |>
      pluck("cell_type") |>
      unique()
  },
  simplify = FALSE
)

# [2025-04-18 Fri] As expected, got nothing from this
tissue_specific <- sapply(
  names(tissues2cells),
  \(n) {
    other_tissues <- tissues2cells[names(tissues2cells) != n]
    the_rest <- unlist(other_tissues)
    setdiff(tissues2cells[[n]], the_rest)
  },
  simplify = FALSE,
  USE.NAMES = TRUE
)

## * PanglaoDB

panglaodb <- read_tsv(files$panglaodb) |>
  filter(
    str_detect(species, "Hs") & !is.na(organ) & !is.na(`canonical marker`)
  ) |>
  inner_join(
    symbol2ensembl,
    by = join_by(x$`official gene symbol` == y$symbol)
  ) |>
  mutate(organ_ct = paste0(organ, `cell type`)) |>
  group_by(organ_ct) |>
  mutate(agg_sensitivity_hs = median(sensitivity_human, na.rm = TRUE))

# `sensitivity` refers to how frequently the marker is expressed in this particular cell type

# Select top 3 cell types from each organ that have the highest median sensitivity for
# their marker genes
source_tbs$panglaodb <- panglaodb |>
  mutate(
    tissue = str_to_lower(case_match(
      organ,
      "Immune system" ~ "common",
      "Thyroid" ~ "thyroid_gland",
      "Skeletal muscle" ~ "skeletal muscle",
      "Adrenal glands" ~ "adrenal_glands",
      .default = organ
    ))
  ) |>
  filter(tissue %in% names(wanted_cells)) |>
  group_by(tissue) |>
  nest() |>
  mutate(
    data = lapply(data, \(tb) {
      slice_max(tb, order_by = agg_sensitivity_hs, n = 3, with_ties = FALSE)
    }),
    from_tme = FALSE
  ) |>
  unnest(cols = c(data)) |>
  ungroup() |>
  dplyr::rename(cell_type = "cell type") |>
  select(all_of(wanted_cols))


## * Enriched from hpa
## https://www.proteinatlas.org/humanproteome/single+cell/tissue+cell+type
# %%
get_hpa_query <- function(query, ...) {
  base_url <- "www.proteinatlas.org/api/search_download.php?"
  string <- url_query_build(list(search = query, ...), .multi = "comma")
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
  colon = c(
    "Colon enterocytes",
    "Colon enteroendocrine cells",
    "Enteric glia cells"
  ),
  skin = c(
    "Keratinocyte (other)",
    "Keratinocyte (granular)",
    "Sebaceous gland cells"
  ),
  `heart muscle` = c("Cardiomyocytes", "Fibroblasts", "Endothelial cells"),
  kidney = c(
    "Proximal tubular cells",
    "Podocytes",
    "Ascending Loop of Henle cells"
  ),
  liver = c("Hepatocytes", "Hepatic stellate cells", "Erythroid cells"),
  lung = c(
    "Respiratory ciliated cells",
    "Plasma cells",
    "Mitotic cells (Lung)"
  ),
  pancreas = c("Alpha cells", "Beta cells", "Exocrine glandular cells"),
  prostate = c(
    "Prostate glandular cells",
    "Smooth muscle cells",
    "Endothelial cells"
  ),
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
      specificity_level <- lget(
        hpa_specificity_map,
        glue("{t} {c}"),
        "Very high"
      )
      cname <- lget(hpa_aliases, glue("{t} {c}"), c)
      q <- glue("ce_enriched:{t};{cname};{specificity_level}")
      url <- get_hpa_query(
        q,
        format = "tsv",
        columns = hpa_wanted_cols,
        compress = "no"
      )
      Sys.sleep(1)
      resp <- request(url) |> req_perform()
      tb <- as_tibble(fread(resp_body_string(resp))) |>
        mutate(
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

hpa_tb <- read_existing(files$hpa_all, get_hpa, read_csv) |>
  mutate(cell_type = str_replace(cell_type, "_[213]\\*", ""))


source_tbs$hpa <- hpa_tb |>
  filter(
    `RNA cancer specificity` %in% c("Low cancer specificity", "Not detected")
  ) |>
  mutate(tissue = str_replace(query_tissue, " ", "_"), from_tme = FALSE) |>
  dplyr::rename(ensembl = Ensembl) |>
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
  mutate(
    tissue = case_match(
      tissue,
      "Breast (female)" ~ "breast",
      "Kidney (cortex)" ~ "kidney",
      "Skin (non-sun exposed)" ~ "skin",
      .default = str_to_lower(tissue) |> str_replace(" ", "_")
    ),
    from_tme = FALSE
  ) |>
  select(all_of(wanted_cols))

## * Cancer stem cells
# Data from https://academic.oup.com/database/article/doi/10.1093/database/baac082/6725752?login=true
# How reliable is this?
bcscdb <- read_csv(
  files$bcscdb,
  col_names = c(
    "symbol",
    "marker_type",
    "expression_level",
    "hgnc_id",
    "cancer_type",
    "histological_type",
    "cell_line",
    "csc_enrichment",
    "method",
    "confidence_scoring",
    "global_scoring",
    "pubmed"
  )
)


## * Collate resources

combined <- lmap(source_tbs, \(x) mutate(x[[1]], source = names(x))) |>
  bind_rows() |>
  mutate(
    cell_type = str_replace_all(
      str_trim(str_remove(cell_type, "\\(.*\\)")),
      "  ",
      " "
    ),
    cell_type = str_replace_all(str_to_lower(cell_type), " ", "_"),
    cell_type = str_replace(cell_type, "cells$", "cell"),
    cell_type = case_match(
      cell_type,
      "myocytes" ~ "myocyte",
      "adrenergic_neurons" ~ "adrenergic_neuron",
      "adipocytes" ~ "adipocyte",
      "colon_enterocytes" ~ "colon_enterocyte",
      "cardiomyocytes" ~ "cardiomyocyte",
      "fibroblasts" ~ "fibroblast",
      "podocytes" ~ "podocyte",
      "hepatocytes" ~ "hepatocyte",
      "myocytes" ~ "myocyte",
      "macrophages" ~ "macrophage",
      "early_spermatids" ~ "early_spermatid",
      "late_spermatids" ~ "late_spermatid",
      "spermatocytes" ~ "spermatocyte",
      "neutrophils" ~ "neutrophil",
      "skeletal_myocytes" ~ "skeletal_myocyte",
      .default = cell_type
    )
  )

min_markers <- 5
cell_tb <- local({
  tb <- combined |>
    mutate(
      cell_type = case_when(
        from_tme ~ paste0(cell_type, "-tme"),
        .default = cell_type
      ),
      cell_type = paste0(tissue, "-", cell_type)
    ) |>
    select(cell_type, ensembl) |>
    group_by(cell_type) |>
    nest() |>
    filter(map_dbl(data, nrow) >= min_markers) |>
    mutate(
      data = lapply(data, \(x) unique(x$ensembl)),
      marker_count = map_dbl(data, \(x) length(x))
    )
})

cell_list <- setNames(cell_tb$data, cell_tb$cell_type)


## * Quantify and resolve overlaps

new_list <- cell_list

overlap_tracker <- list(n_intersect = c(), larger = c(), smaller = c())
# Only let the smaller of the two intersecting gene sets keep the overlapping genes
for (j in seq_along(new_list)) {
  for (i in seq_along(new_list)) {
    if (j < i) {
      cj <- new_list[[j]]
      ci <- new_list[[i]]
      l_int <- length(intersect(cj, ci))
      if (l_int > 0) {
        overlap_tracker$n_intersect <- c(overlap_tracker$n_intersect, l_int)
        if (length(cj) > length(ci)) {
          new_list[[j]] <- setdiff(cj, ci)
          overlap_tracker$larger <- c(overlap_tracker$larger, names(new_list)[j])
          overlap_tracker$smaller <- c(
            overlap_tracker$smaller,
            names(new_list)[i]
          )
        } else {
          new_list[[i]] <- setdiff(ci, cj)
          overlap_tracker$larger <- c(
            overlap_tracker$larger,
            names(new_list)[i]
          )
          overlap_tracker$smaller <- c(
            overlap_tracker$smaller,
            names(new_list)[j]
          )
        }
      }
    }
  }
}


new_list <- new_list[map_dbl(new_list, length) >= min_markers]
overlap_tb <- as_tibble(overlap_tracker)
yaml::write_yaml(new_list, final_file)
write_tsv(
  tibble(
    set_name = names(new_list),
    marker_count = map_dbl(new_list, length)
  ),
  final_file_metrics
)

combined |>
  filter(ensembl %in% unlist(new_list)) |>
  write_tsv(final_file_meta)

# [2025-04-23 Wed] Only 142 cell types after all the filtering, not great
