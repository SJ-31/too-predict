library(Seurat)
library(here)
library(tidyverse)
library(edgeR)
library(glue)
library(paletteer)
library(reticulate)
use_condaenv("too-predict")

if (path.expand("~") != "/home/shannc") {
  storage_dir <- here("remote", "public_data")
} else {
  storage_dir <- here("data", "tests", "scr_ref")
}
AnnotationHub::setAnnotationHubOption("CACHE", here("data", ".AnnotationHub"))
ad <- import("anndata")

dirs <- list(
  htc_atlas = here(storage_dir, "htca_2025-4-21"),
  gtex = here(storage_dir, "GTEx_single_cell_2025-4-29"),
  tabula = here(storage_dir, "Tabula_Sapiens_V2_2025-4-29")
  # TODO: complete the python script for getting the cellxgene data which includes
  # Tabula
  ## sc_atlas = here(storage_dir, "singlecellatlas_2025-4-28")
)
to_ignore <- list(
  htc_atlas = c(
    "HTCA_ADULT_SMALL_INTESTINE.rds",
    "HTCA_ADULT_SPLEEN.rds",
    "HTCA_ADULT_STOMACH.rds",
  ),
  gtex = c(),
)
tissue_mapping <- list(
  htc_atlas = local({
    files <- list.files(dirs$htc_atlas) |> as.list()
    tissues <- str_to_lower(files) |> str_extract("htca_adult_(.*).rds", group = 1)
    setNames(files, tissues)
  }),
  gtex = list(),
  tabula = list()
)

## * Sources

# read_fn has two arguments: filename and tissue. It returns a Seurat object and should
# also add a column "source" describing project
merge_seurat_from_files <- function(dir, ignore_list, tissue_map, read_fn) {
  file_list <- list.files(dir) |> discard(\(x) x %in% ignore_list)
  fnames <- tools::file_path_sans_ext(file_list)
  objs <- lapply(file_list, \(x) {
    read_fn(paste(dir, "/", x), tissue_map[[x]])
  })
  if (length(objs) > 1) {
    merge(objs[[1]], add.cells.ids = fnames)
  } else {
    objs
  }
}

## ** HTC atlas

htca_fn <- function(file, tissue) {
  data <- readRDS(here(dirs$htc_atlas, file))
  ## avg <- AverageExpression(test, group.by = "Cell_Type", return.seurat = TRUE, layer = "counts")
  data[[]]$source <- "htca"
  data[[]]$tissue <- tissue
  data
}

## ** Tabula Sapiens

# Will just get this from cellxgene_census
tabula_fn <- function(file, tissue) {
  data <- adata2seurat(file)
  data[[]]$source <- "tabula"
  data[[]]$tissue <- tissue
  data
}

## ** GTEx

# TODO:

## ** Single Cell Atlas
# REVIEW: Files are matrices, but no annotation. Should only use this as last resort

## ** Celldex
# TODO: should you even use this? Might be better to just use only single-cell
# data, which you can integrate more easily
library(celldex)
# lf for label.fine, lm for label.main
wanted_cells <- list(
  hpca_lf = c(
    "T_cell:gamma-delta", "T_cell:Treg:Naive", "Endothelial_cells:blood_vessel",
    "Endothelial_cells:lymphatic"
  ),
  hpca_lm = c(
    "MSC", "Neurons", "Neutrophils", "Macrophage", "Monocyte", "B_cell",
    "NK_cell", "Platelets"
  ),
  encode_lm = c(
    "Adipocytes", "DC", "CD4+ T-cells", "CD8+ T-cells", "B-cells",
    "Eosinophils", "Macrophages", "Monocytes", "Neurons", "Neutrophils", "Pericytes",
    "NK cells"
  ),
  encode_lf = c("Tregs")
)

hpca <- HumanPrimaryCellAtlasData(ensembl = TRUE)
encode <- BlueprintEncodeData(ensembl = TRUE)
# Can't use DICE due to incompatible reference genome

hpca_mask <- (colData(hpca)$label.fine %in% wanted_cells$hpca_lf) | (colData(hpca)$label.main %in% wanted_cells$hpca_lm)
encode_mask <- (colData(encode)$label.fine %in% wanted_cells$encode_lf) |
  (colData(encode)$label.main %in% wanted_cells$encode_lm)

shared_genes <- intersect(rownames(rowData(hpca)), rownames(rowData(encode)))
hpca_f <- hpca[rownames(rowData(hpca)) %in% shared_genes, hpca_mask]
colData(encode_f)$source <- "encode"
colData(hpca_f)$source <- "hpca"
encode_f <- encode[rownames(rowData(encode)) %in% shared_genes, encode_mask]
together <- cbind(encode_f, hpca_f)
colData(together)$cell_type <- with(colData(together), case_when(
  str_to_lower(label.fine) %in% c("t_cell:treg:naive", "tregs") ~ "treg",
  str_to_lower(label.fine) %in% c("t_cell:gamma-delta") ~ "gamma_delta_t_cell",
  str_to_lower(label.fine) %in% c("endothelial_cells:blood_vessel") ~ "vascular_endothelial_cell",
  str_to_lower(label.fine) %in% c("endothelial_cells:lymphatic") ~ "lymphatic_endothelial_cell",
  str_to_lower(label.main) %in% c("neutrophils") ~ "neutrophil",
  str_to_lower(label.main) %in% c("monocytes", "monocyte") ~ "monocyte",
  str_to_lower(label.main) %in% c("cd4+ t-cells") ~ "cd4+_t_cell",
  str_to_lower(label.main) %in% c("cd8+ t-cells") ~ "cd8+_t_cell",
  str_to_lower(label.main) %in% c("nk cells", "nk_cell") ~ "natural_killer_cell",
  str_to_lower(label.main) %in% c("b-cells", "b_cell") ~ "b_cell",
  str_to_lower(label.main) %in% c("macrophages", "macrophage") ~ "macrophage",
  str_to_lower(label.main) %in% c("dc") ~ "dendritic_cell",
  str_to_lower(label.main) %in% c("eosinophils") ~ "eosinophil",
  str_to_lower(label.main) %in% c("adipocytes") ~ "adipocyte",
  str_to_lower(label.main) %in% c("neurons") ~ "neuron",
  str_to_lower(label.main) %in% c("pericytes") ~ "pericyte",
  str_to_lower(label.main) %in% c("platelets") ~ "platelet",
  str_to_lower(label.main) %in% c("msc") ~ "mesenchymal_stem_cell",
  TRUE ~ str_to_lower(label.main)
))


## * Combine
# Plan is to aggregate all the cell data together into one Seurat object
# Combine and remove batch effect

read_fns <- list(
  htc_atlas = htca_fn,
  gtex = print,
  tabula = print,
)

all_objs <- sapply(names(dirs), \(x) {
  merge_seurat_from_files(dirs[[x]],
    ignore_list = to_ignore[[x]],
    tissue_map = tissue_map,
    read_fn = read_fns[[x]]
  )
}, simplify = FALSE, USE.NAMES = TRUE)
