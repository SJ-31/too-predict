library(ensembldb)
library(Seurat)
library(here)
library(tidyverse)
library(edgeR)
library(glue)
library(paletteer)
library(reticulate)
use_condaenv("too-predict")

test <- FALSE
if (sys.nframe() == 0) {
  library("optparse")
  parser <- OptionParser()
  parser <- add_option(parser, c("-t", "--test"), type = "logical", help = "test", default = FALSE)
  args <- parse_args(parser)
  test <- args$test
}

if (path.expand("~") != "/home/shannc" && !test) {
  storage_dir <- here("remote", "public_data")
} else {
  storage_dir <- here("data", "tests", "scr_ref")
}
AnnotationHub::setAnnotationHubOption("CACHE", here("data", ".AnnotationHub"))
ad <- import("anndata")
ut <- import("too_predict.utils")

dirs <- list(
  htc_atlas = here(storage_dir, "htca_2025-4-21"),
  gtex = here(storage_dir, "GTEx_single_cell_2025-4-29"),
  cellxgene = here(storage_dir, "cellxgene-census")
)
to_ignore <- list(
  htc_atlas = c(
    "HTCA_ADULT_SMALL_INTESTINE.rds",
    "HTCA_ADULT_SPLEEN.rds",
    "HTCA_ADULT_STOMACH.rds",
  ),
  gtex = c(),
  cellxgene = c(),
)
# NOTE: probably don't need tissue_mapping anymore
tissue_mapping <- list(
  htc_atlas = local({
    files <- list.files(dirs$htc_atlas) |> as.list()
    tissues <- str_to_lower(files) |> str_extract("htca_adult_(.*).rds", group = 1)
    setNames(files, tissues)
  }),
  gtex = list(),
  cellxgene = list()
)

## * Sources

# read_fn has two arguments: filename and tissue. It returns a Seurat object and should
# also add a column "source" describing project
merge_seurat_from_files <- function(dir, ignore_list, tissue_map, read_fn) {
  file_list <- list.files(dir) |> discard(\(x) x %in% ignore_list)
  fnames <- tools::file_path_sans_ext(file_list)
  objs <- lapply(file_list, \(x) {
    read_fn(paste(dir, "/", x))
  })
  if (length(objs) > 1) {
    merge(objs[[1]], objs[2:length(objs)], add.cells.ids = fnames)
  } else {
    objs
  }
}

## ** HTC atlas

symbol2ensembl <- local({
  tb <- read_tsv(here("data", "mappings", "ensembl_113_id_mapping.tsv")) |> distinct(ensembl, .keep_all = TRUE)
  setNames(tb$ensembl, tb$symbol)
})

ensdb <- EnsDb(as.character(ut$get_data("reference/Homo_sapiens.GRCh38.113.sqlite")))

htca_fn <- function(file) {
  obj <- readRDS(here(dirs$htc_atlas, file))
  obj[[]]$source <- paste0("htca-", obs[[]]$Project)
  obj[[]] <- obj[[]] |>
    dplyr::rename(tissue = Tissue, subject = Sample_ID, cell_type = Cell_Type) |>
    select(tissue, subject, cell_type, source)
  obj[["RNA"]][[]]$GENENAME <- rownames(obj[["RNA"]][[]])
  obj <- rename_seurat_features(obj, symbol2ensembl, mapping = TRUE)
  var_meta <- AnnotationDbi::select(ensdb,
    keys = rownames(obj), columns = c("GENEBIOTYPE", "SEQLENGTH"),
    keytype = "GENEID"
  )
  obj[["RNA"]][[]]$GENEID <- rownames(obj)
  obj[["RNA"]][[]] <- left_join(obj[["RNA"]][[]], var_meta, by = join_by(GENEID))
  obj
}

## ** Cellxgene

dataset_id_map <- local({
  tb <- read_csv(here("data", "mappings", "cellxgene_datasets.csv"))
  setNames(tb$collection_name, tb$dataset_id)
})

cellxgene_fn <- function(file) {
  adata <- ad$read_h5ad(file)
  adata <- adata[, !is.na(adata$var$feature_id)]
  rownames(adata$var) <- adata$var$feature_id
  mapping <- c(
    "feature_id" = "GENEID", "feature_type" = "GENEBIOTYPE", "feature_length" = "SEQLENGTH",
    "feature_name" = "GENENAME"
  )
  adata$var <- adata$var |>
    dplyr::rename(all_of(mapping)) |>
    select(all_of(mapping))
  adata$obs$source <- paste0("cellxgene", "-", dataset_id_map[adata$obs$dataset_id])
  adata$obs$subject <- paste0(adata$obs$dataset_id, "-", adata$obs$donor_id)
  adata$obs <- adata$obs |> select(cell_type, tissue, source, subject)
  adata2seurat(adata)
}

## ** GTEx

gtex_fn <- function(file) {
  adata <- ad$read_h5ad(file)
  adata$X <- adata$layers$counts
  wanted_cols <- c(
    "tissue", "Participant ID", "Cell types level 2",
    "batch", "prep", "Tissue Site Detail", "Broad cell type",
    "Granular cell type", "Tissue composition", "PercentMito", "PercentRibo", "scrublet"
  )
  adata$obsp <- NULL
  adata$obsm <- NULL
  adata$varm <- NULL
  adata <- adata[!as.logical(adata$obs$scrublet), ]
  adata$obs <- select(adata$obs, any_of(wanted_cols)) |>
    dplyr::rename_with(\(x) str_to_lower(str_replace_all(x, " ", "_"))) |>
    mutate(granular_cell_type = str_extract(granular_cell_type, "\\((.*)\\)", group = 1))
  adata$obs$source <- "GTEx"
  adata <- adata[, !is.na(adata$var$gene_ids)]
  rownames(adata$var) <- adata$var$gene_ids
  adata$var <- adata$var |>
    select("gene_ids", "gene_name", "gene_biotype", "gene_length") |>
    dplyr::rename_with(\(x) str_replace(str_to_upper(x), "_", "")) |>
    dplyr::rename(SEQLENGTH = "GENELENGTH", GENEID = "GENEIDS") |>
    select(SEQLENGTH, GENEID, GENENAME, GENEBIOTYPE)
  adata2seurat(adata)
}


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

# var columns: GENEID, GENEBIOTYPE, SEQLENGTH
# obs columns: cell_type, tissue, subject, source

read_fns <- list(
  htc_atlas = htca_fn,
  gtex = gtex_fn,
  cellxgene = cellxgene_fn,
)

all_objs <- sapply(names(dirs), \(x) {
  merge_seurat_from_files(dirs[[x]],
    ignore_list = to_ignore[[x]],
    tissue_map = tissue_map,
    read_fn = read_fns[[x]]
  )
}, simplify = FALSE, USE.NAMES = TRUE)
combined <- merge(all_objs[[1]], all_objs[2:length(all_objs)], add.cells.ids = names(all_objs))
write_csv(combined[[]], here("data", "reference", "sc_ref_all_obs.csv"))
## SaveSeuratRds(combined, file = here(storage_dir, "sc_ref_all.rds"))
