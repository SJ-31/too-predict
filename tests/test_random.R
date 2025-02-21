library(BiocParallel)
library(ALDEx2)
library(tidyverse)
library(zellkonverter)
library(scRNAseq)
library(here)

register(MulticoreParam(workers = 2))
source(here("src", "R", "utils.R"))

data <- readH5AD(here("data", "tests", "TCGA_CESC-DLBC-ESCA-GBM.h5ad"))

# TODO: can include the sequencing tech and the tumor type as factors to account
# for their effects
## --- CODE BLOCK ---
group <- "Project_ID"
technical_factors <- c("Sample_Type")

data <- data[1:50, ]

project_id <- colData(data)$Project_ID
mm <- model.matrix(~Project_ID, data = colData(data))

model.matrix(~project_id) == mm

counts <- assays(data)$X
rownames(counts) <- rowData(data)$gene_id
result <- aldex.clr(counts, mm, gamma = 0.5, verbose = TRUE)


names <- getFeatureNames(result)
n_instances <- numMCInstances(result)
n_samples <- nump

getDirichletSample(result, 1) |> dim()
