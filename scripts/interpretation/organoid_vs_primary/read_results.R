library(tidyverse)
library(limma)
library(glue)
library(here)
library(igraph)
library(ComplexHeatmap)

outdir_o <- here("data", "output", "chula_organoid_comparison", "de_enrichment")
contrasts <- list(
  sample_type = c(1, -1, 1, -1, 1, -1, 1, -1),
  coad_read = "COAD_READ.organoid - COAD_READ.primary",
  lihc = "LIHC.organoid - LIHC.primary",
  paad = "PAAD.organoid - PAAD.primary",
  chol = "CHOL.organoid - CHOL.primary"
)

ensembl2symbol <- read_tsv(here("data", "mappings", "ensembl_113_id_mapping.tsv")) |>
  distinct(ensembl, .keep_all = TRUE)

## * Gene families

fhierarchy <- read_csv(here("data", "reference", "2025-5-16_hgnc_family_hierarchy.csv"))
families <- read_csv(here("data", "reference", "2025-5-16-hgnc_families.csv"))

hgnc_graph <- graph_from_edgelist(as.matrix(fhierarchy))
# Check that it's acyclic
print(is_acyclic(hgnc_graph))

root_families <- local({
  ids <- as.character(V(hgnc_graph)[degree(hgnc_graph, mode = "in") == 0])
  setNames(families$name, families$id)[ids] |> discard(is.na)
  # node is a root if indegree is 0
})

fmap <- read_tsv(here("data", "mappings", "2025-5-16-hgnc_families_map.tsv"))

with_root_fam <- fmap |>
  group_by(`Ensembl gene ID`) |>
  nest() |>
  mutate(family = lapply(data, \(x) {
    fam <- x$`Family name`
    if (length(fam) > 1) {
      first(keep(fam, \(f) f %in% root_families))
      # Unfortunately genes can belong to multiple root families
    } else {
      fam
    }
  })) |>
  select(-data) |>
  mutate(family = as.character(family))


top_tags <- sapply(names(contrasts), \(n) {
  read_tsv(here(outdir_o, glue("{n}_top_tags.tsv"))) |>
    inner_join(with_root_fam, by = join_by(x$GENEID == y$`Ensembl gene ID`)) |>
    arrange(logFC) |>
    filter(PValue <= 0.05 & !is.na(GENEID))
}, simplify = FALSE, USE.NAMES = TRUE)

batch_var <- read_tsv(here("data", "output", "explanations", "batch_correction", "var.csv"))


n <- 1000


# Upset plot of top 1000 most downreg DE genes
to_upset <- sapply(list_modify(top_tags, sample_type = zap()), \(x) {
  x |>
    filter(logFC < 0) |>
    pluck("GENENAME")
}, simplify = FALSE, USE.NAMES = TRUE)

mt <- make_comb_mat(to_upset)
png(here(outdir_o, "downreg_de.png"), width = 8, height = 8, units = "in", res = 1080)
UpSet(mt)
dev.off()

# Shared characteristics of uniquely downreg genes
unique_downreg <- local({
  to_get <- list(paad = "0010", coad_read = "1000", lihc = "0100", chol = "0001")
  lst <- sapply(to_get, \(x) extract_comb(mt, x), simplify = FALSE, USE.NAMES = TRUE)
  tibble(tumor_type = names(lst), GENENAME = lst) |>
    unnest(GENENAME) |>
    left_join(select(ensembl2symbol, ensembl, symbol), by = join_by(x$GENENAME == y$symbol)) |>
    left_join(with_root_fam, by = join_by(x$ensembl == y$`Ensembl gene ID`))
})
ud_families <- table(unique_downreg$tumor_type, unique_downreg$family)


top_tags_all <- list_modify(top_tags, sample_type = zap()) |>
  lmap(\(x) mutate(x[[1]], tumor_type = names(x[1]))) |>
  bind_rows()
fcounts <- table(top_tags_all$tumor_type, top_tags_all$family)
