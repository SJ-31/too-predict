library(here)
library(glue)
library(tidyverse)

from_yaml <- yaml::read_yaml(here("tests", "meta_paths.yaml"))

WANTED_COLS <- c("case_prefix", "case_number", "containing_folder", "tumor_type", "available", "has_raw")

process_one <- function(spec, tumor_type) {
  get_case_number <- function(names) {
    for (r in spec$to_remove) {
      names <- str_remove(names, r)
    }
    str_remove(names, "_[BC]$") |> str_remove("-[BC]$")
  }

  names <- spec$path |>
    str_split_1(" ") |>
    tools::file_path_sans_ext() |>
    str_remove("_1.fastq") |>
    str_remove("_2.fastq")
  tb <- tibble(filename = names) |>
    mutate(case_number = get_case_number(names)) |>
    group_by(case_number) |>
    nest() |>
    mutate(available = map_chr(data, \(x) {
      stopifnot("WARNING: more samples than expected" = nrow(data) <= 2)
      f <- x$filename
      has_b <- str_detect(f, "[_-]B") |> any()
      has_c <- str_detect(f, "[_-]C") |> any()
      if (has_b && has_c) {
        "BC"
      } else if (has_b) {
        "B"
      } else {
        "C"
      }
    }))
  tb |>
    select(-data) |>
    mutate(
      containing_folder = str_remove(spec$folder, "/ssh:shannc@161.200.107.77:"),
      tumor_type = tumor_type,
      has_raw = ifelse(spec$raw, "T", "F"),
      case_prefix = ifelse(is.null(spec$prefix), "", spec$prefix)
    ) |>
    select(all_of(WANTED_COLS)) |>
    arrange(case_number)
}

process_all <- function(spec_list, tumor_type, outfile) {
  joined <- lapply(spec_list, \(x) process_one(x, tumor_type)) |>
    bind_rows() |>
    group_by(case_number) |>
    mutate(n = n())
  joined |>
    ungroup() |>
    filter(n == 1) |>
    select(all_of(WANTED_COLS)) |>
    write_tsv(outfile)
  filter(joined, n > 1) |>
    mutate(
      available = map_chr(available, \(x) {
        paste0(x) |>
          str_split_1("") |>
          unique() |>
          paste0(collapse = "")
      }),
      has_raw = map_chr(has_raw, \(x) {
        if ("T" %in% x) {
          "T"
        } else {
          "F"
        }
      })
    ) |>
    summarise(across(everything(), first)) |>
    select(all_of(WANTED_COLS))
}

# CRC
## outdir <- here("tests")
## check <- process_all(from_yaml$Exome$CRC, tumor_type = "CRC", here(outdir, "crc_meta.tsv"))
## write_tsv(check, here(outdir, "check.tsv"))

## process_one(from_yaml$Exome$CRC[[1]])

# All rna
for (tt in names(from_yaml$RNASeq)) {
  check <- process_all(from_yaml$RNASeq[[tt]],
    tumor_type = tt,
    here(outdir, glue("{tt}_rna_meta.tsv"))
  )
  write_tsv(check, here(outdir, glue("{tt}_rna_check.tsv")))
}

## final |> write_tsv(here("tests", "cur_meta.tsv"))
