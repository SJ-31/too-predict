#!/usr/bin/env ipython

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import sklearn.datasets as datasets
from snakemake.script import snakemake as smk

# smk is object of class "Snakemake"


def get_iris(csv_out, meta_out):
    iris: dict = datasets.load_iris()
    df = pd.DataFrame(iris["data"])
    df.loc[:, "label"] = iris["target"]
    df.to_csv(csv_out, index=False)
    with open(meta_out, "w") as f:
        f.write(iris["DESCR"])


def process_cars(product: str, upstream, fn: str):
    df = pd.read_csv(upstream["download_cars"]["cars"])
    if fn == "median":
        summarized = df.median(axis=0)
    summarized.to_csv(product)


def param_dict(output: Sequence, spec: dict):
    print(output)
    print(type(output))  # "output" is class OutputFiles
    for o, (k, v) in zip(output, spec.items()):
        Path(str(o)).write_text(f"{k}: {v}")


if smk.rule == "get_iris":
    get_iris(csv_out=smk.output["csv"], meta_out=smk.output["meta"])
elif smk.rule == "show_dict":
    param_dict(smk.output, smk.config["specs"]["first"])
