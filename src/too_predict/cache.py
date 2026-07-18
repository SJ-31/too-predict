#!/usr/bin/env python3

import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from attrs import define, field


@define
class NamedCache:
    dir: Path = field(converter=lambda x: Path(x))
    writer: Callable
    reader: Callable = pd.read_csv
    suffix: str = ".csv"

    def __attrs_post_init__(self):
        if not self.dir.exists():
            self.dir.mkdir(parents=True)

    @staticmethod
    def pickle(file, obj, load: bool = False):
        if load:
            with open(file, "rb") as f:
                obj = pickle.load(f)
            return obj
        with open(file, "wb") as f:
            if obj is None:
                raise ValueError(f"obj being saved to {file} is None")
            pickle.dump(obj, f)

    def __call__(
        self,
        fn,
        name: str,
        suffix: str | None = None,
        reader: Callable | None = None,
        writer: Callable | None = None,
        pkl: bool = False,
        **kws,
    ):
        suffix = suffix or self.suffix
        if pkl:
            writer = self.pickle
            reader = lambda x: self.pickle(x, obj=None, load=True)
            suffix = ".pkl"
        else:
            writer = self.writer if writer is None else writer
            reader = self.reader if reader is None else reader
        file = self.dir / f"{name}{suffix}"
        if file.exists():
            return reader(file)
        obj = fn(**kws)
        writer(file, obj)
        assert file.exists(), f"Writer function did not write to {file}"
        return obj


def with_repeat_caching(
    cache_dir: Path | str,
    n: int,
    writer: Callable[[int, Path], None],
    reader: Callable,
    combine: Callable[[list[Any]], None],
    suffix: str = ".csv",
) -> Any:
    """Helper function to cache the results of a expensive function
    that involves aribtrary repeats e.g. a bootstrap procedure

    Parameters
    ----------
    cache_dir : Path
        Path to save results of each round
    writer : Callable
        Function taking iteration round and filename as input. Must
        write to the given file
    reader : Callable
        Function used to read cached results
    combine : Callable
        Function to combine cached results after reading them with `reader`.
        e.g. pl.concat
    """
    cache_dir = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)
    n_previous = len(list(cache_dir.glob(f"*{suffix}")))
    n_remaining: int = n - n_previous
    for i in range(n_remaining):
        i = i + n_previous
        to_cache = cache_dir / f"{i}{suffix}"
        writer(i, to_cache)
        assert to_cache.exists(), "Writer function did not write file"
    result = combine([reader(f) for f in cache_dir.glob(f"*{suffix}")])
    return result
