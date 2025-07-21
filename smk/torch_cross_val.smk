include: "Snakefile"


outpath = f"{OUT}/deep/cross_validation/{config.get('date', TODAY)}"
models = config["models"]["dl"].keys()
model_cv_results = expand("{out}/{model}/cv_results.csv", out=outpath, model=models)
n_folds = range(config["defaults"]["dl"]["cv"]["n_splits"])
fold_output = expand("{out}/{model}/fold_{n}", out=outpath, model=models, n=n_folds)


rule all:
    input:
        model_cv_results,


rule preprocess:
    output:
        expand(
            "{storage}/adatas/torch_main/{models}.h5ad",
            storage=REPOS,
            models=config["models"]["dl"].keys(),
        ),
    script:
        f"{config['scripts']}/torch_cross_val.py"


rule cross_validate:
    input:
        rules.preprocess.output,
    params:
        outdir="{out}/deep/cross_validation/{date}/".format(
            out=OUT, date=config.get("date", TODAY)
        ),
    output:
        cv=model_cv_results,
        fold_dir=directory(fold_output),
    script:
        f"{config['scripts']}/torch_cross_val.py"
