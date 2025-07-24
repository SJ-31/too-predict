include: "Snakefile"


outpath = f"{OUT}/deep/cross_validation/{config.get('date', TODAY)}"
model_dict = config["models"]["dl"]
models = [k for k in model_dict.keys() if not model_dict[k].get("skip")]
model_cv_results = expand("{out}/{model}/cv_results.csv", out=outpath, model=models)
model_logs = [
    directory(d) for d in expand("{out}/{model}/tensorboard", out=outpath, model=models)
]


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
        outdir="{out}/deep/cross_validation/{date}/".format(out=OUT, date=DATE),
        date=DATE,
    output:
        cv=model_cv_results,
        log=model_logs,
    script:
        f"{config['scripts']}/torch_cross_val.py"
