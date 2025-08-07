import yaml


include: "Snakefile"


outpath = f"{OUT}/deep/cross_validation/{config.get('date', TODAY)}{RUN}"


model_dict = config["models"]["dl"]
models = [k for k in model_dict.keys() if not model_dict[k].get("skip")]
model_cv_results = expand("{out}/{model}/cv_results.csv", out=outpath, model=models)
model_logs = [
    directory(d) for d in expand("{out}/{model}/tensorboard", out=outpath, model=models)
]
models = models + ["baseline"]
baseline_cv = f"{outpath}/baseline_cv.csv"


rule all:
    input:
        cv=model_cv_results,
        baseline_cv=baseline_cv,


rule preprocess:
    output:
        main=expand(
            "{storage}/adatas/torch_cv_{date}/{models}.h5ad",
            storage=REPOS,
            date=DATE,
            models=models,
        ),
        baseline=f"{REPOS}/adatas/torch_cv_{DATE}/baseline.h5ad",
    script:
        f"{config['scripts']}/torch_cross_val.py"


rule baseline:
    input:
        rules.preprocess.output.baseline,
    output:
        rules.all.input.baseline_cv,
    script:
        f"{config['scripts']}/torch_cross_val.py"


rule cross_validate:
    input:
        rules.preprocess.output.main,
    params:
        outdir=outpath,
        date=DATE,
    output:
        cv=model_cv_results,
        log=model_logs,
    script:
        f"{config['scripts']}/torch_cross_val.py"


with open(f"{outpath}/config.yaml", "w") as f:
    yaml.safe_dump(config, f)
