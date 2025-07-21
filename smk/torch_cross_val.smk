include: "setup.smk"


model_cv_results = expand(
    "{out}/deep/cross_validation/{date}/{model}/cv_results.csv",
    out=OUT,
    date=config.get("date", TODAY),
    model=config["models"]["dl"].keys(),
)


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
        model_cv_results,
    script:
        f"{config['scripts']}/torch_cross_val.py"
