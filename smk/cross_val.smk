import yaml


include: "Snakefile"


cv_outdir = f"{OUT}/shallow/cross_validation/{config.get('date', TODAY)}-{RUN}"
holdout_outdir = f"{OUT}/shallow/holdout/{config.get('date', TODAY)}-{RUN}"

model_dct = config["models"]["shallow"]

print("----CROSS VALIDATION----")
if only := config.get("run_only", []):
    if isinstance(only, str):
        only = only.split(",")
    models = list(set(model_dct.keys()) & set(only))
    print(f"Running with {models}")
    print("------------------------")
else:
    models = [k for k in model_dct.keys() if not model_dct[k].get("skip")]
    print("Running with models defined in yaml")
    print("Use the run_only key to run specific models")
print("------------------------")


outputs = {}
if config["do_cv"]:
    outputs["cv"] = f"{cv_outdir}/all_results.csv"
if config["do_holdout"]:
    outputs["holdout"] = f"{holdout_outdir}/all_results.csv"


rule all:
    input:
        **outputs,


rule cross_validate:
    params:
        outdir=cv_outdir,
        models=models,
    output:
        dirs=[directory(f"{cv_outdir}/{m}") for m in models],
    script:
        "scripts/shallow_main.py"


rule holdout:
    params:
        outdir=holdout_outdir,
        models=models,
    output:
        dirs=[directory(f"{holdout_outdir}/{m}") for m in models],
    script:
        "scripts/shallow_main.py"


rule evaluate_cv:
    input:
        rules.cross_validate.output.dirs,
    output:
        out=rules.all.input.cv,
    script:
        "scripts/cross_val_combine.R"


rule evaluate_holdout:
    input:
        rules.holdout.output.dirs,
    output:
        out=rules.all.input.holdout,
    script:
        "TODO"


onsuccess:
    with open(f"{cv_outdir}/snakemake_config.yaml", "w") as f:
        yaml.safe_dump(config, f)
