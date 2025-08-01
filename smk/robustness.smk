include: "Snakefile"


script_path = f"{config['scripts']}/robustness.py"
model_dict = config["models"]["dl"]
models = [k for k in model_dict.keys() if not model_dict[k].get("skip")]
model_checkpoints = {m: f"{REPOS}/{DATE}/effective_robustness/{m}.ckpt" for m in models}

outdir = f"{OUT}/{DATE}_effective_robustness"


rule all:
    input:
        beta=f"{outdir}/beta.pkl",
        ckpts=model_checkpoints.values(),
        evaluation=f"{outdir}/effective_robustness.csv",


rule prep:
    output:
        expand(
            "{out}/{adatas}",
            out=f"{REPOS}/{DATE}/effective_robustness",
            adatas=["train.h5ad", "shifted_test.h5ad", "standard_test.h5ad"],
        ),
    script:
        script_path


rule fit_deep:
    input:
        train=rules.prep.output[0],
    output:
        **model_checkpoints,
    script:
        script_path


rule get_beta:
    # Obtain the linear regression model for computing effective robustness
    input:
        rules.prep.output,
    output:
        rules.all.input.beta,
    script:
        script_path


rule evaluate:
    input:
        beta_path=rules.all.input.beta,
        train=rules.prep.output[0],
        shifted_test=rules.prep.output[1],
        standard_test=rules.prep.output[2],
        ckpts=model_checkpoints.values(),
    output:
        rules.all.input.evaluation,
    params:
        **model_checkpoints,
    script:
        script_path
