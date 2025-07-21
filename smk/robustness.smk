include: "setup.smk"


rule all:
    input:
        beta=f"{OUT}/effective_robustness_beta.pkl",


rule prep:
    output:
        expand(
            "{out}/{adatas}",
            out=f"{REPOS}/effective_robustness",
            adatas=["train.h5ad", "shifted_test.h5ad", "standard_test.h5ad"],
        ),
    script:
        f"{config['scripts']}/robustness.py"


rule get_beta:
    # Obtain the linear regression model for computing effective robustness
    input:
        rules.eff_prep.output,
    output:
        rules.all.input.beta,
    script:
        f"{config['scripts']}/robustness.py"
