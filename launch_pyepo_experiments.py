import os
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path

SLURM_HEADER_COMMON = """\
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -t 00-13:25
#SBATCH --mem=64g
#SBATCH -o /cluster/tufts/hugheslab/kheuto01/slurmlog/out/log_%j.out
#SBATCH -e /cluster/tufts/hugheslab/kheuto01/slurmlog/err/log_%j.err
"""

SLURM_CONFIGS = {
    "MA": """\
##SBATCH --nodelist=cc1gpu[001,003,004,005]
#SBATCH --gres=gpu:1
#SBATCH -p hugheslab
""",
    "cook": """\
##SBATCH --nodelist=cc1gpu[001,003,004,005]
#SBATCH --gres=gpu:1
#SBATCH -p hugheslab
""",
    "asurv": """\
##SBATCH --nodelist=cc1gpu[001,003,004,005]
#SBATCH --gres=gpu:1
#SBATCH -p hugheslab
"""
}

def make_outdir(base_dir, dataset, method, step_size, extra=""):
    tag = f"{dataset}_{method}_step{step_size:.0e}"
    if extra:
        tag += f"_{extra}"
    outdir = base_dir / tag
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def build_job_script(dataset, method, step_size, pg_noise, outdir):
    K = 100 if dataset in ["MA", "cook"] else 50
    cmd = f"python pyepo_experiment.py --K {K} --step_size {step_size:.5f} --epochs 4000 --seed 12345 --outdir {outdir} --num_score_samples 100 --dataset {dataset} --device cuda --val_freq 1 --method_name {method}"
    if method == "pg":
        cmd += f" --pg_sigma {pg_noise:.5f}"

    slurm_script = "#!/bin/bash\n"
    slurm_script += SLURM_HEADER_COMMON
    slurm_script += SLURM_CONFIGS[dataset]
    slurm_script += "\n" + cmd + "\n"
    return slurm_script

def main(dryrun=False, smoke_test=False, exclude_smoke=False, start_index=0):
    datasets = ["MA", "cook", "asurv"]
    methods = ["spo+", "pg"]
    step_sizes = np.logspace(-1, -3, 10)
    pg_noises = np.logspace(-2, 0, 10)

    base_dir = Path("/cluster/tufts/hugheslab/kheuto01/pyepo_exps") / f"{datetime.now():%Y%m%d}_run"
    base_dir.mkdir(parents=True, exist_ok=True)

    job_scripts = []

    for dataset in datasets:
        for method in methods:
            for i, step_size in enumerate(step_sizes):
                if smoke_test and i > 0:
                    continue
                if exclude_smoke and i == 0:
                    continue

                if method == "pg":
                    for j, noise in enumerate(pg_noises):
                        if smoke_test and j > 0:
                            continue
                        if exclude_smoke and j == 0 and i == 0:
                            continue

                        outdir = make_outdir(base_dir, dataset, method, step_size, f"noise{noise:.0e}")
                        script = build_job_script(dataset, method, step_size, noise, outdir)
                        job_scripts.append(script)
                else:
                    outdir = make_outdir(base_dir, dataset, method, step_size)
                    script = build_job_script(dataset, method, step_size, None, outdir)
                    job_scripts.append(script)

    if dryrun:
        print(f"Dry run: {len(job_scripts)} jobs would be launched.")
    else:
        for i, script in enumerate(job_scripts[start_index:], start=start_index):
            job_file = base_dir / f"job_{i}.slurm"
            with open(job_file, "w") as f:
                f.write(script)
            os.system(f"sbatch {job_file}")
        print(f"{len(job_scripts) - start_index} jobs submitted (starting from index {start_index}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dryrun", action="store_true", help="Show number of jobs to be submitted without launching.")
    parser.add_argument("--smoke_test", action="store_true", help="Launch 1 job per method for testing.")
    parser.add_argument("--exclude_smoke", action="store_true", help="Launch all jobs except the smoke test ones.")
    parser.add_argument("--start_index", type=int, default=0, help="Index to start submitting jobs from.")
    args = parser.parse_args()

    main(dryrun=args.dryrun, smoke_test=args.smoke_test, exclude_smoke=args.exclude_smoke, start_index=args.start_index)
