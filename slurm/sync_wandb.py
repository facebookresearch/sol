import os
import json
import glob
import subprocess
import time
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--job_num', type=int, default=-1)
parser.add_argument('--max_seeds', type=int, default=5)
parser.add_argument('--wandb_proj', type=str, default="")
parser.add_argument('--results_dir', type=str, default="/checkpoint/mikaelhenaff/sf2-exp/exp")

args = parser.parse_args()

if args.job_num != -1:
    jobs_to_sync = [f'{args.job_num}_{i}' for i in range(1, args.max_seeds)]
elif args.wandb_proj != '':
    all_job_ids = [f.split('/')[-1] for f in glob.glob(f"{args.results_dir}/*")]
    jobs_to_sync = []
    print(f"Finding all job IDs for wandb project: {args.wandb_proj}...")

    for job_id in tqdm(all_job_ids):
        job_dir = f"{args.results_dir}/{job_id}"
        cfg_path = f"{job_dir}/exp/config.json"
        if os.path.exists(cfg_path):
            cfg = json.load(open(cfg_path))
            wandb_proj = cfg["wandb_project"]
            if wandb_proj == args.wandb_proj:
                jobs_to_sync.append(job_id)
else:
    raise ValueError("Must have args.job_num != -1 or wandb_proj != ''")

print(f"Will sync {len(jobs_to_sync)} jobs")
        

def get_running_slurm_jobs(user):
    output = subprocess.check_output(["squeue", "-u", user, "--format=%i"])
    job_ids = [line.strip() for line in output.decode().splitlines()]
    return job_ids[1:]


for i, job_id in enumerate(jobs_to_sync):
    job_dir = f"{args.results_dir}/{job_id}"
    cfg = json.load(open(f"{job_dir}/exp/config.json"))
    wandb_dir = cfg["wandb_dir"]
    wandb_id = cfg["wandb_unique_id"]
    run_dirs = glob.glob(f"{wandb_dir}/wandb/*{wandb_id}*")
    for wandb_run in run_dirs:
        wandb_cmd = f"wandb sync {wandb_run}"
        print(f"job {i} / {len(jobs_to_sync)} | executing: {wandb_cmd}")
        t_start = time.time()
        os.system(wandb_cmd)
        t_end = time.time()
        print(f"syncing took {t_end - t_start}s")
        


