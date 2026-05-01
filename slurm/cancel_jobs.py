import os
import json
import glob

# Script to kill all slurm jobs matching the wandb hyperparameters
# listed in CANCEL_CONDITIONS. This is fairly specific to the
# current codebase but does the job.


JOB_DIR = f"/checkpoint/{os.getlogin()}/sf2-exp-craftium/exp"
#JOB_DIR = f"/checkpoint/{os.getlogin()}/sol"

CANCEL_CONDITIONS = {
    "wandb_project": "mikael_sol_craftium_04_08",
#    "num_epochs": 4,
}


def cancel(cfg):
    kill = False
    if all(k in cfg.keys() for k in CANCEL_CONDITIONS.keys()):
        if cfg["wandb_project"] == CANCEL_CONDITIONS["wandb_project"]:
            if all(cfg[k] == v for k, v in CANCEL_CONDITIONS.items()):
                kill = True
    return kill


for folder in glob.glob(f"{JOB_DIR}/*/*/config.json"):
    cfg = json.load(open(folder, "r"))
    if cancel(cfg):
        job_id = str(folder.replace(JOB_DIR, "").split("/")[1])
        print(f"canceling job {job_id}")
        os.system(f"scancel {job_id}")

