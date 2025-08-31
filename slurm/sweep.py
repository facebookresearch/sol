# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import numpy as np
import numpy.random as npr
import math
import time
import os
import sys
import argparse
import yaml
import pdb

import itertools
import subprocess
from datetime import datetime
from subprocess import Popen, DEVNULL
from pprint import pprint

MAX_JOBS = 120

DEFAULT_ARGS = {
    'train_for_env_steps': 50000000000,
    'keep_checkpoints': 5,
    'stats_avg': 1000,
    'num_policies': 1,
    'normalize_returns': False,
    'with_vtrace': True,
}

SYSTEM_ARGS_DEBUG = {
    'num_workers': 1,
    'num_envs_per_worker': 8,
    'batch_size': 1024 * 4,
    'serial_mode': True,
    'with_wandb': False, 
}

SYSTEM_ARGS_CLUSTER = {
    'num_workers': 48,
    'num_envs_per_worker': 40,
    'batch_size': 1024 * 32,
    'with_wandb': True,
    'experiment_summaries_interval': 100,
}




def enforce_args_constraints(args):
    if 'rollout' in args.keys():
        args['recurrence'] = args['rollout']
    return args

# creates commands and job file names for a grid search over lists of hyperparameters
class Overrides(object):
    def __init__(self, default_args=None, args_file=None):
        self.default_args = default_args
        self.args = []
        exp_args = yaml.safe_load(open(args_file, 'r'))
        for name, values in exp_args.items():
            self.add(name, values)

    def add(self, arg_name, arg_values):
        self.args.append([{arg_name: v} for v in arg_values])

    def parse(self, exp_name):
        args_list = []
        for combos in itertools.product(*self.args):
            args = {**self.default_args}
                
            for arg in combos:
                args.update(arg)

            args = enforce_args_constraints(args)
            
            args['experiment'] = exp_name
            args_list.append(args)
        return args_list



               
# good for checking if we have reached max non-array jobs, so we don't get locked out...
def num_jobs_running():
    cmd = f"squeue -u $USER -h -o \"%i\" | cut -d '_' -f 1 | sort -u | wc -l"
    return int(subprocess.check_output(cmd, shell=True))
        



def main():
    
    username = os.environ['USER']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--script', default='single_node_template.sbatch')
    parser.add_argument('--expfile', default='exp_configs/test.yaml')
    parser.add_argument('--exp_name', default='exp')
    parser.add_argument('--wandb_proj', default='test')
    parser.add_argument('--partition', default='')
    parser.add_argument('--num_gpus', default=1)
    parser.add_argument('--num_cpus', default=50)
    parser.add_argument('--days', default=3)
    parser.add_argument('--seeds', default=5)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args()

    # default arguments
    ARGS_DICT = {}
    ARGS_DICT.update(DEFAULT_ARGS)
    ARGS_DICT.update(SYSTEM_ARGS_DEBUG if args.debug else SYSTEM_ARGS_CLUSTER)
    ARGS_DICT.update({'wandb_project': args.wandb_proj})
    ARGS_DICT.update({'wandb_user': username})

    

    # read and parse additional arguments
    overrides = Overrides(default_args=ARGS_DICT, args_file=args.expfile)
    exp_name = args.exp_name
    list_cmd_args = overrides.parse(exp_name)

    env_name = list_cmd_args[0]['env'].lower()
    if 'nethack' in env_name or 'minihack' in env_name:
        BASE_CMD = 'python ../sf_examples/nethack/train_nethack.py'
    elif 'mujoco' in env_name:
        BASE_CMD = 'python ../sf_examples/mujoco/train_mujoco.py'
    else:
        raise ValueError(f'env_name {env_name} should include either: nethack, mujoco')
    
    # read SBATCH template and fill in fields
    script_template = open(args.script, 'r').read().strip()
    script_template = script_template.format(
        partition=args.partition,
        num_gpus=args.num_gpus,
        exp_name=args.exp_name,
        num_cpus=args.num_cpus,
        days=args.days,
        seeds=args.seeds,
        base_cmd=BASE_CMD,
    )

    # in dry mode, we test locally for debugging.
    if args.dry:
        tmp_dir = f'/tmp'
        cmd_args = ' '.join(f'--{k} {v}' for k, v in list_cmd_args[0].items())
        cmd = f'CUDA_VISIBLE_DEVICES=0 {BASE_CMD} {cmd_args}'
        cmd += f' --train_dir {tmp_dir} --with_wandb False'
        if args.debug:
            cmd += f' --batch_size 4096 --num_workers 1 --num_envs_per_worker 2'
        print(f'executing: {cmd}')
        os.system(f'rm -rf {tmp_dir}; {cmd}')
    else:
        print(f'launching {len(list_cmd_args)} jobs, {num_jobs_running()} jobs running')
        pprint(args.__dict__)
        import pdb; pdb.set_trace()
        for i in range(len(list_cmd_args)):   
            if not num_jobs_running() == MAX_JOBS:
                cmd_args = ' '.join(f'--{k} {v}' for k, v in list_cmd_args[i].items())
                script = script_template + ' ' + cmd_args                
                subprocess.run(["sbatch"], input=script.encode())
            

        

if __name__ == '__main__':
    main()
