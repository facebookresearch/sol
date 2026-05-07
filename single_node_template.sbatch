#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# All rights reserved.

#SBATCH --output=/checkpoint/%u/sol/%A_%a/slurm.out
#SBATCH --job-name=single-node
#SBATCH --wait-all-nodes=1
#SBATCH --time={days}-00:00:00
#SBATCH --open-mode=append
#SBATCH --constraint=volta32gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={num_cpus}
#SBATCH --signal=B:USR1@60 # send USR1 60 seconds before the time limit
#SBATCH --array=1-{seeds} # number of seeds

trap_handler () {{
   echo "Caught signal: " $1
   # SIGTERM must be bypassed
   if [ "$1" = "TERM" ]; then
       echo "bypass sigterm"
   else
     echo "Requeuing " $SLURM_JOB_ID
     scontrol requeue $SLURM_JOB_ID
   fi
}}

trap 'trap_handler USR1' USR1
trap 'trap_handler TERM' TERM

export LOGDIR=/checkpoint/$USER/sol/${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}

mkdir -p $LOGDIR

SEED=$SLURM_ARRAY_TASK_ID

srun --output ${{LOGDIR}}/appo_%j.out \
     {base_cmd} --seed $SEED --train_dir $LOGDIR --with_wandb True
  
