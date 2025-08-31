#!/bin/bash

# example script how to restart a job locally (mostly useful for testing)

ORIG_JOB_ID=$1
SLURM_ARRAY_TASK_ID=$2
ORIG_EXP_NAME=exp
ENV=nethack_score_fixed_eat

SEED=$SLURM_ARRAY_TASK_ID

LOGDIR=/checkpoint/$USER/sf2-exp/exp/${ORIG_JOB_ID}_${SLURM_ARRAY_TASK_ID}


python ../sf_examples/nethack/train_nethack.py --env $ENV --seed $SEED --train_dir $LOGDIR --experiment $ORIG_EXP_NAME





