#!/bin/bash

#SBATCH --output=/checkpoint/%u/sf2-exp/exp/%j/slurm.out
#SBATCH --job-name=restart
#SBATCH --time=10-00:00:00
#SBATCH --wait-all-nodes=1
#SBATCH --open-mode=append
#SBATCH --constraint=volta32gb

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=cortex
#SBATCH --cpus-per-task=50

trap_handler () {
   echo "Caught signal: " $1
   # SIGTERM must be bypassed
   if [ "$1" = "TERM" ]; then
       echo "bypass sigterm"
   else
     echo "Requeuing " $SLURM_JOB_ID
     scontrol requeue $SLURM_JOB_ID
   fi
}

trap 'trap_handler USR1' USR1
trap 'trap_handler TERM' TERM


# edit these according to the job you want to restart
ORIG_JOB_ID=$1
ORIG_EXP_NAME=exp
ORIG_ARRAY_TASK_ID=$2
BASE_CMD=../sf_examples/nethack/train_nethack.py
ENV=nethack_score_fixed_eat

LOGDIR=/checkpoint/$USER/sf2-exp/exp/${ORIG_JOB_ID}_${ORIG_ARRAY_TASK_ID}

echo "LOGDIR=${LOGDIR}"


SEED=$ORIG_ARRAY_TASK_ID

srun --output ${LOGDIR}/appo_%j.out \
     python ${BASE_CMD} --env $ENV --seed $SEED --train_dir $LOGDIR --experiment $ORIG_EXP_NAME
  
