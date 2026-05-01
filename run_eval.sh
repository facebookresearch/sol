#!/bin/bash

### example eval script, adapt as needed



### Mujoco

#ENV=mujoco_ant_gmaze_dense2
#TRAIN_DIR=/checkpoint/mikaelhenaff/sf2-exp/exp/50186934_2
#python -m sf_examples.mujoco.enjoy_mujoco --experiment exp --train_dir $TRAIN_DIR --env $ENV


### Doom

#ENV=doom_battle2
#TRAIN_DIR=/checkpoint/mikaelhenaff/sf2-exp/exp/49400573_1
#python -m sf_examples.vizdoom.enjoy_vizdoom --experiment exp --train_dir $TRAIN_DIR --env $ENV


### NetHack

# monk without inv
#ENV=nethack_challenge
#TRAIN_DIR=/checkpoint/mikaelhenaff/sf2-exp/exp/50291571_1

# monk with inv
#ENV=nethack_challenge
#TRAIN_DIR=/checkpoint/mikaelhenaff/sf2-exp/exp/50291580_4

# ranger with inv (highest seed)
#ENV=nethack_challenge
#TRAIN_DIR=/checkpoint/mikaelhenaff/sf2-exp/exp/50291581_2

# ranger with inv (regular seed)
#ENV=nethack_challenge
#TRAIN_DIR=/checkpoint/mikaelhenaff/sf2-exp/exp/50291581_4

# wizard with inv
#ENV=nethack_challenge
#TRAIN_DIR=/checkpoint/mikaelhenaff/sf2-exp/exp/50291582_5

ENV=nethack_challenge
#TRAIN_DIR=/checkpoint/mikaelhenaff/sf2-exp/exp/51799790_5
TRAIN_DIR=/checkpoint/mikaelhenaff/sf2-exp/exp/53511612_1

python -m sf_examples.nethack.enjoy_nethack --experiment exp --train_dir $TRAIN_DIR --env $ENV --fps 5

