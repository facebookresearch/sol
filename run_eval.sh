#!/bin/bash

### example eval script, adapt as needed

ENV=nethack_challenge
TRAIN_DIR=/checkpoint/mikaelhenaff/sf2-exp/exp/53511612_1

python -m sf_examples.nethack.enjoy_nethack --experiment exp --train_dir $TRAIN_DIR --env $ENV --fps 5

