#!/usr/bin/env python
#
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import ast
import contextlib
import os
import random
import termios
import time
import timeit
import tty

import gymnasium as gym
import numpy as np
#import nle  # noqa: F401
#from nle import nethack

import nle_patched.nle as nle
import nle_patched.nle.nethack as nethack

import minihack

from sf_examples.nethack.nethack_env import make_nethack_env
from sf_examples.nethack.train_nethack import (
    register_nethack_envs, 
    register_nethack_components,
    parse_nethack_args,
)


@contextlib.contextmanager
def dummy_context():
    yield None

    
@contextlib.contextmanager
def no_echo():
    tt = termios.tcgetattr(0)
    try:
        tty.setraw(0)
        yield
    finally:
        termios.tcsetattr(0, termios.TCSAFLUSH, tt)

        
def go_back(num_lines):
    print("\033[%dA" % num_lines)

    
def get_action(env):
    while True:
        with no_echo():
            ch = ord(os.read(0, 1))
        if ch in [nethack.C("c")]:
            print("Received exit code {}. Aborting.".format(ch))
            return None
        try:
            action = env.unwrapped.actions.index(ch)
        except ValueError:
            print(
                ("Selected action '%s' is not in action list. Please try again.")
                % chr(ch)
            )
            go_back(2)
            continue
        return action


def play(cfg):

    steps = 0
    total_reward = 0
    action = None
    

    env = make_nethack_env(cfg.env, cfg, render_mode='human')
    
    obs, info = env.reset()

    while True:
        env.render()
        action = get_action(env)

        if action is None:
            break
        
        obs, reward, done, trunc, info = env.step(action)
        steps += 1
        total_reward += reward
        print(f"Step {steps}, reward {reward}")

        if done or trunc:
            break


    print("Final reward:", total_reward)
    print("End status:", info["end_status"].name)
    env.close()


def main():
    
    register_nethack_components()
    cfg = parse_nethack_args()
    play(cfg)


if __name__ == "__main__":
    main()
