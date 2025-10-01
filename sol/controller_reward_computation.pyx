# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# All rights reserved.

# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

import numpy as np
cimport numpy as cnp
cimport cython

ctypedef cnp.float32_t DTYPE_FLOAT_t
ctypedef cnp.int64_t DTYPE_INT_t

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_controller_reward_sums_cython(
    cnp.ndarray[DTYPE_FLOAT_t, ndim=3] obs_rewards,
    cnp.ndarray[DTYPE_INT_t, ndim=2] policy_indx,
    cnp.ndarray[DTYPE_FLOAT_t, ndim=2] rewards_cpu,
    DTYPE_INT_t controller_indx
):
    cdef Py_ssize_t batch_size = policy_indx.shape[0]
    cdef Py_ssize_t time_steps = policy_indx.shape[1]
    cdef Py_ssize_t b, t
    cdef DTYPE_INT_t last_controller_indx
    cdef DTYPE_FLOAT_t controller_sum
    
    for b in range(batch_size):
        last_controller_indx = -1
        controller_sum = 0.0
        
        for t in range(time_steps):
            if policy_indx[b, t] == controller_indx or t == time_steps - 1:
                if last_controller_indx != -1:
                    rewards_cpu[b, last_controller_indx] = controller_sum
                controller_sum = 0.0
                last_controller_indx = t
            else:
                controller_sum += obs_rewards[b, t, controller_indx]

            if policy_indx[b, t] == controller_indx and t == time_steps - 1:
                rewards_cpu[b, t] = 0

    
    return rewards_cpu