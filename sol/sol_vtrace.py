# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# All rights reserved.

# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

"""
This function computes advantages and value targets for all policies in the batch simultaneously. The arguments are:
    
ratios: ratio of action probs between current and old policy
values: bootstrapped value predictions
dones: episode terminals
rewards: rewards of mixed type, see Policy Reward in Table 7.
rho_hat: V-trace truncation parameter
c_hat: V-trace truncation parameter
num_trajectories: number of trajectories in the batch
recurrence: number of timesteps in the batch 
gamma: discounting factor 
policy_indx: the Policy Index in Table 7, also z_t in Section 3.1
num_policies: total number of policies (options and controller, i.e. |\Omega| + 1). 
"""

def compute_vtrace_sol(
        cfg,
        ratios_cpu,
        values_cpu,
        dones_cpu,
        rewards_cpu,
        rho_hat,
        c_hat,
        num_trajectories,
        recurrence,
        gamma,
        policy_indx,
        num_policies,
        dtype=torch.float32,
        final_mask_value=-42.42
):
    vtrace_rho = torch.min(rho_hat, ratios_cpu)
    vtrace_c = torch.min(c_hat, ratios_cpu)

    vs = torch.full((num_trajectories * recurrence,), final_mask_value, dtype = dtype)
    adv = torch.full((num_trajectories * recurrence,), final_mask_value, dtype = dtype)
        
    next_values = torch.zeros(num_trajectories, num_policies, dtype = dtype)
    next_vs = torch.zeros(num_trajectories, num_policies, dtype = dtype)
    delta_s = torch.zeros(num_trajectories, num_policies, dtype = dtype)
    
    is_base_case_handled = torch.zeros(num_trajectories, num_policies, dtype=torch.bool)
    is_episode_done = torch.zeros(num_trajectories, num_policies, dtype=torch.bool)

    last_policies = torch.full((num_trajectories,), -1)
        
    for i in reversed(range(cfg.recurrence)):
        current_policies = policy_indx[i::recurrence]
        current_policies_one_hot = F.one_hot(current_policies, num_classes = num_policies).bool()
        
        rewards = rewards_cpu[i::recurrence]
        curr_dones = dones_cpu[i::recurrence].bool()

        # when we encounter a "done", mark all policies as done.
        # we will unmark the ones at the current timestep for which we mask out the returns. 
        is_episode_done = is_episode_done | curr_dones.view(-1, 1)
            
        dones = is_episode_done.view(-1)[current_policies_one_hot.view(-1)].to(dtype)
        not_done = 1.0 - dones
        not_done_times_gamma = not_done * gamma

        curr_values = values_cpu[i::recurrence]
        curr_vtrace_rho = vtrace_rho[i::recurrence]
        curr_vtrace_c = vtrace_c[i::recurrence]

        # we have accounted for the latest episode termination of the current policies in 'not_done_times_gamma',
        # so reset this until the next 'done' is encountered.
        is_episode_done.view(-1)[current_policies_one_hot.view(-1)] = False

        if cfg.sol_bootstrap_on_option_switch:
            if i < cfg.recurrence - 3:
                controller_indx = num_policies - 1
                trajs_with_changed_options = (
                    (policy_indx[(i+1)::recurrence] == controller_indx) &
                    (policy_indx[i::recurrence] != policy_indx[(i+2)::recurrence])
                )
                # for any trajectories where the option switched, reset the base case so that bootstrapped returns are applies
                is_base_case_handled[current_policies_one_hot] = is_base_case_handled[current_policies_one_hot] & ~trajs_with_changed_options

            
        base_case_indices = (~is_base_case_handled) & current_policies_one_hot
        base_case_indices_any = torch.any(base_case_indices, dim = 1)

        next_values.view(-1)[base_case_indices.view(-1)] = (
            values_cpu[i :: recurrence][base_case_indices_any]
            - rewards_cpu[i :: recurrence][base_case_indices_any]
        ) / gamma

        next_vs.view(-1)[base_case_indices.view(-1)] = next_values.view(-1)[base_case_indices.view(-1)]
            
        is_base_case_handled = is_base_case_handled | base_case_indices

        if not is_base_case_handled.any().item():
            continue

        delta_s.view(-1)[current_policies_one_hot.view(-1)] = curr_vtrace_rho * (
            rewards
            + not_done_times_gamma * next_values.view(-1)[current_policies_one_hot.view(-1)]
            - curr_values
        )

        adv[i::recurrence] = curr_vtrace_rho * (
            rewards
            + not_done_times_gamma * next_vs.view(-1)[current_policies_one_hot.view(-1)]
            - curr_values
        )

        next_vs.view(-1)[current_policies_one_hot.view(-1)] = (
            curr_values
            + delta_s.view(-1)[current_policies_one_hot.view(-1)]
            + not_done_times_gamma
            * curr_vtrace_c
            * (next_vs.view(-1)[current_policies_one_hot.view(-1)] - next_values.view(-1)[current_policies_one_hot.view(-1)])
        )

        vs[i::recurrence] = next_vs.view(-1)[current_policies_one_hot.view(-1)]
            
        next_values.view(-1)[current_policies_one_hot.view(-1)] = curr_values
            

    return adv, vs
