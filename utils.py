import random
import numpy as np
import torch
from scipy.stats import chi2_contingency


def get_cumulative_rewards(rewards, gamma=0.99):
    G = np.zeros_like(rewards, dtype=float)
    G[-1] = rewards[-1]
    for idx in range(-2, -len(rewards) - 1, -1):
        G[idx] = rewards[idx] + gamma * G[idx + 1]
    return G


def to_one_hot(y_tensor, ndims):
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    y_one_hot = torch.zeros(y_tensor.size()[0], ndims).scatter_(1, y_tensor, 1)
    return y_one_hot


def to_one_hot_np(x, ndim=3):
    out = np.zeros((len(x), ndim))
    out[np.arange(len(x)), x] = 1
    return out


# CONVERTS TRAJ LIST TO STEP LIST
def preprocess_traj(traj_list, step_list, n_actions, is_Demo=False):
    """covert three arrays (x, 5), (x, 1), (x, 1) into one of shape (x,7)"""
    step_list = step_list.tolist()
    for traj in traj_list:
        states = np.array(traj[0])
        if is_Demo:
            probs = np.ones((states.shape[0], 1))
        else:
            probs = np.array(traj[3]).reshape(-1, 1)
        if isinstance(traj[1], list) or len(traj[1].shape) < 2:
            # sampled trajectories: are not one hot
            actions = to_one_hot_np(traj[1], n_actions)
        else:
            # expert trajectories: one hot
            actions = traj[1]
        x = np.concatenate((states, probs, actions), axis=1)
        step_list.extend(x)
    return np.array(step_list)


def chi_square(act_real, act_sim, n_actions):
    assert len(act_real) > 1000
    # make counts (real)
    counts_real = np.zeros(n_actions)
    uni_real, counts = np.unique(act_real, return_counts=True)
    counts_real[uni_real.astype(int)] = counts
    # make counts (sim)
    counts_sim = np.zeros(n_actions)
    uni_sim, counts = np.unique(act_sim, return_counts=True)
    counts_sim[uni_sim.astype(int)] = counts

    # test
    contingency_table = np.array([counts_real, counts_sim])
    contingency_table = contingency_table[:, np.any(contingency_table, axis=0)]
    stat, p, _, _ = chi2_contingency(contingency_table)
    return round(stat)
