import numpy as np
import os
import pandas as pd
import argparse
import json
import sys

from experts.PG import PG
from cost import CostNN
from mode_env import ModeEnv, included_modes
from utils import chi_square
from sklearn.metrics import accuracy_score, balanced_accuracy_score

try:
    from v2g4carsharing.mode_choice_model.evaluate import mode_share_plot
except:
    print("v2g4carsharing module not available, pass import")
    pass


# evaluate performance
def eval_performance(eval_env, policy, iters=30, return_act=False):
    rew, max_rew, entropy_list = 0, 0, []
    act_sim, act_real = [], []
    for _ in range(iters):
        s = eval_env.reset()
        # 20 is much higher than the possible steps in mode env
        for k in range(20):
            # get real next action
            real_action = np.argmax(
                eval_env.current_traj[1][eval_env.episode_step]
            )
            # predict action
            action_probs = policy.predict_probs(np.array([s]))[0]
            log_probs = np.log(action_probs + 1e-7)
            a = np.random.choice(policy.n_actions, p=action_probs)
            # save entropy
            entropy_list.append(-(np.sum(action_probs * log_probs)))
            # env step
            s, r, done, _ = eval_env.step(a)
            # save stuff
            rew += r
            act_sim.append(a)
            act_real.append(real_action)
            if done:
                break
        max_rew += (k + 1)
    mean_entropy = sum(list(entropy_list)) / len(entropy_list)
    if return_act:
        return rew, max_rew, mean_entropy, np.array(act_sim
                                                    ), np.array(act_real)
    return rew, max_rew, mean_entropy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", default=300, type=int, help="to load")
    parser.add_argument(
        "-m", "--model", type=str, default="test", help="save name"
    )
    parser.add_argument("--use_prev_mode", action="store_true")
    parser.add_argument("--tofile", action="store_true")
    args = parser.parse_args()

    model = args.model
    eval_env = ModeEnv(data="test", use_prevmode=args.use_prev_mode)
    N_ACTIONS = eval_env.nr_act  # action_space.n
    state_shape = eval_env.nr_feats  # env.observation_space.shape

    if args.tofile:
        f = open(
            os.path.join(
                os.path.join("trained_models", model), "evaluation.txt"
            ), "w"
        )
        sys.stdout = f

    # INITILIZING POLICY AND REWARD FUNCTION
    policy = PG(state_shape, N_ACTIONS)
    policy.load_model(
        os.path.join("trained_models", model, f"policy_{args.epoch}")
    )
    # # cost network (not needed)
    # cost_f = CostNN(state_shape + N_ACTIONS)
    # cost_f.load_model(
    #     os.path.join("trained_models", model, f"costs_{args.epoch}")
    # )

    print("Available epochs: ", eval_env.nr_traj)
    print("Number of actions:", N_ACTIONS)
    (rew, max_rew, mean_entropy, act_sim, act_real) = eval_performance(
        eval_env, policy, iters=eval_env.nr_traj, return_act=True
    )

    print(f"Test reward: {rew} / {max_rew}, {round(rew/max_rew, 2)}")

    # make plot
    included_modes = np.array(included_modes)
    # chi square
    print("chi square stats:", chi_square(act_real, act_sim, N_ACTIONS))

    # mode share plot:
    act_real = included_modes[act_real]
    act_sim = included_modes[act_sim]
    print("Acc:", accuracy_score(act_real, act_sim))
    print("Balanced Acc:", balanced_accuracy_score(act_real, act_sim))
    mode_share_plot(
        act_real, act_sim, out_path=os.path.join("trained_models", model)
    )
    if args.tofile:
        f.close()
