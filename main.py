import os
import random
import pickle
import argparse
from sys import dont_write_bytecode
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

from experts.PG import PG
from cost import CostNN
from utils import to_one_hot, get_cumulative_rewards
from dummy_env import DummyEnv, get_optimal_action
from mode_env import ModeEnv

from torch.optim.lr_scheduler import StepLR

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--system", default="mode", type=str, help="environ")
parser.add_argument("-e", "--epochs", default=2000, type=int, help="nr epochs")
parser.add_argument("-p", "--plot", action="store_true", help="plot training?")
parser.add_argument(
    "-m", "--model", type=str, default="test", help="save name"
)
args = parser.parse_args()
ENV_NAME = args.system

# SEEDS
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def to_one_hot_np(x, ndim=3):
    out = np.zeros((len(x), ndim))
    out[np.arange(len(x)), x] = 1
    return out


# CONVERTS TRAJ LIST TO STEP LIST
def preprocess_traj(traj_list, step_list, is_Demo=False):
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
            actions = to_one_hot_np(traj[1], N_ACTIONS)
        else:
            # expert trajectories: one hot
            actions = traj[1]
        x = np.concatenate((states, probs, actions), axis=1)
        step_list.extend(x)
    return np.array(step_list)


# ENV SETUP
if ENV_NAME == "cart":
    env_name = 'CartPole-v0'
    env = gym.make(env_name).unwrapped
    if seed is not None:
        env.seed(seed)
    # LOADING EXPERT/DEMO SAMPLES
    demo_trajs = np.load('expert_samples/pg_cartpole.npy', allow_pickle=True)
    ONEHOT = True
    eval_env = env
elif ENV_NAME == "dummy":
    env = DummyEnv()
    with open("expert_samples/pg_modeEnv.pkl", "rb") as outfile:
        demo_trajs = pickle.load(outfile)
    ONEHOT = True
    EPISODES_TO_PLAY = 20
    REWARD_FUNCTION_UPDATE = 10
    DEMO_BATCH = 100
    eval_env = env
elif ENV_NAME == "mode":
    env = ModeEnv(os.path.join("expert_samples", "mobis_train.pkl"))
    with open("expert_samples/mobis_train.pkl", "rb") as outfile:
        # loading trajectories, mean and std but mean and std are not used here
        demo_trajs, _, _ = pickle.load(outfile)
    ONEHOT = False
    EPISODES_TO_PLAY = 1000
    REWARD_FUNCTION_UPDATE = 10
    DEMO_BATCH = 100
    eval_env = ModeEnv(os.path.join("expert_samples", "mobis_test.pkl"))
else:
    raise NotImplementedError("wrong environment")
N_ACTIONS = env.nr_act  # action_space.n
state_shape = env.nr_feats  # env.observation_space.shape
state = env.reset()

print(len(demo_trajs))

# INITILIZING POLICY AND REWARD FUNCTION
policy = PG(state_shape, N_ACTIONS)
cost_f = CostNN(state_shape + N_ACTIONS)
# changed both from lr e-2 to lower lr
policy_optimizer = torch.optim.Adam(policy.parameters(), 1e-3)
cost_optimizer = torch.optim.Adam(cost_f.parameters(), 1e-3, weight_decay=1e-4)

mean_rewards = []
mean_costs = []
mean_loss_rew = []

D_demo, D_samp = np.array([]), np.array([])

os.makedirs("trained_models", exist_ok=True)
os.makedirs(os.path.join("trained_models", args.model), exist_ok=True)

D_demo = preprocess_traj(demo_trajs, D_demo, is_Demo=True)
return_list, sum_of_cost_list = [], []
for i in range(args.epochs):
    # t_max=50 in generate session in order to restrict steps per episode -
    # but won't help here because of random factor
    trajs = [policy.generate_session(env) for _ in range(EPISODES_TO_PLAY)]
    D_samp = preprocess_traj(trajs, D_samp)

    # test if the cost function works:
    # if i==1 or i==100:
    #     for traj in trajs[:10]:
    #         states = torch.tensor(traj[0], dtype=torch.float32)
    #         actions = torch.tensor(to_one_hot_np(traj[1], N_ACTIONS)
    #               , dtype=torch.float32)
    #         print(states.size(), actions.size())
    #         with torch.no_grad():
    #             costs = cost_f(torch.cat((states, actions), dim=-1))
    #         for k in range(5):
    #             print(states[k], actions[k], traj[2][k], costs[k])

    # UPDATING REWARD FUNCTION (TAKES IN D_samp, D_demo)
    loss_rew = []
    for _ in range(REWARD_FUNCTION_UPDATE):
        selected_samp = np.random.choice(len(D_samp), DEMO_BATCH)
        selected_demo = np.random.choice(len(D_demo), DEMO_BATCH)

        D_s_samp = D_samp[selected_samp]
        D_s_demo = D_demo[selected_demo]

        #D̂ samp ← D̂ demo ∪ D̂ samp
        D_s_samp = np.concatenate((D_s_demo, D_s_samp), axis=0)

        states, probs, actions = D_s_samp[:, :-N_ACTIONS -
                                          1], D_s_samp[:, -N_ACTIONS - 1
                                                       ], D_s_samp[:,
                                                                   -N_ACTIONS:]
        states_expert, actions_expert = D_s_demo[:, :-N_ACTIONS -
                                                 1], D_s_demo[:, -N_ACTIONS:]

        # Reducing from float64 to float32 for making computaton faster
        states = torch.tensor(states, dtype=torch.float32)
        probs = torch.tensor(probs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        states_expert = torch.tensor(states_expert, dtype=torch.float32)
        actions_expert = torch.tensor(actions_expert, dtype=torch.float32)

        costs_samp = cost_f(torch.cat((states, actions), dim=-1))
        costs_demo = cost_f(torch.cat((states_expert, actions_expert), dim=-1))

        # LOSS CALCULATION FOR IOC (COST FUNCTION)
        loss_IOC = torch.mean(costs_demo) + \
                torch.log(torch.mean(torch.exp(-costs_samp)/(probs+1e-7)))
        # UPDATING THE COST FUNCTION
        cost_optimizer.zero_grad()
        loss_IOC.backward()
        cost_optimizer.step()

        loss_rew.append(loss_IOC.detach())

    # train policy
    for traj in trajs:
        states, actions, rewards, _ = traj
        actions = to_one_hot_np(actions, N_ACTIONS)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)

        # get the estimated reward with the current cost function
        costs = cost_f(torch.cat((states, actions), dim=-1)).detach().numpy()
        cumulative_returns = np.array(
            get_cumulative_rewards(-costs, 0.1)
        )  # 0.99)) TODO
        cumulative_returns = torch.tensor(
            cumulative_returns, dtype=torch.float32
        ).squeeze()

        logits = policy(states)
        # get log probs for all possible actions
        probs = nn.functional.softmax(logits, -1)
        log_probs = nn.functional.log_softmax(logits, -1)

        # get log probs for the action that was actually chosen
        log_probs_for_actions = torch.sum(
            # TODO: probs worked better than log_probs!
            probs * actions,
            dim=1
        )

        entropy = -torch.mean(torch.sum(probs * log_probs), dim=-1)
        loss_per_sample = -1 * (
            log_probs_for_actions * cumulative_returns + entropy * 1e-2
        )
        loss = torch.mean(loss_per_sample) + 1

        # UPDATING THE POLICY NETWORK
        if i > 200:
            # first train the cost function
            policy_optimizer.zero_grad()
            loss.backward()
            policy_optimizer.step()

    returns = sum(rewards)
    sum_of_cost = np.sum(costs)
    return_list.append(returns)
    sum_of_cost_list.append(sum_of_cost)

    mean_rewards.append(np.mean(return_list))
    mean_costs.append(np.mean(sum_of_cost_list))
    mean_loss_rew.append(np.mean(loss_rew))

    # EVAL PERFORMANCE
    if i % 10 == 0:
        if i % 100 == 0:
            print()
            # for ksl in range(5):
            ksl = 0
            print(
                "action:", np.argmax(actions[ksl]), ", reward:", rewards[ksl],
                "cost:", costs[ksl], "log probs act:",
                log_probs_for_actions[ksl].item()
            )
            # model saving (each 100 epochs save properly)
            policy.save_model(
                os.path.join("trained_models", args.model, f"policy_{i}")
            )
            cost_f.save_model(
                os.path.join("trained_models", args.model, f"costs_{i}")
            )

        # evaluation on test data
        if env == eval_env:
            print(
                f"Iter {i}: mean reward:{round(np.mean(return_list), 2)}\
                 loss IOC: {round(loss_IOC.item(), 2)}\
                  loss policy: {round(loss.item(), 2)}"
            )
            continue
        rew, max_rew, entropy_list = 0, 0, []
        for eval_epoch in range(30):
            s = eval_env.reset()
            # 20 is much higher than the possible steps in mode env
            for k in range(20):
                action_probs = policy.predict_probs(np.array([s]))[0]
                log_probs = np.log(action_probs)
                entropy_list.append(-(np.sum(action_probs * log_probs)))
                a = np.random.choice(policy.n_actions, p=action_probs)
                s, r, done, info = eval_env.step(a)
                rew += r
                if done:
                    break
            max_rew += (k + 1)
        print(
            f"Iter {i}: Reward:{round(np.mean(return_list), 2)}\
    loss IOC: {round(loss_IOC.item(), 2)}\
    loss policy: {round(loss.item(), 2)}\
    Entropy: {round(np.mean(entropy_list), 2)}\
    Test reward: {rew} / {max_rew}, {round(rew/max_rew, 2)}"
        )

        # model saving
        policy.save_model(os.path.join("trained_models", args.model, "policy"))
        cost_f.save_model(os.path.join("trained_models", args.model, "costs"))

        if args.plot:
            plt.figure(figsize=[16, 12])
            plt.subplot(2, 2, 1)
            plt.title(f"Mean reward per {EPISODES_TO_PLAY} games")
            plt.plot(mean_rewards)
            plt.grid()

            plt.subplot(2, 2, 2)
            plt.title(f"Mean cost per {EPISODES_TO_PLAY} games")
            plt.plot(mean_costs)
            plt.grid()

            plt.subplot(2, 2, 3)
            plt.title(f"Mean loss per {REWARD_FUNCTION_UPDATE} batches")
            plt.plot(mean_loss_rew)
            plt.grid()

            # plt.show()
            plt.savefig(
                os.path.join(
                    "trained_models", args.model, 'GCL_learning_curve.png'
                )
            )
            plt.close()
