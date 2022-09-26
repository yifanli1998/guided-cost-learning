import pickle
import numpy as np


def get_optimal_action(s):
    if s[1] < 0.3:
        return 0
    elif s[0] < 0.5:
        # car sharing: mornign with high distance
        return 1
    else:
        # car sharing: afternoon with high distance
        return 2
    return action


class ModeEnv:

    def __init__(self):
        self.nr_feats = 2
        self.nr_act = 3

    def actind_to_action(self, actind):
        act = np.zeros(self.nr_act)
        act[actind] = 1
        return act

    def reset(self):
        self.state = np.random.rand(self.nr_feats)
        self.wrong_counter = 0
        # self.state[:self.nr_feats] = np.random.rand(self.nr_feats)
        return self.state

    def step(self, actind):
        opt_act = get_optimal_action(self.state)
        rew = 1 if opt_act == actind else 0  # -1 * np.linalg.norm(action - opt_act)
        if rew == 0:
            self.wrong_counter += 1
        action = self.actind_to_action(actind)
        self.state = np.random.rand(self.nr_feats)
        # basically a supervised task right now
        # self.state[-self.nr_act:] = action
        done = self.wrong_counter >= 20
        return self.state, rew, done, {}


def sample_expert(nr_iters=1000, iters_per_epoch=20):
    env = ModeEnv()
    traj_list = []
    for i in range(nr_iters):
        state = env.reset()
        state_list, action_list, reward_list = [], [], []
        for j in range(iters_per_epoch):
            act = get_optimal_action(state)
            state_list.append(state)
            action_list.append(act)
            # print(state, act)
            state, rew, _, _ = env.step(act)
            reward_list.append(rew)

        traj_list.append([state_list, action_list, reward_list])
        # print(traj_list[-1])
    return traj_list


if __name__ == "__main__":
    traj = sample_expert()
    with open("expert_samples/pg_modeEnv.pkl", "wb") as outfile:
        pickle.dump(traj, outfile)