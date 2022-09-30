import numpy as np
import os
import pandas as pd
import argparse

from experts.PG import PG
from cost import CostNN
from mode_env import ModeEnv

from v2g4carsharing.mode_choice_model.evaluate import mode_share_plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", default=300, type=int, help="to load")
    parser.add_argument(
        "-m", "--model", type=str, default="test", help="save name"
    )
    args = parser.parse_args()

    model = args.model
    eval_env = ModeEnv(os.path.join("expert_samples", "mobis_test.pkl"))
    N_ACTIONS = eval_env.nr_act  # action_space.n
    state_shape = eval_env.nr_feats  # env.observation_space.shape

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

    rew, max_rew = 0, 0
    act_sim, act_real = [], []
    print("Available epochs: ", eval_env.nr_traj)
    print("Number of actions:", N_ACTIONS)
    for eval_epoch in range(eval_env.nr_traj):
        s = eval_env.reset()
        # 20 is much higher than the possible steps in mode env
        for k in range(20):
            real_action = np.argmax(
                eval_env.current_traj[1][eval_env.episode_step]
            )
            action_probs = policy.predict_probs(np.array([s]))[0]
            a = np.random.choice(policy.n_actions, p=action_probs)
            # print([round(elem, 2) for elem in action_probs])
            act_sim.append(a)
            act_real.append(real_action)
            s, r, done, info = eval_env.step(a)
            rew += r
            if done:
                break
        max_rew += (k + 1)
    print(f"Test reward: {rew} / {max_rew}, {round(rew/max_rew, 2)}")

    # make plot
    included_modes = np.array(
        [
            'Mode::Bicycle', 'Mode::Bus', 'Mode::Car',
            'Mode::CarsharingMobility', 'Mode::LightRail',
            'Mode::RegionalTrain', 'Mode::Train', 'Mode::Tram', 'Mode::Walk'
        ]
    )
    act_real = included_modes[np.array(act_real)]
    act_sim = included_modes[np.array(act_sim)]
    print("Accuracy:", np.sum(act_real == act_sim) / len(act_real))
    # # 1) get mobis data
    # mobis_data = pd.read_csv(
    #     os.path.join(args.mobis_data_path, "trips_features.csv")
    # )
    # mobis_data = mobis_data[included_modes]
    # labels_mobis = np.array(mobis_data.columns
    #                         )[np.argmax(np.array(mobis_data), axis=1)]
    # 2) get sim data
    mode_share_plot(
        act_real, act_sim, out_path=os.path.join("trained_models", model)
    )
