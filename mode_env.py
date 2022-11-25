import os
import pickle
import numpy as np
import pandas as pd

category = {
    "Mode::Bicycle": 0,
    "Mode::Bus": 1,
    "Mode::Car": 2,
    "Mode::CarsharingMobility": 2,
    "Mode::LightRail": 1,
    "Mode::RegionalTrain": 2,
    "Mode::Train": 2,
    "Mode::Tram": 1,
    "Mode::Walk": 0
}
included_modes = [
    "Mode::Bicycle", "Mode::Bus", "Mode::Car", "Mode::CarsharingMobility",
    "Mode::LightRail", "Mode::RegionalTrain", "Mode::Train", "Mode::Tram",
    "Mode::Walk"
]
int_to_modelabel = {i: mode for i, mode in enumerate(included_modes)}


def prepare_data(trips, drop_columns=[], use_prevmode=True):
    # remove prevmode if desired
    if not use_prevmode:
        drop_columns = drop_columns + [
            col for col in trips.columns if col.startswith("feat_prev_")
        ]
        print("Prevmode not included, dropping columns")

    # get day
    trips["started_at_destination"] = pd.to_datetime(
        trips["started_at_destination"]
    )
    trips["day"] = trips["started_at_destination"].dt.date
    trips["person_day"] = trips["person_id"] + trips["day"].astype(str)
    # drop geometry if it exists
    dataset = trips.drop(
        ["geom", "geom_origin", "geom_destination"] + drop_columns,
        axis=1,
        errors="ignore"
    )
    # sort by person and time
    dataset = dataset.sort_values(["person_id", "day", "started_at_origin"])
    print("Dataset raw", len(dataset))
    # only include frequently used modes
    included_modes = [
        col for col in dataset.columns if col.startswith("Mode:")
    ]
    print("included_modes", included_modes)

    # only get feature and label columns
    feat_cols = [col for col in dataset.columns if col.startswith("feat")]
    dataset = dataset[feat_cols + included_modes + ["person_day"]]

    # drop columns with too many nans:
    max_unavail = 0.1  # if more than 10% are NaN
    feature_avail_ratio = pd.isna(dataset).sum() / len(dataset)
    features_not_avail = feature_avail_ratio[feature_avail_ratio > max_unavail
                                             ].index
    dataset.drop(features_not_avail, axis=1, inplace=True)
    print("dataset len now", len(dataset))

    # remove other NaNs (the ones because of missing origin or destination ID)
    dataset.dropna(inplace=True)
    print("dataset len after dropna", len(dataset))

    # normalize features
    feat_cols = [col for col in dataset.columns if col.startswith("feat")]
    prev_feat_cols = [col for col in feat_cols if col.startswith("feat_prev")]
    # prev_start_ind = feat_cols.index(prev_feat_cols[0])
    feat_array = dataset[feat_cols]
    feat_mean, feat_std = feat_array.mean(), feat_array.std()
    feat_mean[prev_feat_cols] = 0
    feat_std[prev_feat_cols] = 1
    feat_array_normed = (feat_array - feat_mean) / feat_std

    # get labels
    labels = dataset[included_modes]

    # counter = 0
    traj_list = []
    for _, traj in dataset.groupby("person_day"):
        row_inds = traj.index
        states = np.array(feat_array_normed.loc[row_inds])
        actions = np.array(labels.loc[row_inds])
        # # For testing
        # prev_mode = -1
        # for s, a in zip(states, actions):
        #     current_prev_mode = np.argmax(
        #         s[prev_start_ind:prev_start_ind + len(prev_feat_cols)]
        #     )
        #     if prev_mode != -1 and current_prev_mode != prev_mode:
        #         counter += 1  # print(current_prev_mode, prev_mode)
        #     prev_mode = np.argmax(a)
        rew = np.ones(len(row_inds))
        traj_list.append([states, actions, rew])

    return traj_list, (feat_mean, feat_std)


class ModeEnv:

    def __init__(
        self,
        path=os.path.join("expert_samples"),
        data="train",
        soft_reward=False,
        use_prevmode=True
    ):
        # load trajectories --> this is the only data we have
        data_name = "prevmode" if use_prevmode else "noprevmode"
        path_traj = os.path.join(path, data_name, f"mobis_{data}.pkl")
        with open(path_traj, "rb") as outfile:
            (self.traj_list, self.feat_mean, _) = pickle.load(outfile)
        print(f"Loaded {len(self.traj_list)} trajectories")
        # Trajectories has format [[states, actions, rewards]] - access states:
        self.soft_reward = soft_reward
        self.nr_feats = self.traj_list[0][0].shape[1]
        self.nr_act = self.traj_list[0][1].shape[1]
        self.nr_traj = len(self.traj_list)
        self.traj_order = np.random.permutation(len(self.traj_list))
        self.current_ind = 0
        self.feat_column_names = list(self.feat_mean.index)
        print("features", self.feat_column_names)
        self.mode_cols = [
            col for col in self.feat_column_names if "feat_prev" in col
        ]
        # self.wrong_counter = 0
        # self.correct_counter = 0
        if len(self.mode_cols) > 0:
            self.start_mode_col = self.feat_column_names.index(
                self.mode_cols[0]
            )
            self.end_mode_col = self.start_mode_col + len(self.mode_cols)
        self.current_traj = self.traj_list[self.traj_order[self.current_ind]]
        # TODO: possibly normalize the reward such that the rewards of one
        # episode sum up to 1
        self.compute_entropy()

    def compute_entropy(self):
        all_actions = [act for traj in self.traj_list for act in traj[1]]
        all_actions = np.argmax(np.array(all_actions), axis=1)
        _, counts = np.unique(all_actions, return_counts=True)
        # assert len(
        #     counts
        # ) == self.nr_act, "in train data there are not all actions"
        probs = counts / np.sum(counts)
        self.dist_labels = counts
        self.entropy = -1 * np.sum(probs * np.log(probs))

    def reset(self):
        self.current_ind = (self.current_ind + 1) % self.nr_traj
        self.current_traj = self.traj_list[self.traj_order[self.current_ind]]
        self.state = self.current_traj[0][0]
        self.episode_step = 0
        return self.state

    def step(self, action):
        # translate one hot to real
        real_action = np.argmax(self.current_traj[1][self.episode_step])
        action_is_correct = int(real_action == action)
        if self.soft_reward:
            if action_is_correct:
                rew = 1
            elif category[int_to_modelabel[real_action]
                          ] == category[int_to_modelabel[action]]:
                rew = 0.5
            else:
                rew = 0
        else:
            rew = 1 if action_is_correct else 0
        # TODO: give part_reward if action is similar mode

        done = False
        if self.episode_step == len(self.current_traj[0]) - 1:
            done = True
            self.episode_step = 0
        else:
            self.episode_step += 1
            self.state = self.current_traj[0][self.episode_step].copy()

        # Update prev mode feature - first set all to zero
        if len(self.mode_cols) > 0:
            self.state[self.start_mode_col:self.end_mode_col] = 0
            self.state[self.start_mode_col + action] = 1

        # # Testing if the real action corresponds to the new prev_mode entry
        # prev_mode_in_state = np.argmax(
        #     self.state[self.start_mode_col:self.end_mode_col]
        # )
        # if prev_mode_in_state != real_action and not done:
        #     self.wrong_counter += 1
        # else:
        #     self.correct_counter += 1

        # TODO: in dummy env we set done=True if the model made several mistakes
        return self.state, rew, done, {}


if __name__ == "__main__":
    out_name = "noprevmode"
    include_prevmode = False
    os.makedirs(os.path.join("expert_samples", out_name), exist_ok=True)

    trips = pd.read_csv("../../data/mobis/trips_features.csv")

    traj_list, (feat_mean,
                feat_std) = prepare_data(trips, use_prevmode=include_prevmode)

    # split in train and test
    nr_data = len(traj_list)
    cutoff = int(nr_data * 0.9)
    traj_list_train = traj_list[:cutoff]
    traj_list_test = traj_list[cutoff:]

    # save train and test
    with open(
        os.path.join("expert_samples", out_name, "mobis_train.pkl"), "wb"
    ) as outfile:
        pickle.dump((traj_list_train, feat_mean, feat_std), outfile)

    with open(
        os.path.join("expert_samples", out_name, "mobis_test.pkl"), "wb"
    ) as outfile:
        pickle.dump((traj_list_test, feat_mean, feat_std), outfile)
