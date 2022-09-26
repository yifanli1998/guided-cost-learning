import os
import pickle
import numpy as np
import pandas as pd


def prepare_data(
    trips, min_number_trips=500, return_normed=True, drop_columns=[]
):
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
    nr_trips_with_mode = trips[[
        col for col in trips.columns if col.startswith("Mode")
    ]].sum()
    included_modes = list(
        nr_trips_with_mode[nr_trips_with_mode > min_number_trips
                           ].index.tolist()
    )
    print("included_modes", included_modes)
    # TODO: group into public transport, slow transport, car, shared car
    dataset = dataset[dataset[included_modes].sum(axis=1) > 0]
    print("after removing other modes:", len(dataset))

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
    feat_array = dataset[feat_cols]
    feat_mean, feat_std = feat_array.mean(), feat_array.std()
    feat_array_normed = (feat_array - feat_mean) / feat_std

    # get labels
    labels = dataset[included_modes]

    traj_list = []
    for _, traj in dataset.groupby("person_day"):
        row_inds = traj.index
        states = np.array(feat_array_normed.loc[row_inds])
        actions = np.array(labels.loc[row_inds])
        rew = np.ones(len(row_inds))
        traj_list.append([states, actions, rew])

    return traj_list, (feat_mean, feat_std)


class ModeEnv:

    def __init__(self, path_traj=os.path.join("expert_samples", "mobis.pkl")):
        # load trajectories --> this is the only data we have
        with open(path_traj, "rb") as outfile:
            (self.traj_list, feat_mean, feat_std) = pickle.load(outfile)
        print(f"Loaded {len(self.traj_list)} trajectories")
        # Trajectories has format [[states, actions, rewards]] - access states:
        self.nr_feats = self.traj_list[0][0].shape[1]
        self.nr_act = self.traj_list[0][1].shape[1]
        self.nr_traj = len(self.traj_list)
        self.traj_order = np.random.permutation(len(self.traj_list))
        self.current_ind = 0
        self.current_traj = self.traj_list[self.traj_order[self.current_ind]]
        # TODO: possibly normalize the reward such that the rewards of one
        # episode sum up to 1

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
        rew = 1 if action_is_correct else 0
        # TODO: give part_reward if action is similar mode

        done = False
        if self.episode_step == len(self.current_traj[0]) - 1:
            done = True
            self.episode_step = 0
        else:
            self.episode_step += 1
            self.state = self.current_traj[0][self.episode_step]
        # TODO: in dummy env we set done=True if the model made several mistakes
        return self.state, rew, done, {}


if __name__ == "__main__":
    trips = pd.read_csv("../../data/mobis/trips_features.csv")
    traj_list, (feat_mean, feat_std) = prepare_data(trips)
    with open(os.path.join("expert_samples", "mobis.pkl"), "wb") as outfile:
        pickle.dump((traj_list, feat_mean, feat_std), outfile)
