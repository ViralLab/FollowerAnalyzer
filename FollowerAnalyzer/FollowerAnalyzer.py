import numpy as np
import pandas as pd


def prepare_data(follower_data):
    """
    Prepares the follower data dictionary for the sliding histogram algorithm.
    """
    df = pd.DataFrame.from_dict(follower_data, orient="index").reset_index(
        names=["username"]
    )

    # Reversing the rank column
    # The algorithm was originally designed for a reversed ranking system (Last follower has rank 0)
    # This was later reversed since it is more intuitive to have rank 0 as the first follower
    df["rank"] = df["rank"].max() - df["rank"]
    df = df.sort_values(by="rank", ascending=True).reset_index(drop=True)

    # Adding a column for the upper bound of the timestamp
    df["upper_bound"] = df.sort_values(by="rank", ascending=False)[
        "timestamp"
    ].cummax()[::-1]

    df["follow_time_lower_bound"] = df["upper_bound"].copy()

    # Setting first 10 follower upperbound to their max timestamp to reduce noise
    max_rank = df["rank"].max()
    df.loc[max_rank - 10 : max_rank, "upper_bound"] = df.loc[
        max_rank - 10 : max_rank, "timestamp"
    ].max()

    # Adding a column for the relative timestamp
    df["relative_timestamp"] = (df["timestamp"] - df["timestamp"].min()) / (
        df["upper_bound"] - df["timestamp"].min()
    )

    return df


def get_start_indexes(data, size, stride):
    """
    Returns a list of start indexes for each window of the sliding histogram.
    The list is designed to cover all ranks in the data.
    To ensure that the windows cover the entire data, we increase the stride size of some random windows by 1.
    """
    # A list of start indexes for each window
    idx_list = np.arange(0, len(data) - size + 1, stride)

    # The number of remaining ranks at the end
    remaining_ranks = data.shape[0] - (idx_list[-1] + size)

    # Increasing the stride to cover all ranks
    n = remaining_ranks // len(idx_list)
    idx_list += n * np.arange(len(idx_list))
    remaining_ranks = remaining_ranks % len(idx_list)

    # Randomly adding strides to include the remaining ranks
    if remaining_ranks > 0:
        added_ranks = np.zeros_like(idx_list)
        insert_idx = np.random.choice(
            np.arange(1, idx_list.shape[0]), size=remaining_ranks, replace=False
        )
        added_ranks[insert_idx] = 1
        added_ranks = np.cumsum(added_ranks)
        idx_list += added_ranks

    return idx_list


def get_signals(follower_data, n_ranks=200, stride=40, creation_date_bins=12):
    """
    Computes the histograms (signals) for each window of the sliding histogram.
    Returns:

    - df:
        The follower dataframe as processed by the prepare_data function.
    - signals:
        A list of histograms (signals) for each window.
    - ranks:
        A list of median ranks for each window.
    - rank_bins:
        A list of tuples containing the start and end rank for each window.
    - relative_creation_bins:
        The bins used for the histogram, i.e., profile creation date bins,
        computed relatively to the lower bound and upper bound at each rank.
    """
    df = prepare_data(follower_data)

    idx_list = get_start_indexes(df, n_ranks, stride)

    lower_bound = df["timestamp"].min()

    signals = []
    ranks = []
    rank_bins = []

    relative_creation_bins = np.linspace(0, 1, creation_date_bins + 1)

    for start_idx in idx_list:
        bin_data = df.loc[start_idx : start_idx + n_ranks].copy()

        relative_timestamp = (bin_data["timestamp"] - lower_bound) / (
            bin_data["upper_bound"] - lower_bound
        )

        bin_data["timestamp_bins"], edges = pd.cut(
            relative_timestamp, bins=relative_creation_bins, retbins=True
        )

        signal = bin_data.groupby("timestamp_bins")["timestamp"].count().to_numpy()

        signals.append(signal)
        ranks.append(bin_data["rank"].median())
        rank_bins.append(
            (start_idx, min(start_idx + n_ranks, bin_data["rank"].iloc[-1]))
        )

    return df, signals, ranks, rank_bins, relative_creation_bins


def get_anomaly_scores(follower_data, n_ranks=200, stride=40, creation_date_bins=12):
    """
    Computes the anomaly scores for each follower in the follower data.
    """

    df, signals, ranks, rank_bins, relative_creation_bins = get_signals(
        follower_data, n_ranks, stride, creation_date_bins
    )

    signal_arr = np.vstack(signals)
    median_arr = np.median(signal_arr, axis=0)
    q1 = np.quantile(signal_arr, q=0.25, axis=0)
    q3 = np.quantile(signal_arr, q=0.75, axis=0)
    iqr_arr = q3 - q1

    distance_to_median = signal_arr - median_arr
    # Increasing the distance to median by one to accomodate for the added 1 to IQR
    distance_to_median += np.sign(distance_to_median)
    signal_scores = np.divide(
        distance_to_median, iqr_arr + 1
    )  # Adding one to the IQR array to avoid zero-division

    rank_bins = np.array(rank_bins)
    ranks = np.array(ranks)

    rank_centers = np.median(rank_bins, axis=1)

    score_list = []
    for idx, row in df.iterrows():
        rank = row["rank"]
        relative_timestamp = row["relative_timestamp"]

        cond = (rank >= rank_bins[:, 0]) & (rank <= rank_bins[:, 1])
        # Getting weight array for the averaged weight of signal scores
        weight_arr = (
            n_ranks / 2 - np.abs(rank - rank_centers[cond])
        ) + 1  # Adding one to avoid div. by zero
        weight_arr = weight_arr / weight_arr.sum()

        # Finding which creation date bins contain this follower
        for idx, relative_timestamp_bin in enumerate(relative_creation_bins[1:]):
            if relative_timestamp <= relative_timestamp_bin:
                break

        score = np.sum(signal_scores[cond, idx] * weight_arr)
        score_list.append(score)

    usernames = df["username"].values
    score_dict = dict(zip(usernames, score_list))

    return score_dict


def estimate_following_time(follower_data, profile_creation_date=None):
    """
    Estimates the following time of each follower in the follower data.

    follower_data: dict
        Dictionary of the form:
            {
                username1: {"rank": 0, "timestamp": 238423022},
                username1: {"rank": 1, "timestamp": 238423022},
                ...
            }
        "rank" is the rank of the follower, the oldest follower has rank 0 and the newest follower has rank NumFollowers-1
        "timestamp" is the timestamp of the follower's creation date (Number of seconds since epoch).

    profile_creation_date: int
        Profile creation date of the user in seconds. Used to set the lower bound of the following time.
    """
    df = prepare_data(follower_data)

    # Computing lower bound
    # We set the lower bound to the running maximum profile creation date
    # i.e., we iterate through the followers from the oldest follower and record the most recent profile
    # creation date up to the current profile.
    if "follow_time_lower_bound" not in df.columns:
        df["follow_time_lower_bound"] = (
            df.sort_values(by="rank", ascending=False)["timestamp"]
            .cummax()
            .to_list()[::-1]
        )

    if profile_creation_date is not None:
        cond = df["follow_time_lower_bound"] < profile_creation_date
        df.loc[cond, "follow_time_lower_bound"] = profile_creation_date

    # Computing upper bound
    # We set the upper bound to the next unique maximum creation date after the current (rolling) maximum creation date
    unique_max_time = df["follow_time_lower_bound"].unique()

    # Creating a series with unique indexes for each unique lower bound date
    repl_dict = dict(zip(unique_max_time, range(len(unique_max_time))))
    ser = df["follow_time_lower_bound"].map(repl_dict.get)

    # Setting the upper bound as the next unique maximum creation date value
    repl_dict = dict(enumerate(unique_max_time))
    df["follow_time_upper_bound"] = [
        repl_dict.get(x - 1, unique_max_time[0]) for x in ser
    ]

    # Computing the ranks of the followers defining the upper and lower bounds
    bound_rank_dict = dict(
        zip(*np.unique(df["follow_time_lower_bound"].sort_values(), return_index=True))
    )
    urank = df["rank"].max() - df["follow_time_upper_bound"].map(bound_rank_dict.get)
    lrank = df["rank"].max() - df["follow_time_lower_bound"].map(bound_rank_dict.get)

    df["follow_time"] = pd.Timestamp(0, unit="s")

    lbound = df["follow_time_lower_bound"]
    ubound = df["follow_time_upper_bound"]

    cond = urank == lrank
    df.loc[cond, "follow_time"] = lbound[cond]
    df.loc[~cond, "follow_time"] = lbound[~cond] + (
        (ubound[~cond] - lbound[~cond])
        * (df.loc[~cond, "rank"] - lrank[~cond])
        / (urank[~cond] - lrank[~cond])
    )

    follow_time_dict = df.set_index("username")["follow_time"].to_dict()

    return follow_time_dict


class FollowerAnalyzer:
    """
    FollowerAnalyzer class for analyzing the follower data of a user.
    It allows for computing the anomaly scores and following times of the followers of a certain user.
    Requires as input a dictionary of the form:
        {
            username1: {"rank": 0, "timestamp": 238423037},
            username1: {"rank": 1, "timestamp": 238445022},
            ...
        }

        "rank" is the rank of the follower, the oldest follower has rank 0 and the newest follower has rank NumFollowers-1
        "timestamp" is the timestamp of the follower's creation date (Number of seconds since epoch).

    Methods:

    get_anomaly_scores:
        Computes the anomaly scores of the followers.

    get_following_times:
        Computes the following times of the followers.

    get_dataframe:
        Returns a dataframe containing the follower usernames, ranks, and timestamps.
        Optionally, it can compute and include the anomaly scores and following times of the followers.
    """

    def __init__(self, follower_data, profile_creation_date=None):
        """
        follower_data: dict
            Dictionary of the form:
                {
                    username1: {"rank": 0, "timestamp": 238423022},
                    username1: {"rank": 1, "timestamp": 238423022},
                    ...
                }
            "rank" is the rank of the follower, the oldest follower has rank 0 and the newest follower has rank NumFollowers-1
            "timestamp" is the timestamp of the follower's creation date (Number of seconds since epoch).

        profile_creation_date: int
            Profile creation date of the user in seconds. Used to set the lower bound of the following time.
        """

        self.follower_data = follower_data
        self.profile_creation_date = profile_creation_date

    def get_anomaly_scores(self, n_ranks=200, stride=40, creation_date_bins=12):
        """
        n_ranks: int
            Number of ranks to include in each window.

        stride: int
            Number of ranks to skip between windows.

        creation_date_bins: int
            Number of bins to use for the creation date of the followers.

        Returns a dictionary that maps each follower username to its anomaly score.
        """
        return get_anomaly_scores(
            self.follower_data, n_ranks, stride, creation_date_bins
        )

    def get_following_times(self):
        """
        Returns a dictionary that maps each follower username to its following time.
        """
        return estimate_following_time(self.follower_data, self.profile_creation_date)

    def get_dataframe(
        self,
        include_anomaly_scores=False,
        include_follow_times=False,
        n_ranks=200,
        stride=40,
        creation_date_bins=12,
    ):
        """
        include_anomaly_scores: bool
            If True, the dataframe will include the anomaly scores of the followers.

        include_follow_times: bool
            If True, the dataframe will include the following times of the followers.

        Returns a dataframe containing the follower usernames, ranks, and timestamps.
        """
        df = prepare_data(self.follower_data)[["username", "rank", "timestamp"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

        if include_follow_times:
            follow_times = self.get_following_times()
            df["follow_time"] = df["username"].map(follow_times)
            df["follow_time"] = pd.to_datetime(df["follow_time"], unit="s")

        if include_anomaly_scores:
            anomaly_scores = self.get_anomaly_scores(
                n_ranks, stride, creation_date_bins
            )
            df["anomaly_score"] = df["username"].map(anomaly_scores)

        df["rank"] = df["rank"].max() - df["rank"]

        return df
