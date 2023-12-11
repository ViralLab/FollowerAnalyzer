import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import LogNorm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates


def plot_follower_map(
    data, ax=None, zoom=None, log_scale=False, marker_color="gray", s=1, alpha=0.5
):
    """
    Parameters:
    -----------

    data: pd.DataFrame
        Dataframe containing the follower data.
        Must contain the columns "rank" and "timestamp".

    ax: matplotlib.axes.Axes
        Axes to plot on. If None, a new figure and axes is created.

    zoom: tuple
        Tuple of the form (start_rank, end_rank) to zoom in on a certain range of ranks.

    log_scale: bool
        Whether to use a logarithmic scale for the marker color.

    marker_color: str
        Column name of the dataframe to use for the marker color.
        If the column does not exist, it will be used as a constant color for all markers.

    s: float
        Marker size.

    alpha: float
        Marker transparency.
    """
    df = data.copy()

    if zoom is not None:
        cond = df["rank"].isin(np.arange(*zoom))
        df = df.loc[cond]

    if marker_color in df.columns:
        color = df[marker_color]
        cbar_flag = True
    else:
        color = marker_color
        cbar_flag = False

    x = df["rank"]
    y = df["timestamp"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 4))

    if log_scale and cbar_flag:
        sc = ax.scatter(x, y, c=color, s=s, alpha=alpha, norm=LogNorm())
    else:
        sc = ax.scatter(x, y, c=color, s=s, alpha=alpha)

    fig = ax.get_figure()
    if cbar_flag:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.ax.set_ylabel(marker_color)

    ax.set_xlabel("Old followers << Follower rank >> Recent followers")
    ax.set_ylabel("Follower profile creation dates")

    return ax


def get_heatmap_data(data, col, rank_bins=100, creation_date_bins=100, aggfunc="mean"):
    """
    Return a binned dataframe where the index is the creation date bin, the columns are the rank bins, and the values are "col"
    values aggregated using "aggfunc".

    The bins that do not have anomalies are replaced with NaN values so that they appear
    white in the plotted heatmap instead of having the color corresponding to zero.
    """
    df = data.copy()

    df["count"] = 1

    # Creating bins
    df["rank_bin"], edges = pd.cut(df["rank"], bins=rank_bins, retbins=True)

    df["timestamp"] = df["timestamp"].astype("int64") // 1e9
    df["timestamp_bin"], edges = pd.cut(
        df["timestamp"], bins=creation_date_bins, retbins=True
    )

    # Pivoting dataframe
    heatmap_data = df[["rank_bin", "timestamp_bin", col]].pivot_table(
        columns="rank_bin",
        index="timestamp_bin",
        values=col,
        aggfunc=aggfunc,
        dropna=False,
    )
    count_data = df[["rank_bin", "timestamp_bin", "count"]].pivot_table(
        columns="rank_bin",
        index="timestamp_bin",
        values="count",
        aggfunc="sum",
        dropna=False,
    )

    # Setting bins that do not contain followers to NaN
    count_data = count_data.replace({0: np.nan})
    cond = count_data.isna()
    heatmap_data[cond] = np.nan

    heatmap_data = heatmap_data.sort_index(ascending=False)

    return heatmap_data


def get_yticks(data):
    df = data.copy()

    idx = df.index.map(lambda x: pd.to_datetime(x.right, unit="s")).astype(
        "datetime64[ns]"
    )
    min_year, max_year = (idx[-1].year, idx[0].year)
    date_range = pd.date_range(
        start=f"{min_year}-01-01", end=f"{max_year}-01-03", freq="1y"
    ) + pd.Timedelta("1d")

    xp = idx.sort_values()
    fp = np.arange(df.shape[0])[::-1]
    yticks = np.interp(date_range, xp, fp).tolist()

    # Adding first and last years
    diff = np.diff(yticks).mean()
    yticks = [yticks[0] - diff] + yticks + [yticks[-1] + diff]
    yticklabels = (
        [min_year] + date_range.map(lambda x: x.year).tolist() + [max_year + 1]
    )

    return yticks, yticklabels


def get_xticks(data):
    df = data.copy()

    idx = df.columns.map(lambda x: x.right).astype("float").tolist()
    diff = np.diff(idx).mean()
    idx = [df.columns[0].left] + idx
    min_rank, max_rank = 0, int(idx[-1])

    step = max_rank // 20
    step -= step % 100
    rank_range = np.arange(min_rank, max_rank, step).astype("int")

    xp = idx
    fp = np.arange(df.shape[1] + 1)
    xticks = np.interp(rank_range, xp, fp).tolist()

    return xticks, rank_range


def plot_follower_heatmap(
    data,
    marker_color="anomaly_score",
    rank_bins=100,
    creation_date_bins=50,
    aggfunc="mean",
    ax=None,
    log_scale=False,
    cmap="viridis",
):
    """
    Plots a follower map as a heatmap.

    Parameters:
    -----------

    data: pd.DataFrame
        Dataframe containing the follower data.
        Must contain the columns "rank" and "timestamp".

    marker_color: str
        Column name of the dataframe to use for the marker color.
        If the column does not exist, it will be used as a constant color for all markers.

    rank_bins: int
        Number of heatmap bins to use for the rank of the followers.

    creation_date_bins: int
        Number of heatmap bins to use for the creation date of the followers.

    aggfunc: str
        Aggregation function to use for the heatmap values.
        Must be one of "mean", "median", "max", "min".

    ax: matplotlib.axes.Axes
        Axes to plot on. If None, a new figure and axes is created.

    log_scale: bool
        If True, the heatmap will be plotted using a logarithmic scale.

    cmap: str
        Colormap to use for the heatmap.
    """

    heatmap_data = get_heatmap_data(
        data, marker_color, rank_bins, creation_date_bins, aggfunc
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 4))

    if log_scale:
        norm = LogNorm()
    else:
        norm = None

    ax = sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap=cmap,
        norm=norm,
        cbar_kws={"label": marker_color},
    )

    yticks, yticklabels = get_yticks(heatmap_data)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    xticks, xticklabels = get_xticks(heatmap_data)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ax.set_xlabel("Old followers << Follower rank >> Recent followers")
    ax.set_ylabel("Profile creation date")

    return ax


#############################################################
# Detailed view
#############################################################


def plot_followers_with_scores(
    data, ax=None, zoom=None, marker_color=True, colorbar=True, fraction=0.15, s=1
):
    df = data.copy()

    if zoom is not None:
        cond = df["rank"].isin(np.arange(*zoom))
        df = df.loc[cond]

    if marker_color:
        color = df["anomaly_score"]
    else:
        color = "gray"

    x = df["rank"]
    y = df["timestamp"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 4))

    sc = ax.scatter(x, y, c=color, s=s, alpha=0.5)

    fig = ax.get_figure()
    if marker_color and colorbar:
        cbar = fig.colorbar(sc, ax=ax, fraction=fraction, pad=fraction / 3)
        cbar.ax.set_ylabel("Anomaly score")

    ax.set_xlabel("Old followers << Follower rank >> Recent followers")
    ax.set_ylabel("Follower profile creation dates")

    return ax


def transform_coordinate(coord, axis, ax):
    tick_fun_dict = {"x": lambda x: x.get_xticks(), "y": lambda x: x.get_yticks()}

    ticklabel_fun_dict = {
        "x": lambda x: x.get_xticklabels(),
        "y": lambda x: x.get_yticklabels(),
    }

    if axis == "y":
        coord = coord.year + coord.dayofyear / 366
    tick_fun = tick_fun_dict[axis]
    ticklabel_fun = ticklabel_fun_dict[axis]
    ticks = tick_fun(ax)
    ticklabels = ticklabel_fun(ax)
    tickvalues = [float(t.get_text()) for t in ticklabels]

    return np.interp(coord, tickvalues, ticks)


def create_zoom_patch(axMain, data, zoom, heatmap):
    df = data.copy()

    cond = df["rank"].isin(np.arange(zoom[0], zoom[1]))
    min_date = df.loc[cond, "timestamp"].min() - pd.Timedelta(days=90)
    max_date = df.loc[cond, "timestamp"].max() + pd.Timedelta(days=180)
    ymin = mdates.date2num(min_date)
    ymax = mdates.date2num(max_date)

    if heatmap:
        # Convert data coordinates to ax coordinates
        x1 = transform_coordinate(zoom[0], "x", axMain)
        x2 = transform_coordinate(zoom[1], "x", axMain)
        y1 = transform_coordinate(min_date, "y", axMain)
        y2 = transform_coordinate(max_date, "y", axMain)
    else:
        x1, x2 = zoom
        y1 = mdates.date2num(min_date)
        y2 = mdates.date2num(max_date)

    height = y2 - y1
    width = x2 - x1

    patch = Rectangle(xy=(x1, y1), width=width, height=height)

    return patch


def plot_heatmap(
    data,
    col,
    cbar_label=None,
    rank_bins=100,
    creation_date_bins=50,
    aggfunc="mean",
    ax=None,
    log_scale=False,
    cmap="flare_r",
    fraction=0.15,
):
    heatmap_data = get_heatmap_data(data, col, rank_bins, creation_date_bins, aggfunc)

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 4))

    if log_scale:
        norm = LogNorm()
    else:
        norm = None

    ax = sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap=cmap,
        norm=norm,
        cbar_kws={"label": cbar_label, "fraction": fraction},
    )

    yticks, yticklabels = get_yticks(heatmap_data)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    xticks, xticklabels = get_xticks(heatmap_data)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ax.set_xlabel("Old followers << Follower rank >> Recent followers")
    ax.set_ylabel("Profile creation date")

    return ax


def plot_detailed_view(data, zoom_arr, heatmap=True, sample=None, window_color="red"):
    """
    Plots the entire follower map in the upper subplot.
    Plots the zoomed-in follower map(s) in the lower subplots.

    Parameters:
    -----------

    data: pd.DataFrame
        Dataframe containing the follower map data and anomaly scores.

    zoom_arr: list of tuples
        List of tuples containing the zoom range for each subplot.

    heatmap: bool
        If True, the heatmap is plotted in the upper subplot.

    sample: int
        Number of followers to sample for the upper subplot.
        Can be used if the number of followers is too large and a scatter plot is required.
        There is no guarantee that the sampled followers will be representative of the follower distribution.
        The lower subplots will always plot all the followers in the zoom range.

    window_color: str
        Color of the zoom window.
    """

    df = data.copy()

    n_subplots = len(zoom_arr)
    fig_def = (
        "A" * n_subplots + ";" + "".join([chr(i) for i in range(66, 66 + n_subplots)])
    )
    fig, axs = plt.subplot_mosaic(fig_def, figsize=(16, 8), tight_layout=True)

    axMain = axs["A"]
    if heatmap:
        ax = plot_heatmap(
            df,
            "anomaly_score",
            rank_bins=200,
            creation_date_bins=100,
            aggfunc="mean",
            ax=axMain,
            log_scale=False,
            cmap="viridis",
            fraction=0.15 / n_subplots,
        )
    else:
        if sample is not None and sample < df.shape[0]:
            df_plot = df.sample(sample, replace=False)
        else:
            df_plot = df
        ax = plot_followers_with_scores(
            df_plot, marker_color=True, ax=axMain, fraction=0.15 / n_subplots
        )

    patch_list = []

    for zoom, ax_id in zip(zoom_arr, fig_def[fig_def.find(";") + 1 :]):
        ax = axs[ax_id]

        patch = create_zoom_patch(axMain, df, zoom, heatmap)
        patch_list.append(patch)

        ax = plot_followers_with_scores(df, zoom=zoom, marker_color=True, ax=ax)
        ax.set_title(f"Zooming on ranks {zoom[0]} - {zoom[1]}")

    collection = PatchCollection(
        patch_list, facecolor="none", alpha=1, edgecolor=window_color, linewidth=2
    )
    axMain.add_collection(collection)

    return axs
