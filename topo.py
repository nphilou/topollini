import matplotlib.pyplot as plt
import numpy as np


def plot_raw_ts(xy, neuron_id, n_ts, separate=False):
    if neuron_id == -1:
        xy_filtered = xy
    else:
        xy_filtered = xy[xy.neuron_id == neuron_id]

    if n_ts == -1:
        x_nid = xy_filtered
    else:
        x_nid = xy_filtered.groupby('TARGET').head(n_ts)

    xy_sample_0 = x_nid[x_nid.TARGET == 0]
    xy_sample_1 = x_nid[x_nid.TARGET == 1]

    x0 = xy_sample_0.loc[:, 'timestamp_0':'timestamp_49']
    x1 = xy_sample_1.loc[:, 'timestamp_0':'timestamp_49']

    plt.figure(figsize=(20, 6))
    plt.plot(x0.T, c='blue')

    if separate:
        plt.show()
        plt.figure(figsize=(20, 6))

    plt.plot(x1.T, c='red')
    plt.show()


def gaus_kde(x, window, min_right_lim, bigm, scale):
    # Apply gaussian a spike + padding
    spike_pad = lambda x: np.pad(window, (int(x), int(min_right_lim - x - bigm + 1)), 'constant',
                                 constant_values=(0, 0))

    #  Sum all gaussian spikes
    custom_kde = lambda x: np.sum(np.array(list(map(spike_pad, x * scale))),
                                  axis=0)

    return np.array(list(map(custom_kde, x)))


def plot_gaus_kde_sum(x_gaus, y_np, xy_nid, scale):
    y_0 = y_np.ravel() == 0
    y_1 = y_np.ravel() == 1

    plt.figure(figsize=(20, 4))
    plt.plot(np.sum(x_gaus[y_0].T, axis=1) / len(x_gaus[y_0]), c='blue')
    plt.plot(np.sum(x_gaus[y_1].T, axis=1) / len(x_gaus[y_1]), c='red')

    last_spike_mean = xy_nid.timestamp_49.mean()
    last_spike_min = xy_nid.timestamp_49.min()
    last_spike_max = xy_nid.timestamp_49.max()

    plt.axvline(x=last_spike_mean * scale)
    plt.axvline(x=last_spike_min * scale)
    plt.axvline(x=last_spike_max * scale)

    plt.show()


def plot_gaus_kde(x_resamp, y_np, n0, n1, samefig=True):
    y_0 = y_np.ravel() == 0
    y_1 = y_np.ravel() == 1

    plt.figure(figsize=(20, 4))
    if n0 == -1:
        plt.plot(x_resamp[y_0].T, c='blue')
    else:
        for i in range(n0):
            plt.plot(x_resamp[y_0][i].T, c='blue')

    if not samefig:
        plt.show()
        plt.figure(figsize=(20, 4))

    if n1 == -1:
        plt.plot(x_resamp[y_1].T, c='red')
    else:
        for j in range(n1):
            plt.plot(x_resamp[y_1][j].T, c='red')
    plt.show()
