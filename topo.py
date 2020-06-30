import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from giotto.diagrams import BettiCurve, Amplitude
from giotto.diagrams._utils import _subdiagrams
from giotto.homology import VietorisRipsPersistence
from giotto.time_series import Resampler, TakensEmbedding
from scipy import signal
from sklearn.preprocessing import normalize, MinMaxScaler


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


def gaus_kde(x, window, right_lim, bigm, scale):
    # Apply gaussian a spike + padding
    spike_pad = lambda x: np.pad(window, (int(x), int(right_lim - x - bigm + 1)), 'constant',
                                 constant_values=(0, 0))

    #  Sum all gaussian spikes
    custom_kde = lambda x: np.sum(np.array(list(map(spike_pad, x * scale))), axis=0)

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

    #  plt.axvline(x=last_spike_mean * scale)
    #  plt.axvline(x=last_spike_min * scale)
    #  plt.axvline(x=last_spike_max * scale)

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


def topo_pipeline(x_input, bigm=160, rel_std=16, scale=100, period=1, time_delay=1, dimension=3,
                  homology_dimensions=None, max_edge_length=20, n_values_bc=2000, tech='raw', takens=True):
    if homology_dimensions is None:
        homology_dimensions = [0, 1]

    #  Drop id and neuron_id
    x_np = x_input.values[:, 2:]

    x_resamp = 0

    if tech == 'kde':
        window = signal.gaussian(bigm, std=bigm / rel_std)

        # right_lim = int((x_input[:, -1].max() + (bigm / 2))) # In theory
        right_lim = int(np.ceil(x_input.values[:, -1].max() / 10 + 1) * 10 * scale)  #  In practical

        #  Create gaussian spikes
        print('Creating gaussian spikes')
        x_gaus = gaus_kde(x_np, window, right_lim, bigm, scale)

        #  Resampling
        print('Resampling')
        resamp = Resampler(period=period)
        x_resamp = resamp.fit_transform(x_gaus.T).T

    elif tech == 'raw':
        x_resamp = x_np

    elif tech == 'intervals':
        x_resamp = x_input.values[:, 3:] - x_input.values[:, 2:-1]

    if takens:
        print('Creating Takens Embedding')
        te = TakensEmbedding(time_delay=time_delay, dimension=dimension, parameters_type='fixed')
        x_tak = np.apply_along_axis(te.fit_transform, 1, x_resamp)
    else:
        x_tak = np.expand_dims(x_resamp, axis=2)

    print('Creating V-R Persistence Diagrams')
    vrp = VietorisRipsPersistence(max_edge_length=max_edge_length, homology_dimensions=homology_dimensions, n_jobs=-1)
    x_vrpd = vrp.fit_transform(x_tak)

    print('Creating Betti curves')
    bc = BettiCurve(n_values=n_values_bc, n_jobs=-1)
    bcs = bc.fit_transform(x_vrpd)

    return bcs, x_vrpd


def gen_topo_features(bcs, x_vrpd, sum_range_0=120, sum_range_1=60):
    print("bc_area")
    X_betti_0 = pd.DataFrame(bcs[:, 0, 0:sum_range_1])
    X_sum0 = pd.DataFrame(X_betti_0.sum(axis=1), columns=['bc0_area'])
    X_betti_1 = pd.DataFrame(bcs[:, 1, 0:sum_range_0])
    X_sum1 = pd.DataFrame(X_betti_1.sum(axis=1), columns=['bc1_area'])

    bc_area = X_sum1[['bc1_area']].join(X_sum0[['bc0_area']])

    print("tropical")

    def yi_minus_xi(X):
        return X[:, :, 1] - X[:, :, 0]

    def feature_1(X, dim):
        X = _subdiagrams(X, [dim])
        return np.sum((X[:, :, 0] * yi_minus_xi(X)), axis=1) / X.shape[1]

    def feature_2(X, dim):
        X = _subdiagrams(X, [dim])
        return np.sum((np.max(X[:, :, 1], axis=1).reshape((X.shape[0], 1)) - X[:, :, 1]) * (yi_minus_xi(X)), axis=1) / \
               X.shape[1]

    def feature_3(X, dim):
        X = _subdiagrams(X, [dim])
        return np.sum(((X[:, :, 0] ** 2) * (yi_minus_xi(X) ** 4)), axis=1) / X.shape[1]

    def feature_4(X, dim):
        X = _subdiagrams(X, [dim])
        return np.sum(((np.max(X[:, :, 1], axis=1).reshape((X.shape[0], 1)) - X[:, :, 1]) ** 2) * (yi_minus_xi(X) ** 4),
                      axis=1) / X.shape[1]

    def feature_5(X, dim):
        X = _subdiagrams(X, [dim])
        return np.max(yi_minus_xi(X), axis=1) / X.shape[1]

    fts_func = [feature_1, feature_2, feature_3, feature_4, feature_5]

    fts = {}
    for feature_number in range(5):
        for dim in range(2):
            fts[f'f{feature_number}_d{dim}'] = fts_func[feature_number](x_vrpd, dim)

    tropical = pd.DataFrame(fts)

    print("amplitudes")
    amps = {}
    for metric in ['bottleneck', 'wasserstein', 'landscape', 'betti', 'heat']:
        amp = Amplitude(metric=metric, metric_params=None, n_jobs=-1)
        x_amp = amp.fit_transform(x_vrpd)
        amps[metric] = x_amp.ravel()

    amplitudes = pd.DataFrame(amps)

    m_1 = np.sum(normalize(bcs[:, 1], norm='max', axis=1) * np.power(np.arange(bcs.shape[2]), 1), axis=1)
    m_2 = np.sum(normalize(bcs[:, 1], norm='max', axis=1) * np.power(np.arange(bcs.shape[2]), 2), axis=1)

    print("moments")
    moments = {
        'moment_1': m_1,
        'moment_2': m_2,
    }
    moments_df = pd.DataFrame(moments)

    topo_features = bc_area.join(tropical).join(amplitudes).join(moments_df).drop(columns=['f0_d0', 'f2_d0'])

    print("scaler")
    scaler = MinMaxScaler()
    topo_features[topo_features.columns] = scaler.fit_transform(topo_features[topo_features.columns])

    return topo_features
