#!/usr/bin/env python
import numpy as np
import tables
import os, sys
import glob


def zscore(ts):
    """Converts each timeseries to have 0 mean and unit variance."""
    means = np.tile(np.mean(ts, axis=1).T, [ts.shape[1], 1]).T
    stdev = np.tile(np.std(ts, axis=1).T, [ts.shape[1], 1]).T
    return((ts-means)/stdev)


def replace_timeseries(ts):
    """replace any all-zero timeseries with white noise"""
    idx = np.where(np.sum(np.abs(ts), axis=1) == 0)[0]
    if len(idx) != 0:
        for i in idx:
            ts[i, :] = np.random.uniform(size=ts.shape[1])
    return(ts, len(idx))


def tukeywin(win_length, alpha=0.75):
    """
    The Tukey window, also known as the tapered cosine window, is a cosine lobe
    of width alpha * N / 2 that is convolved with a rectangular window of width
    (1 - alpha / 2). At alpha = 1 it becomes rectangular, and at alpha = 0 it
    becomes a Hann window.
    http://leohart.wordpress.com/2006/01/29/hello-world/
    http://www.mathworks.com/access/helpdesk/help/toolbox/signal/tukeywin.html
    """
    # special cases
    if alpha <= 0:
        return np.ones(win_length)
    elif alpha >= 1:
        return np.hanning(win_length)

    # normal case
    x = np.linspace(0, 1, win_length)
    window = np.ones(x.shape)

    # first condition: 0 <= x < alpha/2
    c1 = x < alpha/2
    window[c1] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[c1] - alpha/2)))

    # second condition already taken care of
    # third condition 1 - alpha / 2 <= x <= 1
    c3 = x >= (1 - alpha/2)
    window[c3] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[c3] - 1 + alpha/2)))

    return window

def dynamic_connectivity(ts, win_length, win_step):
    """
    Calculates dynamic (sliding window) connectivity from input timeseries data,
    and outputs a roi x window matrix.
    """
    n_roi, n_tr = ts.shape

    # initialize the window
    idx_start = 0
    idx_end = win_length # no correction: 0 indexing balances numpy ranges

    # precompute the start and end of each window
    windows = []
    while idx_end <= n_tr-1:
        windows.append((idx_start, idx_end))
        idx_start += win_step
        idx_end += win_step

    # store the upper half of each connectivity matrix for each window
    idx_triu = np.triu_indices(n_roi, k=1)
    output = np.zeros((len(idx_triu[0]), len(windows)))

    # calculate taper (downweight early and late timepoints)
    taper = np.atleast_2d(tukeywin(win_length))

    for i, window in enumerate(windows):
        # extract sample, apply taper
        sample = ts[:, window[0]:window[1]] * np.repeat(taper, n_roi, axis=0)

        # keep upper triangle of correlation matrix
        test = np.corrcoef(sample)
        output[:, i] = np.corrcoef(sample)[idx_triu]

    return(output.T)


data_dir = '/mnt/tigrlab/projects/jviviano/data/dynamics/data'
timeseries = glob.glob(os.path.join(data_dir, '*.csv'))
timeseries.sort()

names = []
ts_replaced = []
nans = []
for i, ts in enumerate(timeseries):

    # save subject name
    subject = '_'.join(os.path.basename(ts).split('_')[:5])

    # load z-scored time series data, replace all-0 timeseries with white noise
    ts = np.loadtxt(ts, delimiter=',')
    ts, n_replaced = replace_timeseries(ts)
    ts = zscore(ts)

    # calculate states
    states = dynamic_connectivity(ts, 30, 1)
    n_nans = len(np.where(np.isnan(states))[0])

    # initialize storage (assumes all data came from the same n ROIs)
    if i == 0:
        fid = tables.open_file("states.h5", "w")
        filters = tables.Filters(complevel=5, complib='blosc')
        all_states = fid.create_earray(fid.root, 'states',
            atom=tables.Atom.from_dtype(states.dtype), shape=(0, states.shape[-1]),
            expectedrows=(len(states)))

    # save data in order
    print('{}/{}: replaced {} ts, {} nans remain'.format(
        i+1, len(timeseries), n_replaced, n_nans))
    all_states.append(states)
    names.append(subject)
    ts_replaced.append(n_replaced)
    nans.append(n_nans)


fid.close()
np.save('subject_ids.npy', names)
np.save('subject_ts_replaced.npy', ts_replaced)
np.save('subject_nans.npy', nans)


