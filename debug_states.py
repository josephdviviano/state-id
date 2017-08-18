#!/usr/bin/env python

import tables
import numpy as np
import os, sys

# load data
fid = tables.open_file('states.h5', mode='r')
data = fid.root.states
m, n = np.shape(data)

n_nans = np.zeros(n)
for i in range(n):
    d = data[:, i]
    n_nans[i] = len(np.where(np.isnan(d))[0])
    print('found {} nans for feature {}/{}'.format(n_nans[i], i+1, n))

import IPython; IPython.embed()
np.save('n_nans.npy', n_nans)

