#!/usr/bin/env python

import tables
from denoising_autoencoder import DenoisingAutoencoder

# load data
fid = tables.open_file('states.h5', mode='r')
data = fid.root.states
clf = DenoisingAutoencoder()
clf.fit(data)

import IPython; IPython.embed()
