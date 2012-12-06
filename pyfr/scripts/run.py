#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from argparse import ArgumentParser
from ConfigParser import SafeConfigParser

from mpi4py import MPI
import numpy as np

from pyfr.backends.cuda import CudaBackend
from pyfr.mesh_partition import MeshPartition
from pyfr.rank_allocator import get_rank_allocation
from pyfr.util import proxylist

def main():
    dt = 0.01

    be = CudaBackend()

    # Load the config file
    cfg = SafeConfigParser()
    cfg.readfp(open(sys.argv[1]))

    # Open up the mesh
    mfile = cfg.get('mesh', 'file')
    mesh = np.load(os.path.join(os.path.dirname(sys.argv[1]), mfile))

    # Get the mapping from physical ranks to MPI ranks
    rallocs = get_rank_allocation(mesh, cfg)

    # Construct the mesh partition
    mpt = MeshPartition(be, rallocs, mesh, 2, cfg)
    ele_banks = mpt.ele_banks

    # Forwards Euler (u += Δt*f) on each element type
    euler_step = proxylist([be.kernel('axnpby', eb[0], eb[1])
                             for eb in ele_banks])

    # Get our own queue
    q = be.queue()

    for i in xrange(1):
        # [in] u(0), [out] f(1)
        mpt(0, 1)

        print ele_banks[0][1].get().max()

        # An euler step on f
        q % euler_step(1.0, dt)


if __name__ == '__main__':
    main()
