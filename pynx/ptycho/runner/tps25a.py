#! /opt/local/bin/python
# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import sys
import time
import locale
import timeit

from ...utils import h5py
import numpy as np
import pandas as pd  # <--- Added pandas import

from .runner import PtychoRunner, PtychoRunnerScan, PtychoRunnerException, default_params

helptext_beamline = """
Script to perform a ptychography analysis on data from 25A@TPS

Example:
    pynx-ptycho-tps25a h5file=pty_41_61_master.h5 specfile=scan.dat probe=gauss,3e-6x3e-6
      algorithm=analysis,ML**100,DM**200,nbprobe=3,probe=1 verbose=10 save=all saveplot liveplot

Command-line arguments (beamline-specific):
    scanfile=/some/dir/to/scan.txt: path to text file with motor positions [mandatory]

    data=path/to/data.h5: path to hdf5 data file, with images, wavelength, detector distance [mandatory]
    
    h5data=entry/data/data_%06d: relative path to data inside hdf5 file [default=entry/data/data_%06d]
"""

params_beamline = {'scanfile': None, 'data': None, 'object': 'random,0.9,1,-.2,.2', 'instrument': 'TPS 25A',
                   'h5data': 'entry/data/data_%06d'}

params = default_params.copy()
for k, v in params_beamline.items():
    params[k] = v


class PtychoRunnerScanTPS25A(PtychoRunnerScan):
    def __init__(self, params, scan, timings=None):
        super(PtychoRunnerScanTPS25A, self).__init__(params, scan, timings=timings)

    # --- MODIFICATION: load_scan using pandas with default motor_prefix, and source ---
    def load_scan(self, motor_prefix='cisamf',source='user_setpoint'):
        # Usage: [motor_prefix]_[x/y/z]_[source]
        # Example 1 for read user_setpoint:  motor_prefix='cisamf', source='user_setpoint' -> cisamf_x_user_setpoint in primary.csv
        # Example 2 for read readback value: motor_prefix='cisamf', source=''              -> cisamf_x in primary.csv
        print("Reading motor positions from: ", self.params['scanfile'])
        df = pd.read_csv(self.params['scanfile'])
        print(f"Searching motor scan with motor_prefix='{motor_prefix}', source='{source}' for x and z axes...")

        # Find column names
        # Logic: Column name must contain the motor_prefix, ('x' or 'z'), and source
        col_x = [c for c in df.columns if motor_prefix in c and 'x' in c and source in c]
        col_z = [c for c in df.columns if motor_prefix in c and 'z' in c and source in c]

        name_frame = 'seq_num'

        if not col_x or not col_z:
            raise PtychoRunnerException(f"Could not find columns matching '{motor_prefix}', '{source}', and ('x' or 'z'). Columns found: {df.columns.tolist()}")
        if len(col_x) > 1 or len(col_z) > 1:
            raise PtychoRunnerException(f"Found multiple columns matching '{motor_prefix}', '{source}', and ('x' or 'z'). Matchings found: {col_x}, {col_z}")
        
        name_x = col_x[0]
        name_z = col_z[0]

        print(f"Using columns: X -> {name_x}, Y(Z) -> {name_z}" + (f", Frame -> {name_frame}" if name_frame else ""))

        # Assign data from DataFrame to numpy arrays
        # Note: PyNX expects self.x and self.y in METERS. 
        self.x = df[name_x].to_numpy(copy=True, dtype=np.float64)
        self.y = df[name_z].to_numpy(copy=True, dtype=np.float64)
        
        # frames -> frames_idx_from_eiger, with more index since injection
        self.frames_idx_from_eiger = df[name_frame].values.astype(int)
        self.imgn = np.arange(len(self.x), dtype=int)

        # Handle moduloframe and maxframe
        if self.params['moduloframe'] is not None:
            n1, n2 = self.params['moduloframe']
            idx = np.where(self.imgn % n1 == n2)[0]
            self.imgn = self.imgn.take(idx)
            self.x = self.x.take(idx)
            self.y = self.y.take(idx)
            self.frames_idx_from_eiger = self.frames_idx_from_eiger.take(idx)

        if self.params['maxframe'] is not None:
            N = self.params['maxframe']
            if len(self.imgn) > N:
                print("MAXFRAME: only using first %d frames" % (N))
                self.imgn = self.imgn[:N]
                self.x = self.x[:N]
                self.y = self.y[:N]
                self.frames_idx_from_eiger = self.frames_idx_from_eiger[:N]

    def load_data(self):
        self.h5meta = h5py.File(self.params['data'], 'r')
        h5prefix = 'entry/instrument/'

        if self.params['pixelsize'] is None:
            self.params['pixelsize'] = np.array(self.h5meta.get(h5prefix+'detector/x_pixel_size'))
            print("Pixelsize?", self.params['pixelsize'])

        self.params['detectordistance'] = np.array(self.h5meta.get(h5prefix+'detector/detector_distance'))

        if self.params['nrj'] is None:
            assert (self.h5meta.get(h5prefix+'beam/incident_wavelength')[()] > 0.1)
            # TODO: Load h,c to calculate nrj
            self.params['nrj'] = 12.3984 / self.h5meta.get(h5prefix+'beam/incident_wavelength')[()]

        if len(self.x) < 4:
            raise PtychoRunnerException("Less than 4 scan positions, is this a ptycho scan ?")
        # Spec values are in microns, convert to meters
        self.x *= 1e-6
        self.y *= 1e-6

        # Load all frames
        t0 = timeit.default_timer()

        self.print('Reading HDF5 frames: ')
        # frames are grouped in different subentries
        entry0 = 1
        ct = 0
        i0 = 0
        vimg = None
        
        while True:
            h5entry = self.params["h5data"] % entry0
            if h5entry not in self.h5meta:
                # The key does not exist in the master file at all.
                break
            
            # --- MODIFICATION: Handle Broken Links ---
            try:
                h5d = self.h5meta[h5entry]
                nb = len(h5d) # Access property to ensure file exists/is readable
            except Exception as e:
                print(f"Stopping data read at {h5entry}: Link exists but file cannot be accessed (end of data).")
                break
            # -----------------------------------------

            # Read nb frames, note that frames_idx_from_eiger is 1-based
            idx = self.frames_idx_from_eiger[i0:i0+nb]-1

            if len(idx):
                self.print("Reading h5 data entry: %s [%d frames]" % (h5entry, len(idx)))
                if vimg is None:
                    # Init array
                    frame = np.array(h5d[0])
                    vimg = np.empty((len(self.imgn), frame.shape[0], frame.shape[1]), dtype=frame.dtype)
                
                # Read data
                vimg[i0:i0+nb] = h5d[idx]
                
                ct += len(idx)

            entry0 += 1
            i0 += nb
            
        if ct != len(self.imgn):
            raise PtychoRunnerException("Could not read the expected number of frames: %d (read) < %d (scan points)" % (ct, len(self.imgn)))

        dt = timeit.default_timer() - t0
        print('Time to read all frames: %4.1fs [%5.2f Mpixel/s]' % (dt, vimg.size / 1e6 / dt))

        # Build mask, this is for dead pixels
        self.raw_mask = ((vimg > (2 ** 32 - 3)).sum(axis=0) > 0).astype(np.int8)
        if self.raw_mask.sum() == 0:
            self.raw_mask = None
        else:
            self.print("\nMASKING %d pixels from detector flags" % (self.raw_mask.sum()))
        # TODO: add mask for non-linear pixels

        self.raw_data = vimg
        self.load_data_post_process()


class PtychoRunnerTPS25A(PtychoRunner):
    """
    Class to process a series of scans with a series of algorithms, given from the command-line
    """

    def __init__(self, argv, params, *args, **kwargs):
        super(PtychoRunnerTPS25A, self).__init__(argv, params)
        self.PtychoRunnerScan = PtychoRunnerScanTPS25A

    @classmethod
    def make_parser(cls, default_par, description=None, script_name="pynx-ptycho-tps25a", epilog=None):
        if epilog is None:
            epilog = helptext_beamline
        if description is None:
            description = "Script to perform a ptychography analysis on data from 25A@TPS"
        p = default_par

        parser = super().make_parser(default_par, script_name, description, epilog)
        grp = parser.add_argument_group("TPS25A parameters")
        grp.add_argument('--scanfile', type=str, default=None,
                         help='path to text file with motor positions [mandatory]')
        grp.add_argument('--data', type=str, default=None,
                         help='path to hdf5 data file, with images, wavelength, detector distance [mandatory]')
        grp.add_argument('--h5data', type=str, default='entry/data/data_%06d',
                         help='relative path to data inside hdf5 file [default=entry/data/data_%%06d]')
        return parser

    def check_params_beamline(self):
        """
        Check if self.params includes a minimal set of valid parameters, specific to a beamiline
        Returns: Nothing. Will raise an exception if necessary
        """
        if self.params['scanfile'] is None:
            raise PtychoRunnerException('Missing argument: no scanfile given')
        if self.params['data'] is None:
            raise PtychoRunnerException('Missing argument: no data (hdf5 master) given')
