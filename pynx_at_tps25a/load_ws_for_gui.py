"""
Prepares a `ws` workspace using pynx_set_data() (bypasses the scanID/primary/baseline
file-search route, since C:\\Users\\User\\Desktop\\AllenCheng\\data doesn't have those
files yet). Run this with %run inside the SAME IPython session before calling simple_gui(),
so `ws` exists as a global variable the GUI's "Set To ws" button can pick up.
"""
import hdf5plugin  # noqa: F401 (register compression filters before any h5py use)
import numpy as np
import pandas as pd

base = r"C:\Users\User\Desktop\AllenCheng\data"

data = np.load(base + r"\cropped_ptycho_9x500x500.npy").astype(np.float64)
mask = np.load(base + r"\cropped_dead_pixel_mask_500x500.npy")  # True = dead pixel, excluded

pos = pd.read_csv(base + r"\scan_positions.csv")
scan_x = -pos['rel_x'].to_numpy(dtype=np.float64)  # um, negated (motor readback vs sample-relative direction)
scan_z = -pos['rel_z'].to_numpy(dtype=np.float64)  # um, negated

pixelsize = 7.5e-5          # m
detectordistance = 1.41     # m  (corrected)
energy_keV = 9.67           # keV (corrected)
wavelength = 12.3984 / energy_keV * 1e-10   # m

ws = pynx_set_data(
    data=data,
    mask=mask,
    scan_x=scan_x,
    scan_z=scan_z,
    pixelsize=pixelsize,
    detectordistance=detectordistance,
    wavelength=wavelength,
    nbprobe=3,
)
print("\n[ready] `ws` created via pynx_set_data(). Now call simple_gui(), then click 'Set To ws'.")
