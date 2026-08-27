# PyNX Ptychography @ TPS 25A

Working files for running [PyNX](https://gitlab.esrf.fr/favre/PyNX) ptychography
reconstructions and its PyQt5 GUI, customized for the TPS 25A beamline setup.

## Files

- **`pynx_at_tps25a/main.py`** — Main toolkit: data loading helpers (`get_exp_path`,
  `load_exp_condition`, `pynx_run`, `pynx_start`), manual data injection
  (`pynx_set_data`), plotting (`pynx_plot_overview`, `pynx_plot_obj`, `pynx_plot_probe`),
  and the `simple_gui()` PyQt5 control panel.
- **`pynx/ptycho/runner/tps25a.py`** — Unmodified copy of the PyNX beamline runner
  (`PtychoRunnerScanTPS25A`) that `main.py` builds on, included for reference/reproducibility.
- **`pynx_at_tps25a/load_ws_for_gui.py`** — Example script that builds a `ws` workspace via
  `pynx_set_data()` directly from a pre-cropped `.npy` array, a dead-pixel mask, and a scan
  position CSV, bypassing the `scanID`/file-search data loading path. Useful when the raw
  data folder doesn't (yet) have the `primary`/`baseline`/`data` files and master-file
  metadata that `get_exp_path()` / `load_exp_condition()` expect.

  **Note**: this script has hardcoded paths and experimental parameters (pixel size,
  detector distance, energy, wavelength) for one specific local test dataset — treat it as
  a template to copy and adapt, not a portable script.
- **`scan_tools/fermat_spiral.py`** — Fermat's spiral point pattern generator. Supports Bluesky
  dual-arm / single-arm modes, continuous trajectory, and golden angle sampling with square boundary cropping.
- **`scan_tools/generate_fermat_scan_cmd.py`** — Generates beamline motor control (`SFL`, `RES`) and
  detector trigger (`TRG`) command sequences (`STR` ... `END`) from Fermat spiral scan trajectories.
- **`scan_tools/fermat_scan_commands.txt`** — Example beamline command file generated for XZ motor scanning.

## Fixed bug: GUI deadlock on Run DM / Run ML / Run All

The GUI's `Run DM`, `Run ML`, and `Run All` buttons used to run the reconstruction
algorithm in a background `threading.Thread`, manually pushing/popping a PyCUDA context
(`cu_ctx.push()` / `.pop()`) that had been created on the main thread. CUDA contexts are
per-thread: pushing a context that is still current on another thread is not supported by
the driver's per-thread context stack, and reliably deadlocks (GPU usage drops to 0%,
IPython becomes unresponsive).

Fix: run the algorithm synchronously on the main (GUI) thread, same as the other buttons
(`Set To ws`, `pynx_init`, the plot buttons). Tradeoff: the GUI window is unresponsive for
the duration of the run (no live-updating during the algorithm), which is preferable to a
silent deadlock.

## Basic usage

```powershell
# Activate the PyNX venv, then:
ipython

# In IPython (use -i so both scripts share the interactive namespace):
%run -i main.py
%run -i pynx_at_tps25a/load_ws_for_gui.py   # or your own data-loading code, must define `ws`
simple_gui()
```

In the GUI: click **Set To ws** first (not Run Start / Run All, which need the
`scanID` + `primary`/`baseline`/`data` file layout) to pick up the `ws` you created in
IPython, adjust parameters, then use **Run DM** / **Run ML** / the algorithm-string based
**Run** button, **Plot Overview/Obj/Probe**, and **Save Obj Probe** as needed.
