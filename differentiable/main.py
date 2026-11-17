# Couple JCM and Veros using JAX-ESM (JEM).

# Must be imported before h5py (pulled in by veros/jem): the pip wheels of
# h5py and netCDF4 each bundle their own HDF5, and if h5py's loads first,
# every netCDF4 write fails with "RuntimeError: NetCDF: HDF error".
import netCDF4  # noqa: F401

import json
import sys
from pathlib import Path

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)  # only cache if compile took >1s

#jax.config.update("jax_enable_x64", False) 
import jax_datetime as jdt
import jem

from model_setup import build_model

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--total-simulation-days", type=int, help="Total time of simulation in days", default=10)
parser.add_argument("--simulation-interval-days", type=int, help="Simulation interval in days", default=73)
parser.add_argument("--simulation-name", type=str, help="Simulation name for output", default="default")
parser.add_argument("--debug-mode", action="store_true", help="Turn on debug mode. Detect NaN and enter breakpoint.")
parser.add_argument("--do-not-average-time", action="store_true", help="Write instantaneous output instead of chunk averages.")
parser.add_argument("--checkpoint-interval",type=int, help="Save a checkpoint every this many simulation intervals.", default=50)
args = parser.parse_args()

print(f"jem library is located at: {jem.__file__}")

# Check available devices
print(f"Available devices: {jax.devices()}")
print(f"Number of devices: {len(jax.devices())}")

# Configurations
start_datetime = jdt.to_datetime("2000-01-01")
# Must be a whole multiple of Veros' dt_tracer (86400 s in global_4deg_no_cycle.py).
coupling_timestep = jdt.to_timedelta(86400, "second")

output_dir = (Path(f"output") / args.simulation_name).resolve()
output_dir.mkdir(exist_ok=True, parents=True)
# The run's options, for the diagnostics (e.g. diagnostics/mld.py reads the
# equation of state from it); rewritten at every (re)start.
(output_dir / "run_args.json").write_text(json.dumps(vars(args), indent=2) + "\n")

# Build the coupled JCM + Veros + fake-land system. Packaged as a function
# in `model_setup.py` so that other scripts (e.g. a jax.grad sensitivity
# experiment) can build exactly the same model.
model = build_model(
    start_datetime=start_datetime,
    coupling_timestep=coupling_timestep,
    debug_mode=args.debug_mode,
)

print(f"Model info: {model!r}")

# Run Coupled Model
#
# `jem.run_chunked` owns the loop: it integrates one simulation interval
# ("chunk") at a time, writes `<component>-<first coupled step>.nc` for each,
# runs JCM's blow-up diagnostics on the chunk, and keeps a single restart
# directory at `output_dir / "checkpoint"`, rewritten at each checkpoint
# interval. Re-running this same command resumes from that checkpoint. A
# chunk that fails the health check is still written (for debugging) but not
# checkpointed, and the run stops there.
result = jem.run_chunked(
    model,
    total_time=f"{args.total_simulation_days:d} days",
    chunk=f"{args.simulation_interval_days:d} days",
    output_dir=output_dir,
    output_averages=not args.do_not_average_time,
    checkpoint_interval=f"{args.checkpoint_interval * args.simulation_interval_days:d} days",
)

if not result.completed:
    print(f"Error: the health check stopped the run at coupled step {result.steps_completed:d}.")
    print(f"Last report: {result.reports[-1] if result.reports else '(none)'}")
    print("Exit program.")
    sys.exit(1)

print(f"Program ends.")
