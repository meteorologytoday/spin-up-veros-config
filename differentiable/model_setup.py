# Construction of a Veros-only setup

import glob
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import dataclasses

import xarray as xr

from jem import Coupler
from jem.components import (
    VerosComponent,
)
from jem.components.veros_component import GHOST_CELLS, SEAWATER_HEAT_CAPACITY
from veros.core.operators import update, at

from global_4deg_no_cycle import GlobalFourDegreeSetup 

def is_pytree_all_finite(tree):
    return jax.tree_util.tree_reduce(
        lambda acc, x: acc & jnp.all(jnp.isfinite(x)),
        tree,
        jnp.array(True),
    )

def report_first_nonfinite(name, x, lon=None, lat=None):
    """If `x` has any non-finite value, debug-print the index of its
    first occurrence (and lon/lat, if `x` is a 2D (lon, lat) field).

    `jax.debug.print` cannot format a tuple-of-tracers (e.g. the result
    of `jnp.unravel_index`) directly, so each index component is passed
    as its own scalar kwarg. The format string is built in Python from
    `x.ndim`, which is static under tracing.
    """
    x = jnp.asarray(x)
    finite = jnp.isfinite(x)
    any_bad = jnp.any(~finite)
    flat_idx = jnp.argmin(finite.ravel().astype(jnp.int32))
    idx = jnp.unravel_index(flat_idx, x.shape)
    val = x.ravel()[flat_idx]

    idx_fmt = ", ".join(f"{{i{d}}}" for d in range(x.ndim))
    idx_kwargs = {f"i{d}": idx[d] for d in range(x.ndim)}

    def report(_):
        fmt = name + f": first non-finite at idx=({idx_fmt}), value={{val:.6e}}"
        kwargs = dict(idx_kwargs, val=val)
        if lon is not None and lat is not None and x.ndim == 2:
            fmt += ", lon={lo:.2f}, lat={la:.2f}"
            # `lon`/`lat` are plain numpy arrays (since `jnp.pi` is a
            # Python float, `numpy_array * 180 / jnp.pi` stays numpy).
            # Indexing a numpy array with a JAX tracer raises
            # TracerArrayConversionError, so convert to jnp first.
            kwargs["lo"] = jnp.asarray(lon)[idx[0]]
            kwargs["la"] = jnp.asarray(lat)[idx[1]]
        jax.debug.print(fmt, **kwargs)

    jax.lax.cond(any_bad, report, lambda _: None, None)


def build_model(
    *,
    start_datetime,
    coupling_timestep,
    debug_mode=False,
):
    """Build the Veros-only system

    Returns the assembled `Coupler` (which owns the workflow and the clock)
    together with a `config` dict of supporting objects, so that callers can
    build exactly the same model. The clock is `jax_datetime`'s proleptic
    Gregorian calendar.

    """

    ocn_model = GlobalFourDegreeSetup()
    ocn_model.setup()
    ocn_component = VerosComponent(ocn_model)

    def interaction(components, time):
        ocn = components["ocn"]
        updated_components = dict(components, ocn=ocn)

        if debug_mode:

            print("Notice: debug_mode is on.")

            # Debug: report wind speed (atm),  velocity, and stratification (ocn) extremes
            vs_ocn = ocn["state"].variables
            jax.debug.print(
                "max|wind_u| (atm) = {wu:.6e}, max|wind_v| (atm) = {wv:.6e}, "
                "max|u| (ocn) = {ou:.6e}, max|v| (ocn) = {ov:.6e}, min(Nsqr) (ocn) = {nsqr:.6e}",
                wu=jnp.max(jnp.abs(wind_x)),
                wv=jnp.max(jnp.abs(wind_y)),
                ou=jnp.max(jnp.abs(vs_ocn.u[..., vs_ocn.tau])),
                ov=jnp.max(jnp.abs(vs_ocn.v[..., vs_ocn.tau])),
                nsqr=jnp.min(vs_ocn.Nsqr[..., vs_ocn.tau]),
            )

            # Debug model explosion
            def breakpoint_if_nonfinite(x):
                all_finite = is_pytree_all_finite(x)
                def true_fn(x):
                    pass
                def false_fn(x):
                    jax.debug.print("Model exploded. Activate breakpoint to see what happened.")
                    report_first_nonfinite("ocn.state.temp", vs_ocn.temp[..., vs_ocn.tau])
                    report_first_nonfinite("ocn.state.salt", vs_ocn.salt[..., vs_ocn.tau])
                    report_first_nonfinite("ocn.state.u", vs_ocn.u[..., vs_ocn.tau])
                    report_first_nonfinite("ocn.state.v", vs_ocn.v[..., vs_ocn.tau])
                    report_first_nonfinite("ocn.state.Nsqr", vs_ocn.Nsqr[..., vs_ocn.tau])
                jax.lax.cond(all_finite, true_fn, false_fn, x)

            breakpoint_if_nonfinite(updated_components) 

        return updated_components

    # One coupled step: exchange first, then every component integrates the
    # same exchanged state over `coupling_timestep`.
    model = Coupler(
        components=dict(
            ocn=ocn_component,

        ),
        exchangers=dict(mapper=interaction),
        coupling_timestep=coupling_timestep,
        start_date=start_datetime,
        workflow=["mapper", "ocn"],
    )

    return model
