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
from veros_case_setup import generateVerosSetup
from veros.core.operators import update, at

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

    ocn_model = generateVerosSetup(
        rotated_grid=False,
        scrip_grid_file = grid_folder / f"{ocean_grid:s}.SCRIP.nc",
        landsea_mask_file = grid_folder / f"ocn_topo_{ocean_grid:s}.nc",
        dt_mom=veros_dt_mom,
        dt_tracer=veros_dt_tracer,
        # 30-layer geometric-stretch grid, same 5200 m total depth as the
        # previous hand-tuned 15-layer profile. `_surface_thickness`
        # is exposed as its own parameter -- a 10 m surface layer (5x
        # smaller heat capacity for the same heat flux) used to trigger an
        # immediate atm humidity explosion (~day 2) regardless of 
        # timestep (tested 3600/1800/900s, all failed identically). The
        # suspected cause was that the *entire* absorbed solar flux was
        # being dumped into that one top layer, making the daytime heating
        # rate scale as 1/`_surface_thickness`; see
        # `enable_shortwave_penetration` and
        # `_shortwave_penetration_divergence`, which spread it over the
        # top ~`shortwave_efold2` metres instead.
        ddz=ocean_ddz,
        cold_start_ocean_temperature_reference_K=ocean_initial_surface_temperature,
        initial_temp_salt=initial_temp_salt,
        eq_of_state_type=ocean_eq_of_state_type,
    )()
    ocn_model.setup()
    ocn_component = VerosComponent(ocn_model)

    def interaction(components, time):
        ocn = components["ocn"]

        ocean_heat_flux = atm["derived"].total_heat_flux
        freshwater_flux = atm["derived"].total_freshwater_flux

        # W/m^2 * 1/m / (J/kg/K) / (kg/m^3) = K/s. `tempsalt_sources`
        # multiplies by `maskT` itself, so no mask is needed here.
        vs_ocn_forcing = ocn["state"].variables
        interior = slice(GHOST_CELLS, -GHOST_CELLS)
        with vs_ocn_forcing.unlock():
            vs_ocn_forcing.temp_source = update(
                vs_ocn_forcing.temp_source,
                at[interior, interior, :],
                solar_flux[:, :, None]
                * divpen_shortwave[None, None, :]
                / ocn_cp_0
                / ocn_rho_0,
            )

        ocn_forcing = ocn["forcing"]
        ocn = dict(ocn, forcing=ocn_forcing.replace(
            surface_taux=surface_taux,
            surface_tauy=surface_tauy,
            heat_flux=seaice["derived"].ocean_heat_flux_up,
            sea_ice_freshwater_flux=-seaice["derived"].ocean_freshwater_flux_up,
            runoff=lnd["derived"].runoff_to_ocean.astype(ocn_forcing.runoff.dtype),
        ))
 
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
                    report_first_nonfinite("wind_x", wind_x, lon_deg, lat_deg)
                    report_first_nonfinite("wind_y", wind_y, lon_deg, lat_deg)
                    report_first_nonfinite("ocn.forcing.heat_flux", ocn["forcing"].heat_flux, lon_deg, lat_deg)
                    report_first_nonfinite("atm.forcing.stl_am", atm["forcing"].stl_am, lon_deg, lat_deg)
                    # The crash was first seen in the  state (u, v, Nsqr
                    # all NaN while the atmosphere was still finite), so check
                    # the 's prognostic fields too (3D: x, y, depth).
                    report_first_nonfinite("ocn.state.temp", vs_ocn.temp[..., vs_ocn.tau])
                    report_first_nonfinite("ocn.state.salt", vs_ocn.salt[..., vs_ocn.tau])
                    report_first_nonfinite("ocn.state.u", vs_ocn.u[..., vs_ocn.tau])
                    report_first_nonfinite("ocn.state.v", vs_ocn.v[..., vs_ocn.tau])
                    report_first_nonfinite("ocn.state.Nsqr", vs_ocn.Nsqr[..., vs_ocn.tau])
                    # jax.debug.breakpoint()  # disabled: no TTY in SLURM batch jobs -> JaxRuntimeError
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
