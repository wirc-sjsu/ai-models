# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import earthkit.data as ekd
import numpy as np
import tqdm
from earthkit.data.core.temporary import temp_file
from earthkit.data.indexing.fieldlist import FieldArray

LOG = logging.getLogger(__name__)

G = 9.80665  # Same a pgen


def make_z_from_gh(ds):

    tmp = temp_file()

    out = ekd.new_grib_output(tmp.path)
    other = []

    for f in tqdm.tqdm(ds, delay=0.5, desc="GH to Z", leave=False):

        if f.metadata("param") == "gh":
            out.write(f.to_numpy() * G, template=f, param="z")
        else:
            other.append(f)

    out.close()

    result = FieldArray(other) + ekd.from_source("file", tmp.path)
    result._tmp = tmp

    return result

def compute_tcwv(q_levels, p_levels):
    """Compute Total Column Water Vapor (tcwv) from specific humidity and pressure levels."""
    dp = np.diff(p_levels, prepend=p_levels[0])  # Pressure differences (Pa)
    tcwv = np.sum(q_levels * dp[:, np.newaxis, np.newaxis], axis=0) / G
    return tcwv

def make_tcwv_from_q(ds):
    """Compute TCWV and add it to an Earthkit GRIB dataset."""
    
    tmp = temp_file()  # Temporary output file
    out = ekd.new_grib_output(tmp.path)

    # Extract Specific Humidity (q)
    q_data = {f.metadata("level"): f for f in ds if f.metadata("param") == "q"}
    available_levels = sorted(set(q_data.keys()))
    p_levels = np.array(available_levels) * 100  # Convert hPa to Pa

    if len(available_levels):
        q_values = np.array([q_data[level].to_numpy() for level in available_levels])
        tcwv_value = compute_tcwv(q_values, p_levels)
        out.write(tcwv_value, template=next(iter(q_data.values())), param="tcwv")

    out.close()

    result = ds + ekd.from_source("file", tmp.path)
    result._tmp = tmp  # Keep reference to temporary file

    return result

def compute_rh(q, T, P):
    """Compute Relative Humidity (RH) from specific humidity, temperature, and pressure."""
    e = (q * P) / (0.622 + q)  # Actual vapor pressure (hPa)
    es = 6.112 * np.exp((17.67 * T) / (T + 243.5))  # Saturation vapor pressure (hPa)
    rh = (e / es) * 100  # Convert to percentage
    return np.clip(rh, 0, 100)  # Ensure RH is between 0-100%

def make_rh_from_t_and_q(ds):
    """Compute RH at each level and add it to an Earthkit GRIB dataset."""
    
    tmp = temp_file()  # Temporary output file
    out = ekd.new_grib_output(tmp.path)

    # Get available pressure levels
    available_levels = sorted(set(ds.metadata("level")))

    # Extract Specific Humidity (q) and Temperature (T)
    q_data = {f.metadata("level"): f for f in ds if f.metadata("param") == "q"}
    t_data = {f.metadata("level"): f for f in ds if f.metadata("param") == "t"}

    # Compute RH at each level
    for level in tqdm.tqdm(available_levels, delay=0.5, desc="Computing RH", leave=False):
        if level in q_data and level in t_data:
            q_level = q_data[level].to_numpy()
            t_level = t_data[level].to_numpy()
            rh_value = compute_rh(q_level, t_level, level)
            out.write(rh_value, template=q_data[level], param="r")

    out.close()

    result = ds + ekd.from_source("file", tmp.path)
    result._tmp = tmp  # Keep reference to temporary file

    return result
