# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property

import earthkit.data as ekd
import entrypoints

from .compute import make_rh_from_t_and_q, make_tcwv_from_q

LOG = logging.getLogger(__name__)


class FileInput:
    def __init__(self, owner, file, **kwargs):
        self.file = file
        self.owner = owner

    @cached_property
    def fields_sfc(self):
        return self.all_fields.sel(levtype="sfc")

    @cached_property
    def fields_pl(self):
        return self.all_fields.sel(levtype="pl")

    @cached_property
    def fields_ml(self):
        return self.all_fields.sel(levtype="ml")

    @cached_property
    def all_fields(self):
        ds = ekd.from_source("file", self.file)
        param = [p.lower() for p in ds.metadata("param")]
        assert isinstance(param, (list, tuple))
        if "tcwv" not in param and "q" in param:
            logging.warning("Parameter 'tcwv' on surface levels is not available, computing it from 'q' instead")
            ds = make_tcwv_from_q(ds)
        if "r" not in param and "q" in param:
            logging.warning("Parameter 'r' on pressure levels is not available, computing it from 't' and 'q' instead")
            ds = make_rh_from_t_and_q(ds)
        return ds


def get_input(name, *args, **kwargs):
    return available_inputs()[name].load()(*args, **kwargs)


def available_inputs():
    result = {}
    for e in entrypoints.get_group_all("ai_models.input"):
        result[e.name] = e
    return result
