"""Subset-only month/day climatology and v2 anomaly dataset loading."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from . import subset_preprocessing as v1

SCHEMA = 'subset_normalization_v2'
V3_SCHEMA = 'subset_normalization_v3'
ANOMALIES = ['sst_anom', 'thetao_z0m_anom', 'thetao_z16m_anom', 'thetao_z34m_anom',
             'thetao_z78m_anom', 'thetao_z110m_anom', 'thetao_z156m_anom']
DEPTH_INDICES = [0, 10, 15, 20, 22, 24]
RULE = '366 month/day bins aligned to leap year 2000; selected finite ocean means; empty bins periodic linear interpolation of nearest supported bins; no Full fallback'


def calendar_bins(time):
    """0-based leap-aligned month/day, so March 1 is always bin 60."""
    dates = pd.DatetimeIndex(time)
    offsets = np.array([0,31,60,91,121,152,182,213,244,274,305,335])
    return offsets[dates.month.to_numpy()-1] + dates.day.to_numpy()-1


def fit_climatology(raw, bins, selected, ocean):
    """Only raw[selected] is inspected. Land is excluded and stored as zero."""
    selected = np.asarray(selected, dtype=np.int64)
    if selected.size == 0 or np.any(np.diff(selected) <= 0):
        raise ValueError('Expected sorted unique input-history dates')
    values = raw[selected][:, ocean].astype(np.float64)
    if np.isinf(values).any():
        raise ValueError('Infinite selected raw ocean values')
    sb = bins[selected]
    count = np.zeros((366, values.shape[1]), dtype=np.int32)
    means = np.full(count.shape, np.nan, dtype=np.float64)
    for d in np.unique(sb):
        v = values[sb == d]
        finite = np.isfinite(v)
        count[d] = finite.sum(axis=0)
        total = np.where(finite, v, 0.).sum(axis=0)
        means[d] = np.divide(total, count[d], out=np.full(v.shape[1], np.nan), where=count[d]>0)
    for cell in range(values.shape[1]):
        supported = np.flatnonzero(count[:,cell])
        if len(supported) < 2:
            raise ValueError('Fewer than two supported calendar bins for an ocean cell')
        # Periodic, convex linear interpolation: year boundary is not an edge.
        means[:,cell] = np.interp(np.arange(366), supported, means[supported,cell], period=366)
    clim = np.zeros((366, *ocean.shape), dtype=np.float32)
    valid_counts = np.zeros(clim.shape, dtype=np.int32)
    clim[:,ocean] = means.astype(np.float32)
    valid_counts[:,ocean] = count
    if not np.isfinite(clim).all():
        raise ValueError('Nonfinite climatology after subset-only interpolation')
    return clim, valid_counts


def anomaly(raw, bins, clim, ocean):
    """Land is a neutral zero; missing ocean observations are never silently filled."""
    result = np.zeros(raw.shape, dtype=np.float32)
    selected = raw[:,ocean]
    if not np.isfinite(selected).all():
        raise ValueError('Nonfinite raw ocean observation; cannot silently impute an anomaly')
    result[:,ocean] = selected - clim[bins][:,ocean]
    return result


def load_raw_anomaly_fields(ds, precursor_path):
    """Read SST and the same six original thetao depths; no old anomalies used."""
    raw = {'sst_anom': ds['sst_mean'].values.astype(np.float32)}
    with xr.open_dataset(precursor_path) as precursor:
        for dim in ('time','latitude','longitude'):
            np.testing.assert_array_equal(precursor[dim].values, ds[dim].values)
        depths = precursor.depth.values
        for name, index in zip(ANOMALIES[1:], DEPTH_INDICES):
            expected = int(name.split('_z')[1].split('m')[0])
            if int(round(float(depths[index]))) != expected:
                raise ValueError(f'Unexpected raw depth for {name}')
            raw[name] = precursor.thetao.isel(depth=index).values.astype(np.float32)
    return raw


def apply_to_base(base, raw_fields, bins, clims):
    """A view replaces only input anomalies. SST targets/physics stay raw."""
    out = copy.copy(base)
    out.dynamic_arrays = dict(base.dynamic_arrays)
    ocean = base.ocean_mask.astype(bool)
    for name in ANOMALIES:
        out.dynamic_arrays[name] = anomaly(raw_fields[name], bins, clims[name], ocean)
    return out


def fit_anomaly_stats(base, selected):
    ocean = base.ocean_mask.astype(bool)
    return {name:v1._mean_std(base.dynamic_arrays[name][selected][:,ocean]) for name in ANOMALIES}


def read_climatology(manifest, manifest_path):
    p = Path(manifest_path).parent / manifest['climatology_file']
    if v1.sha256_file(p) != manifest['climatology_sha256']:
        raise ValueError('Climatology artifact hash mismatch')
    with np.load(p, allow_pickle=False) as archive:
        if not np.array_equal(archive['input_time_indices'],manifest['input_time_indices']):
            raise ValueError('Climatology fitting dates differ from manifest')
        return {name:archive[name].copy() for name in ANOMALIES}


def bundle_from_v2(ds, manifest, base, raw_fields, clims):
    if manifest.get('schema') not in (SCHEMA, V3_SCHEMA) or manifest['climatology_rule'] != RULE:
        raise ValueError('Unsupported subset climatology definition')
    rebased = apply_to_base(base, raw_fields, calendar_bins(ds.time.values), clims)
    compat = dict(manifest, schema=v1.SCHEMA)
    bundle = v1.bundle_from_manifest(ds, compat, base=rebased)
    return bundle


def load_v2_bundle(manifest_path, data_path=None):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    path = Path(data_path or manifest['data_path'])
    precursor = Path(manifest['raw_thetao_path'])
    for p,key in ((path,'data_sha256'),(precursor,'raw_thetao_sha256')):
        if v1.sha256_file(p) != manifest[key]:
            raise ValueError(f'Raw source hash mismatch: {p}')
    ds = xr.open_dataset(path)
    try:
        clims = read_climatology(manifest, manifest_path)
        raw = load_raw_anomaly_fields(ds, precursor)
        return bundle_from_v2(ds, manifest, v1.make_raw_base(ds,manifest['history'],manifest['lead']),raw,clims), manifest
    except Exception:
        ds.close()
        raise
