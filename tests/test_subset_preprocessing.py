"""Regression checks for information boundaries; no model training."""
import unittest
from types import SimpleNamespace
import numpy as np
import xarray as xr

from mhw.subset_preprocessing import (
    BASE_DYNAMIC_VARS, fit_subset_stats, selected_time_indices,
    apply_manifest_physics, make_raw_base, compute_subset_rhs_scale,
)


def synthetic():
    shape = (30, 3, 4)
    a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 100 + 1
    dims = ("time", "latitude", "longitude")
    ds = xr.Dataset({name: (dims, a.copy()) for name in [*BASE_DYNAMIC_VARS, "qnet"]},
                    coords={"time": np.arange(30), "latitude": [30., 31., 32.],
                            "longitude": [125., 126., 127., 128.]})
    ds["ocean_mask_static"] = (("latitude", "longitude"), np.ones((3, 4), np.float32))
    return ds


class SubsetStatisticsTests(unittest.TestCase):
    def test_exact_days_and_population_statistics(self):
        ds = synthetic()
        xdays, ydays = selected_time_indices([2, 6], 2, 1, 30)
        np.testing.assert_array_equal(xdays, [2, 3, 6, 7])
        np.testing.assert_array_equal(ydays, [4, 8])
        stats = fit_subset_stats(ds, [2, 6], 2, 1)
        for key, days in (("sst_mean", xdays), ("target", ydays)):
            raw = ds.sst_mean.values[days].astype(np.float64)
            np.testing.assert_allclose(stats[key], (raw.mean(), raw.std()), rtol=1e-12, atol=1e-12)

    def test_unselected_and_heldout_observations_do_not_affect_statistics(self):
        ds = synthetic()
        before = fit_subset_stats(ds, [2, 6], 2, 1)
        untouched = np.setdiff1d(np.arange(30), [2, 3, 4, 6, 7, 8])
        for name in BASE_DYNAMIC_VARS:
            ds[name].values[untouched] = 1e9
        self.assertEqual(before, fit_subset_stats(ds, [2, 6], 2, 1))

    def test_rhs_does_not_use_unselected_dates(self):
        ds = synthetic()
        base = make_raw_base(ds, history=2, lead=1)
        before = compute_subset_rhs_scale(base, [2, 6], 1, batch_size=2)
        for name in ds.data_vars:
            if "time" in ds[name].dims:
                ds[name].values[[0, 1, 5, 9, 20, 29]] = 1e8
        after = compute_subset_rhs_scale(make_raw_base(ds, 2, 1), [2, 6], 1, batch_size=2)
        self.assertEqual(before, after)

    def test_invalid_starts_rejected(self):
        for starts in ([], [2, 2], [6, 2], [-1, 2], [29], [2.5]):
            with self.assertRaises(ValueError):
                selected_time_indices(starts, 2, 1, 30)

    def test_full_scale_override_rejected(self):
        args = SimpleNamespace(history=30, lead=7, physics_mode="p0_original",
                               hm_min=5., entrainment_dt=1., physics_rhs_scale=9.7)
        m = {"history": 30, "lead": 7, "physics_rhs_scale": 2.0}
        with self.assertRaises(ValueError):
            apply_manifest_physics(args, SimpleNamespace(), m)
        args.physics_rhs_scale = None
        physics = SimpleNamespace()
        apply_manifest_physics(args, physics, m)
        self.assertEqual(physics.physics_rhs_scale, 2.0)
        self.assertFalse(args.include_residual_feature)


if __name__ == "__main__":
    unittest.main()
