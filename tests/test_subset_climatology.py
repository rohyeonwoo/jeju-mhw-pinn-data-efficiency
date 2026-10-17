import unittest
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from mhw.subset_climatology import calendar_bins,fit_climatology,anomaly
from train_jeju_hilam import CyclingTrainBatches,enforce_execution_rules

class ClimatologyTests(unittest.TestCase):
    def test_calendar_alignment(self):
        dates=np.array(['1999-02-28','2000-02-28','2000-02-29','1999-03-01','2000-03-01'],dtype='datetime64[D]')
        np.testing.assert_array_equal(calendar_bins(dates),[58,58,59,60,60])

    def test_empty_feb29_and_year_boundary(self):
        bins=np.array([58,60,365,1])
        raw=np.array([2.,6.,10.,14.],np.float32)[:,None,None]
        clim,counts=fit_climatology(raw,bins,np.arange(4),np.ones((1,1),bool))
        self.assertEqual(clim[59,0,0],4.)
        self.assertEqual(clim[0,0,0],12.)
        self.assertEqual(counts[59,0,0],0)

    def test_observed_feb29_used(self):
        bins=np.array([58,59,59,60]);raw=np.array([2,5,7,8],np.float32)[:,None,None]
        clim,count=fit_climatology(raw,bins,np.arange(4),np.ones((1,1),bool))
        self.assertEqual(clim[59,0,0],6.)
        self.assertEqual(count[59,0,0],2)

    def test_unselected_values_cannot_change_fit(self):
        bins=np.arange(12); raw=np.arange(24,dtype=np.float32).reshape(12,1,2)
        mask=np.ones((1,2),bool);idx=np.array([1,5,9])
        clim,count=fit_climatology(raw,bins,idx,mask)
        changed=raw.copy();changed[np.setdiff1d(np.arange(12),idx)]=1e9
        other,n=fit_climatology(changed,bins,idx,mask)
        np.testing.assert_array_equal(clim,other)
        np.testing.assert_array_equal(anomaly(raw[idx],bins[idx],clim,mask),anomaly(changed[idx],bins[idx],other,mask))

    def test_land_excluded_and_missing_ocean_fail_closed(self):
        raw=np.array([[[1,np.nan]],[[3,1e9]],[[5,np.nan]]],np.float32)
        mask=np.array([[True,False]]);bins=np.array([1,3,5])
        clim,count=fit_climatology(raw,bins,np.arange(3),mask)
        self.assertTrue((clim[:,:,1]==0).all())
        self.assertTrue((anomaly(raw,bins,clim,mask)[:,:,1]==0).all())
        raw[0,0,0]=np.nan
        fit_climatology(raw,bins,np.arange(3),mask) # finite selected observations still fit
        with self.assertRaises(ValueError):anomaly(raw,bins,clim,mask)
        raw[:,0,0]=np.nan
        with self.assertRaises(ValueError):fit_climatology(raw,bins,np.arange(3),mask)

    def test_virtual_epoch_keeps_position(self):
        stream=CyclingTrainBatches([0,1,2])
        self.assertEqual(list(stream.take(2)),[0,1])
        self.assertEqual(list(stream.take(5)),[2,0,1,2,0])

    def test_output_path_guard(self):
        args=SimpleNamespace(output_root=Path('/tmp/not-allowed'),virtual_epoch_steps=None,max_train_batches=None)
        with self.assertRaises(ValueError):enforce_execution_rules(args)
        args.output_root=Path('/ceph_data/yeonwoo/MHW_subset/outputs')
        enforce_execution_rules(args)

if __name__=='__main__':unittest.main()
