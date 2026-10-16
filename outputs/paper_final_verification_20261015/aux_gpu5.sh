#!/bin/bash
set -eu
cd /ceph_data/yeonwoo/MHW_subset
export CUDA_VISIBLE_DEVICES=5
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/runtime
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/aux_test.py 0 > /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/auxiliary_test/job0.log 2>&1
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/aux_test.py 3 > /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/auxiliary_test/job3.log 2>&1
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/aux_test.py 6 > /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/auxiliary_test/job6.log 2>&1
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/aux_test.py 9 > /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/auxiliary_test/job9.log 2>&1
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/aux_test.py 12 > /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/auxiliary_test/job12.log 2>&1
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/aux_test.py 15 > /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/auxiliary_test/job15.log 2>&1
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/aux_test.py 18 > /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/auxiliary_test/job18.log 2>&1
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/aux_test.py 21 > /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/auxiliary_test/job21.log 2>&1
