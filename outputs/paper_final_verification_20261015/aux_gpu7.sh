#!/bin/bash
set -eu
cd /ceph_data/yeonwoo/MHW_subset
export CUDA_VISIBLE_DEVICES=7
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/runtime
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/aux_test.py 2 > /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/auxiliary_test/job2.log 2>&1
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/aux_test.py 5 > /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/auxiliary_test/job5.log 2>&1
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/aux_test.py 8 > /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/auxiliary_test/job8.log 2>&1
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/aux_test.py 11 > /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/auxiliary_test/job11.log 2>&1
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/aux_test.py 14 > /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/auxiliary_test/job14.log 2>&1
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/aux_test.py 17 > /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/auxiliary_test/job17.log 2>&1
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/aux_test.py 20 > /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/auxiliary_test/job20.log 2>&1
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/aux_test.py 23 > /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/auxiliary_test/job23.log 2>&1
