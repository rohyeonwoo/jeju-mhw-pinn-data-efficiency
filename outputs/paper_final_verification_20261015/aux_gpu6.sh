#!/bin/bash
set -eu
cd /ceph_data/yeonwoo/MHW_subset
export CUDA_VISIBLE_DEVICES=6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/runtime
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/aux_test.py 1 > /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/auxiliary_test/job1.log 2>&1
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/aux_test.py 4 > /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/auxiliary_test/job4.log 2>&1
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/aux_test.py 7 > /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/auxiliary_test/job7.log 2>&1
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/aux_test.py 10 > /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/auxiliary_test/job10.log 2>&1
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/aux_test.py 13 > /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/auxiliary_test/job13.log 2>&1
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/aux_test.py 16 > /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/auxiliary_test/job16.log 2>&1
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/aux_test.py 19 > /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/auxiliary_test/job19.log 2>&1
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/aux_test.py 22 > /ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/auxiliary_test/job22.log 2>&1
