#!/bin/bash
#SBATCH --job-name=cv-pilot
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=g2-standard-12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm-%j.log

# 从已经完成环境安装和数据准备的项目目录提交，事先 mkdir -p outputs。
# 分区与账户来自课程 Cloud Bursting 文档，不在 Torch 集群运行。
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
CV_PYTHON="${CV_PYTHON:-python}"
export TORCH_HOME="$PWD/data/torch-cache"
export OMP_NUM_THREADS=4

nvidia-smi
"$CV_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
# 短试跑重排时使用新的 attempt 目录，不覆盖先前成果；不假装恢复正式训练。
CV_OUTPUT="outputs/gpu-pilot-${SLURM_JOB_ID}-attempt-${SLURM_RESTART_COUNT:-0}"
"$CV_PYTHON" -u scripts/gpu_pilot.py --output "$CV_OUTPUT"
