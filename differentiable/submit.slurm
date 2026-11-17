#!/bin/bash
#
#SBATCH --job-name=JEM-exp-freelst-freeswa
#SBATCH --nodes 1
#SBATCH --tasks 1
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --gres=gpu:1
#SBATCH --output=sbatch.64.%j.out
#SBATCH --mail-type=ALL,TIME_LIMIT
#SBATCH --time=24:00:00

nvidia-smi

echo "pwd: $(pwd)"
eval "$( /home/tienyiao/miniconda3/bin/conda shell.bash hook 2> /dev/null)"
conda activate jax

source /home/tienyiao/.bashrc_jem
echo "PYTHONPATH=$PYTHONPATH"

export PYTHONHASHSEED=0

#export XLA_FLAGS="--xla_gpu_deterministic_ops=true"

# Resubmitting this same script resumes the run from its checkpoint
# (output_T31/<simulation-name>/checkpoint).
# Checkpoints written before JEM split WintonForcing's precipitation (f759ae4)
# need one extra conversion; it leaves an already-converted one alone.
#ckpt=output_T31/exp_freelst_freeswa/checkpoint/carry.msgpack
#[ -f $ckpt ] && python3 tools/upgrade_winton_checkpoint.py $ckpt

time python3 main.py                                \
    --total-simulation-days $(( 360 ))              \
    --simulation-interval-days 30                   \
    --checkpoint-interval 120                       \
    --simulation-name test                          
