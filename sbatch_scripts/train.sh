#!/bin/bash
#SBATCH --job-name=lm
#
#SBATCH -A eqb@v100
#
##SBATCH -C v100-16g                 # uncomment to target only 16GB V100 GPU
##SBATCH -C v100-32g                 # uncomment to target only 32GB V100 GPU
##SBATCH --partition=gpu_p2          # uncomment for gpu_p2 partition (32GB V100 GPU)
##SBATCH --partition=gpu_p4          # uncomment for gpu_p4 partition (40GB A100 GPU)
##SBATCH -C a100                     # uncomment for gpu_p5 partition (80GB A100 GPU)
# Here, reservation of 10 CPUs (for 1 task) and 1 GPU on a single node:
#SBATCH --nodes=1                    # we request one node
#SBATCH --ntasks-per-node=1          # with one task per node (= number of GPUs here)
#SBATCH --gres=gpu:1                 # number of GPUs per node (max 8 with gpu_p2, gpu_p4, gpu_p5)
# The number of CPUs per task must be adapted according to the partition used. Knowing that here
# only one GPU is reserved (i.e. 1/4 or 1/8 of the GPUs of the node depending on the partition),
# the ideal is to reserve 1/4 or 1/8 of the CPUs of the node for the single task:
#SBATCH --cpus-per-task=10           # number of cores per task (1/4 of the 4-GPUs node)
##SBATCH --cpus-per-task=3           # number of cores per task for gpu_p2 (1/8 of 8-GPUs node)
##SBATCH --cpus-per-task=6           # number of cores per task for gpu_p4 (1/8 of 8-GPUs node)
##SBATCH --cpus-per-task=8           # number of cores per task for gpu_p5 (1/8 of 8-GPUs node)
#
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --time=20:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=out/train_%j.out
#SBATCH --error=out/train_%j.out
 
# Cleans out the modules loaded in interactive and inherited by default 
module purge
 
# Uncomment the following module command if you are using the "gpu_p5" partition
# to have access to the modules compatible with this partition.
#module load cpuarch/amd
 
# Loading of modules
module load python
conda activate lm_feedback
 
# Echo of launched commands
set -x

# Code execution
vocab_size=32000
max_len=128

# Debugging:
# export CUDA_LAUNCH_BLOCKING=1

python -u main.py fit --model.vocab_size $vocab_size --data.vocab_size $vocab_size --model.max_len $max_len --data.max_len $max_len --data.batch_size 128 --model.initial_lr 0.00001 --data.training_track babylm_100M 

#--trainer.val_check_interval 5000
