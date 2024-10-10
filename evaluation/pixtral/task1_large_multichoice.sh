#!/bin/bash -l
#$ -P multilm
#$ -l gpus=1
#$ -l gpu_type=A100
#$ -l gpu_memory=80G
#$ -N task1_large_multichoice

export HF_HOME=/projectnb/multilm/lsusanto/cuisine/HF_CACHE
python3 pixtral_evaluation_multichoice_task1_large.py
