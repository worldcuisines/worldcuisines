#!/bin/bash -l
#$ -P multilm
#$ -l gpus=1
#$ -l gpu_type=A100
#$ -l gpu_memory=80G
#$ -N task2_large_multichoice

export HF_HOME=/projectnb/multilm/lsusanto/cuisine/HF_CACHE
python3 pixtral_evaluation_multichoice_task2_large.py
