#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --partition=ciaq
#SBATCH --gres=gpu:1
#SBATCH --time=6-06:00:00   # specify job time
#SBATCH --mem=16G         # specify memory
#SBATCH --output=job_tune_parameter_output.log

module load cuda-toolkit/12.2.2  # load necessary modules

srun python parameter_search.py  # run your job
