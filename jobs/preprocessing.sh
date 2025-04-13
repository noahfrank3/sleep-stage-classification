#!/bin/bash
#SBATCH --job-name=sleepclf_preprocessing
#SBATCH --output=../output/preprocessing.log
#SBATCH --partition=pace
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4G

module load python
source ~/my_env/bin/activate
cd ../sleepclf
python -m sleepclf.core
