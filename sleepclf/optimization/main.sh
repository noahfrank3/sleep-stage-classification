#!/bin/bash
#SBATCH -J sleep_stage_optimization
#SBATCH -o out.log
#SBATCH --array=0-19
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4G

module load python
source ~/my_env/bin/activate
python main.py
