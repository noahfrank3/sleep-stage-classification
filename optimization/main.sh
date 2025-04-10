#!/bin/bash
#SBATCH -J sleep_stage_optimization
#SBATCH -o out.log
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2G

module load python
source ~/my_env/bin/activate
python main.py
