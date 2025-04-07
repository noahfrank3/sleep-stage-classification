#!/bin/bash
#SBATCH -J sleep_stage_optimization
#SBATCH -o out.txt
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G

module load python
source ~/my_env/bin/activate
python main.py
