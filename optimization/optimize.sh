#!/bin/bash
#SBATCH -J sleep_stage_optimization
#SBATCH -o out.txt
#SBATCH --cpus-per-task=80
#SBATCH --mem=230G

module load python
source ~/my_env/bin/activate
python main.py
