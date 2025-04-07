#!/bin/bash
#SBATCH -J sleep_stage_optimization
#SBATCH -o out.log
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G

module load python
source ~/my_env/bin/activate
python main.py
