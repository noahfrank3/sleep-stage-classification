#!/bin/bash
#SBATCH --job-name=sleepclf_download_data
#SBATCH --output=output/download_data.log
#SBATCH --time=24:00:00

wget -r -N -c -np -P data https://physionet.org/files/sleep-edfx/1.0.0/
