#!/bin/bash
#SBATCH --job-name=sleepclf_download_data
#SBATCH --output=output/download_data.log

wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/
