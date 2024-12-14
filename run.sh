#!/bin/bash
#SBATCH --job-name=avici-ldp          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --gpus=1                 # number of GPUs per node(only valid under large/normal partition)
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=normal  # partition(large/normal/cpu) where you submit
#SBATCH --account=trllmout      # only require for multiple projects


# #SBATCH --mail-user=pmaab@ust.hk #Update your email address
# #SBATCH --mail-type=begin
# #SBATCH --mail-type=end

module purge                     # clear environment modules inherited from submission
module load Anaconda3/2023.09-0  # load the exact modules required
module load cuda12.2/toolkit/12.2.2

conda activate avici
cd ldp
python train.py --config ./config/ldp_train.yaml --smoke_test n