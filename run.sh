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

set -x  # Print commands as they execute
set -e  # Exit immediately on any error

module load Anaconda3/2023.09-0  # load the exact modules required
module load cuda12.2/toolkit/12.2.2

echo "Starting the job..."

# source ~/.bashrc
# conda init
# conda activate avici
cd ldp
conda run -n avici python -u train.py --config config/ldp_train_rlc_sparse.yaml \
    --run_name rlc-sparse --smoke_test n --checkpoint_dir /scratch/trllmout/pmaab/avici/checkpoint \
    --n_steps 50000  > out.log 2>&1
