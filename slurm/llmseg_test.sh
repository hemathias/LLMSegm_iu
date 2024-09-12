#!/bin/bash

#SBATCH --job-name=llmsegm                                                                                                                
#SBATCH --account=trustllm-eu                                                                                                                
#SBATCH --partition=develbooster

#SBATCH --time=00:10:00                                                                                                                 
#SBATCH --nodes 1                                                                                                                       
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1

#SBATCH --output /p/project1/trustllm-eu/stenlund1/LLMSegm_iu/runs/training.%j.out                                                   
#SBATCH --error /p/project1/trustllm-eu/stenlund1/LLMSegm_iu/runs/training.%j.err
#SBATCH --exclusive

echo "***********************************************"
echo "Started at:" $(date)
echo "Compute node: $HOSTNAME"
echo "Job ID: $JOB_ID"
ulimit -aH
ulimit -s unlimited
echo "***********************************************"
echo


venvs_dir="/p/project/trustllm-eu/stenlund1/LLMSegm_iu"
venvs_name="venv_proj1"

# Clean environment and load modules

module --force purge
ml Stages/2023
ml GCC/11.3.0
ml OpenMPI/4.1.4
ml PyTorch/1.12.0-CUDA-11.7

# Activate environment
python3.10 -m venv --system-site-packages ${venvs_dir}/${venvs_name}

source ${venvs_dir}/${venvs_name}/bin/activate

# Ensure python packages installed in the virtual environment are always prefered
export PYTHONPATH=${VIRTUAL_ENV}/lib/python3.10/site-packages:${PYTHONPATH}
echo ${VIRTUAL_ENV} # double check
ls ${VIRTUAL_ENV}/lib/python3.10/site-packages/ # double check

# When installing, uncomment pip install
#pip install -r ${venvs_dir}/requirements_proj1.txt

# set comm
PSP_CUDA=1
PSP_UCP=1
PSP_OPENIB=1
export NCCL_SOCKET_IFNAME=ib
export NCCL_IB_HCA=ipogif0
export NCCL_IB_CUDA_SUPPORT=1
export CUDA_VISIBLE_DEVICES="0"

# Vars
MAIN=/p/project/trustllm-eu/stenlund1/LLMSegm_iu
path2data=$MAIN/data/data_small
path2out=$MAIN/out
model_name=cis-lmu/glot500-base
lang=iu

# Run model
#sys.argv[1] = path to train, dev, test data dir
#sys.argv[2] = path to out dir
#sys.argv[3] = model name
#sys.argv[4] = language abbreviation
srun --cpu-bind=none,v --accel-bind=gn python $MAIN/code/main.py $path2data $path2out $model_name $lang