#!/bin/bash                                                                                                                             

#SBATCH --job-name=llmsegm                                                                                                                
#SBATCH --account=joaiml                                                                                                                
#SBATCH --partition=ml-gpu                                                                                                              
#SBATCH --time=00:30:00                                                                                                                 
#SBATCH --nodes 1                                                                                                                       
#SBATCH --ntasks 1                                                                                                                      
#SBATCH --output /p/project1/joaiml/stenlund1/proj1/LLMSegm_iu/runs/training.%j.out                                                   
#SBATCH --error /p/project1/joaiml/stenlund1/proj1/LLMSegm_iu/runs/training.%j.err




# Clean environment and load modules

module purge     
module load Stages/2023
module load GCC/11.3.0 
module load Python/3.10.4
module load ParaStationMPI
module load TensorFlow
module load OpenCV
module load JupyterLab

# Activate environment
source /p/home/jusers/stenlund1/deep/jupyter_kernel_venvs/stenlund1_proj1/bin/activate

# Ensure python packages installed in the virtual environment are always prefered
export PYTHONPATH=/p/home/jusers/stenlund1/deep/jupyter_kernel_venvs/stenlund1_proj1/lib/python3.10/site-packages:${PYTHONPATH}

# Vars

MAIN=/p/project1/joaiml/stenlund1/proj1/LLMSegm_iu
path2data=$MAIN/data/data_small
path2out=$MAIN/out
model_name=cis-lmu/glot500-base
lang=iu

# Run model

srun python $MAIN/code/main.py $path2data $path2out $model_name $lang