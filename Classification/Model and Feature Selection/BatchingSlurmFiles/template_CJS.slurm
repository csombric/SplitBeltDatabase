#!/bin/bash
#SBATCH --job-name={{ MyJobName }}
#SBATCH --output={{ MyJobName }}.out
#SBATCH --time=6-00:00:00
#SBATCH --mem=250G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --mail-user=cjs180@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails

module purge #make sure the modules environment is sane
module load python/3.7.0

python ClusterTraining_V7.py -l {{ model }} {{ var }} {{ trans}} {{ target }}
