#!/bin/bash

#PBS -N ML_multiclass
#PBS -l nodes=1:ppn=36
#PBS -l mem=200G
#PBS -l walltime=300:00:00
#PBS -M cmmoranga001@st.ug.edu.gh

export NUMTHREADS=36
module load ug/gnu/parallel-20190122
module load anaconda/3
module load library/libicu/50.2
#source activate falconfk
module load openmpi-3.0.0
module load java/8
#source activate r-tensorflow

#alias python3=python3.6
#conda install tensorflow-gpu

module load python/3.6.8
module load R/3.6
cd /home/cmmoranga/machine_learn/multi/


Rscript firework_multiclass.R
Rscript firework_um_nmi.R
Rscript firework_sm_nmi.R
