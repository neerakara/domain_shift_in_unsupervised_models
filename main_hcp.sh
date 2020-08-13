#!/bin/bash
#
# Script to send job to BIWI clusters using qsub.
# Usage: qsub train_i2l_mapper.sh
# Adjust line '-l hostname=xxxxx' before runing.
# The script also requires changing the paths of the CUDA and python environments and the code to the local equivalents of your machines.

## SGE Variables:
#
## otherwise the default shell would be used
#$ -S /bin/bash
#
## <= 1h is short queue, <= 6h is middle queue, <= 48 h is long queue
#$ -l h_rt=23:59:00

## the maximum memory usage of this job, (below 4G does not make much sense)
#$ -l h_vmem=40G

# Host and gpu settings
#$ -l gpu
#$ -l hostname='!biwirender12'&'!biwirender20'   ## <-------------- Comment in or out to force a specific machine

## stderr and stdout are merged together to stdout
#$ -j y
#
# logging directory. preferably on your scratch
#$ -o /usr/bmicnas01/data-biwi-01/nkarani/projects/domain_shift_unsupervised/code/v2.0/logs/  ## <---------------- CHANGE TO MATCH YOUR SYSTEM
#
## send mail on job's end and abort
#$ -m a

# activate virtual environment
source /usr/bmicnas01/data-biwi-01/nkarani/softwares/anaconda/installation_dir/bin/activate tf_v1_12

## EXECUTION OF PYTHON CODE:
python /usr/bmicnas01/data-biwi-01/nkarani/projects/domain_shift_unsupervised/code/v2.0/run_recon_hcp.py --sli $1 --base $2 --usfact $3 --num_iter $4 --n1 $5 --n2 $6
## parser.add_argument('--sli', type = int, default = 3) # 1, 2, 3, 4, 5
## parser.add_argument('--base', default = "EK/") # CK, CX, EK, FP, KE, KT, NK, TC, VA
## parser.add_argument('--usfact', type = float, default = 3) # 2, 3, 4, 5

echo "Hostname was: `hostname`"
echo "Reached end of job file."