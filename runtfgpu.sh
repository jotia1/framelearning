#! /bin/bash

#SBATCH --job-name=DCFF
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=joshua.arnold1@uq.net.au

JOBNAME="DCFF"


#module load tensorflow
source /opt/local/stow/tensorflow-0.10.0/bin/activate
ARGS="--error=$JOBNAME.err --output=$JOBNAME.out"
srun --gres=gpu:1 -n1 $ARGS python first.py > $JOBNAME.log
srun --gres=gpu:1 -n1 echo $LD_LIBRARY_PATH >> u.txt

wait
