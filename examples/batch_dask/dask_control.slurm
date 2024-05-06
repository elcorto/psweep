#!/usr/bin/env bash

#SBATCH --job-name=dask_control
#SBATCH -o log_dask_control-%j.log
#SBATCH -p some_queue,some_other_queue
#SBATCH --account=some_account
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=2-0

# Submit this job manually via "sbatch <this_script>". It will run on some node
# and from there submit the batch jobs to spin up the dask cluster defined in
# run_psweep.py. The dask cluster will then run workloads. After all work is
# done, the dask cluster will be teared down and this job will exit.
#
# The reason for using a batch job to run a the dask control process (so python
# run_psweep.py) is that typically on HPC machines, there is a time limit for
# user processes started on the head / login nodes. Therefore the dask control
# process may get killed before the dask workers have finished processing.

module load python

python run_psweep.py
