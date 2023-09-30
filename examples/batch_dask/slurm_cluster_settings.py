from dask.distributed import Client
from dask_jobqueue import SLURMCluster
##from icecream import ic
import time


"""
SLURM terms

* node: a single physical machine (computer) with a number of CPUs (each having
  multiple cores) and memory (RAM)

* task: a single program instance (e.g. python myscript.py or mycommand), or an
  MPI process (which is essentially a copy of mycommand running under MPI
  control); each MPI process is identified by its MPI "rank"

* -n/--ntasks: default is 1 task per node

* cpu: in SLURM, this is not a hardware CPU but a consumable resource, and,
  depending on the SLURM configuration, can correspond a physical CPU core
  (most likely the case) or a logical core if hyper-threading is enabled

* -c/--cpus-per-task: allocate that many SLURM cpus per task (good for e.g.
  multithreading)

* -N/--nodes: If -N is not specified, the default behavior is to allocate
  enough nodes to satisfy the requirements of the -n and -c options.

* --ntasks-per-node: use for fine control of task distribution, e.g. when using
  MPI + OpenMP over multiple nodes, e.g. 4 nodes, 5 tasks per node, 10 OpenMP
  threads per task = 4*5*10 cores in total:
  --nodes=4 --ntasks-per-node=5 --cpus-per-task=10

* job: batch job, submitted by one sbatch command, can have multiple steps
  (default 1), each step can have multiple (parallel) tasks (default 1)

  Some examples, adapted from https://blog.ronin.cloud/slurm-intro . Each is
  the content of a SLURM batch script.

  Single core task:

    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=1

    echo "I'm a task"

  2 parallel independent tasks per job:

    #SBATCH --ntasks=2
    #SBATCH --cpus-per-task=1

    srun --ntasks=1 echo "I'm task 1"
    srun --ntasks=1 echo "I'm task 2"

  Tasks can have multiple threads, make sure to use 1 node:

    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=8

    mycommand --threads 8

  Multiple multithreaded tasks in one job, one node per task:

    #SBATCH --nodes=4
    #SBATCH --ntasks=4
    #SBATCH --cpus-per-task=4

    srun --ntasks=1 mycommand1 --threads 4
    srun --ntasks=1 mycommand2 --threads 4
    srun --ntasks=1 mycommand3 --threads 4
    srun --ntasks=1 mycommand4 --threads 4

  MPI job with 16 tasks across 2 nodes:

    #SBATCH --ntasks=16
    #SBATCH --ntasks-per-node=8

    mpirun myscript
"""


def func(x):
    time.sleep(5)
    return x**2


# Run this on the cluster head node or better yet, put this into a batch job to
# avoid getting terminated after a time limit on the head node.

if __name__ == "__main__":
    # Create dask cluster scheduler that uses SLURM. Resources (cores, memory,
    # processes, ...) are per batch job.
    #
    # The cluster scheduler will start as many batch jobs as necessary to
    # create the requested number of dask workers (by
    # SLURMCluster(n_workers=...), cluster.scale(n=...) or cluster.adapt()).
    #
    # We can also specify the number of jobs (cluster.scale(jobs=...)). Then
    # the number of dask workers will be processes * jobs. See below. It's
    # complicated.

    # API docs of scale() and adapt() are not rendered at
    # https://jobqueue.dask.org, so here we go.

    # cluster.scale(n=None, jobs=0, memory=None, cores=None)
    #     Scale cluster to specified configurations.
    #
    #     Parameters
    #     ----------
    #     n : int
    #        Target number of workers
    #     jobs : int
    #        Target number of jobs
    #     memory : str
    #        Target amount of memory
    #     cores : int
    #        Target number of cores

    # cluster.adapt(*args, minimum_jobs: int = None, maximum_jobs: int = None, **kwargs)
    #
    #     Scale Dask cluster automatically based on scheduler activity.
    #
    #     Parameters
    #     ----------
    #     minimum : int
    #        Minimum number of workers to keep around for the cluster
    #     maximum : int
    #        Maximum number of workers to keep around for the cluster
    #     minimum_memory : str
    #        Minimum amount of memory for the cluster
    #     maximum_memory : str
    #        Maximum amount of memory for the cluster
    #     minimum_jobs : int
    #        Minimum number of jobs
    #     maximum_jobs : int
    #        Maximum number of jobs
    #     **kwargs :
    #        Extra parameters to pass to dask.distributed.Adaptive
    #
    #     See Also
    #     --------
    #     dask.distributed.Adaptive : for more keyword arguments
    #

    # Examples:
    #
    # cluster.scale(n=n_workers)
    #
    #     n_workers  SLURMCluster params  created
    #     ---------  -------------------  --------------------------
    #     workers    processes   cores    jobs    threads_per_worker
    #     20         20          20       1       1
    #     20         20          40       1       2
    #     20         10          10       2       1
    #     20         1           1        20      1
    #
    # cluster.scale(jobs=n_jobs)
    #
    #     n_jobs     SLURMCluster params  created
    #     -------    -------------------  --------------------------
    #     jobs       processes   cores    workers threads_per_worker
    #     1          20          20       20      1
    #     1          1           20       1       20
    #     1          20          1        20      1
    #     2          20          20       40      1

    # https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.SLURMCluster.html
    cluster = SLURMCluster(
        # Default is 0. We are supposed to use scale() or adapt().
        ##n_workers=50,
        queue="some_queue,some_other_queue",
        ##account="some_account",
        # Number of dask worker processes (independent Python processes) per
        # batch job (dask_worker --nworkers). Default is processes=1.
        # cores / processes is the number of threads per worker
        # (dask_worker --nthreads).
        processes=20,
        # Number of cores per barch job. Those will be distributed among
        # workers (=processes) as threads (dask_worker --nthreads).
        cores=20,
        # Memory for the whole job. Will be distributed among workers
        # (dask_worker --memory-limit).
        memory="10GiB",
        walltime="00:10:00",
        # Each of the settings in this list will end up in the job script, e.g.
        #   #SBATCH --gres gpu:1
        ##job_extra_directives=["--gres gpu:1"],
        # temp worker files
        local_directory="dask_tmp",
        # batch system job logs (stderr, stdout from batch jobs)
        log_directory="dask_log",
        # ssh -L 2222:localhost:3333 cluster
        # localhost$ browser localhost:2222/status
        scheduler_options={"dashboard_address": ":3333"},
    )

    # Job script generated with
    #   processes=5
    #   cores=5
    #   memory="1G"
    #
    #     #SBATCH -J dask-worker
    #     #SBATCH -e dask_log/dask-worker-%J.err
    #     #SBATCH -o dask_log/dask-worker-%J.out
    #     #SBATCH -p defq,gpu,haicu_a100
    #     #SBATCH -A haicu
    #     #SBATCH -n 1
    #     #SBATCH --cpus-per-task=5
    #     #SBATCH --mem=954M
    #     #SBATCH -t 00:10:00
    #
    #     /home/schmer52/.virtualenvs/dask-py3.9.6/bin/python \
    #     -m distributed.cli.dask_worker tcp://149.220.16.61:35020 \
    #     --nthreads 1 --nworkers 5 --memory-limit 190.73MiB \
    #     --name dummy-name --nanny --death-timeout 60 \
    #     --local-directory dask_tmp
    #
    # This script will be used for each batch job. The call
    #
    #   python -m distributed.cli.dask_worker --nworkers 5
    #
    # will launch 5 independent Python processes ("dask workers") which will
    # share the 5 cores.
    print(cluster.job_script())

    # 'http://149.220.16.61:8787/status' with default port 8787
    print(cluster.dashboard_link)

    cluster.scale(jobs=1)
    ##cluster.adapt(minimum_jobs=0, maximum_jobs=20, minimum=0, maximum=20)

    # Create dask.distributed client as interface to the cluster scheduler.
    client = Client(cluster)

    # [<Future: pending, key: lambda-8b1beef1f48327b54f57f473dd4f3144>,
    #  <Future: pending, key: lambda-f9f80cbfbd887a9ca553ab93c9962da0>,
    #  <Future: pending, key: lambda-bc9a75fd8be34c20f5c2266cb90448a2>,
    #  <Future: pending, key: lambda-b566a501bb2ac2b9305714ac74b65543>,
    #  <Future: pending, key: lambda-724a2c5a5a0437a445f5c6d2abdcf63d>,
    #  ...]
    futures = client.map(func, range(100))
    ##ic(futures[:5])

    # [0, 1, 4, 9, 16, ...]
    results = client.gather(futures)
    ##ic(results[:5])

    # Stop scheduler. Called automatically when script ends.
    ##client.close()

    # Stop workers. Called automatically when script ends.
    ##cluster.close()
