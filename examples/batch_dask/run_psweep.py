"""
Start a dask cluster running on HPC infrastructure via SLURMCluster. Run
workloads via ps.run() and collect results. The only difference to a local run
is using ps.run(..., dask_client=client).
"""

import time

from dask.distributed import Client
from dask_jobqueue import SLURMCluster

import numpy as np
import psweep as ps


def func(pset):
    time.sleep(1)
    return dict(b=pset["a"] * np.random.rand(10))


if __name__ == "__main__":
    cluster = SLURMCluster(
        queue="some_queue,some_other_queue",
        ##account="some_account",
        processes=10,
        cores=10,
        memory="10GiB",
        walltime="00:10:00",
        local_directory="dask_tmp",
        log_directory="dask_log",
        scheduler_options={"dashboard_address": ":3333"},
    )

    print(cluster.dashboard_link)
    print(cluster.job_script())

    params = ps.plist("a", range(100))

    # Start 2 batch jobs, each with 10 dask workers (processes=10) and 10
    # cores, so 1 core (1 thread) / worker and 20 workers in total (2 jobs x 10
    # workers). Each worker gets 1 GiB of memory (memory="10GiB" for 10
    # workers). See examples/batch_dask/slurm_cluster_settings.py
    cluster.scale(jobs=2)
    client = Client(cluster)
    df = ps.run(func, params, dask_client=client)

    ps.df_print(df, cols=["_pset_id", "_exec_host"])
