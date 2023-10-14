"""
GPU usage and debugging example.

Request 1 GPU and 1 worker per batch job.

    SLURMCluster(
        ...
        processes=1,
        cores=1,
        job_extra_directives=["--gres gpu:1"],
    )

We can also request more workers and GPUs per job, hoping to get 1 GPU per
worker.

    SLURMCluster(
        ...
        processes=4,
        cores=4,
        job_extra_directives=["--gres gpu:4"],
    )


However, there can be problems with workers not seeing GPU devices, since with
dask_jobqueue we can't pin dask workers to GPU devices as we would do in dask's
CLI (see https://docs.dask.org/en/stable/gpu.html#specifying-gpus-per-machine)

CUDA_VISIBLE_DEVICES=0 dask-worker ...
CUDA_VISIBLE_DEVICES=1 dask-worker ...

To investigate this, we can use the func() definition below to record which
devices workers saw, together with

    cluster = SLURMCluster(
        ...
        processes=2,
        cores=2,
        job_extra_directives=["--gres gpu:2"],
    )

and a little bit of pandas to find out what's going on:

    >>> cols=["_exec_host", "dev_input", "dev_output1", "dev_output2", "devs_avail", "worker_name"]
    >>> ps.df_print(df[cols].sort_values("worker_name").drop_duplicates(), cols=cols)
       _exec_host  dev_input dev_output1 dev_output2             devs_avail      worker_name
    ga002.cluster TFRT_CPU_0  TFRT_CPU_0  TFRT_CPU_0      [CpuDevice(id=0)] SLURMCluster-0-0
    ga002.cluster      gpu:0       gpu:0       gpu:0 [gpu(id=0), gpu(id=1)] SLURMCluster-0-1
    ga003.cluster      gpu:0       gpu:0       gpu:0 [gpu(id=0), gpu(id=1)] SLURMCluster-1-0
    ga003.cluster TFRT_CPU_0  TFRT_CPU_0  TFRT_CPU_0      [CpuDevice(id=0)] SLURMCluster-1-1
    ga001.cluster      gpu:1       gpu:1       gpu:1            [gpu(id=1)] SLURMCluster-2-0
    ga001.cluster      gpu:0       gpu:0       gpu:0            [gpu(id=0)] SLURMCluster-2-1

We see that:

* on node ga001, each worker (SLURMCluster-2-0, SLURMCluster-2-1) sees one GPU
* on nodes ga002 and ga003, one worker sees the CPU, the other sees both GPUs
  but only gpu:0 is used

So better stick to 1 GPU and 1 dask worker (process) per batch job.

What however should work is requesting multiple GPUs and 1 worker

    SLURMCluster(
        ...
        processes=1,
        cores=1,
        job_extra_directives=["--gres gpu:4"],
    )

if the workload in func() can take advantage of multiple GPUs (e.g. parallel
neural network training).
"""

import os
import sys
import textwrap

from dask.distributed import Client, get_worker
from dask_jobqueue import SLURMCluster

import psweep as ps

import jax.numpy as jnp
import jax

##import logging
##logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def print_device_info(device: jax.Device, prefix=""):
    print(
        textwrap.dedent(
            f"""
            {prefix}{device=}
            {prefix}{device.id=}
            {prefix}{device.device_kind=}
            {prefix}{device.platform=}
            """
        )
    )


def func(pset):
    print(f">>> enter {pset=}")

    # Find print() logs in dask_log/dask-worker-XXXXXX.out
    cmd = """
hostname
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
"""

    # The dask worker running this workload (funcv()). worker.name is
    # "SLURMCluster-X-Y"
    worker = get_worker()
    print(f"{worker.name=}")

    # All available compute devices that jax sees on the node where func() is
    # run. Can be CPU, GPU, TPU.
    print(f"{jax.devices()=}")
    print(ps.system(cmd).stdout.decode())

    # Make sure we are on a GPU node. This only checks that we *have* GPUs
    # (https://github.com/google/jax/issues/971). After creating arrays and
    # doing some computations, we can also check that GPUs were used by looking
    # at array.device(). Since jax will fall back to CPU if needed, we
    # skip the assert here. Also that way we can debug worker <-> GPU device
    # assignment.
    ##assert jax.default_backend() == "gpu"

    # Generate a jax array. In contrast to pytorch, jax will automatically try
    # to allocate this on the "best" device available, which is
    # jax.default_backend(), so a GPU if available, and else fall back to CPU.
    nn = pset["nn"]
    X = jax.random.normal(jax.random.PRNGKey(0), (nn, nn))

    # Example: the first GPU on the node. If CUDA_VISIBLE_DEVICES=0,1,... then
    # this is GPU0.
    #
    # device=gpu(id=0)
    # device.id=0
    # device.device_kind='NVIDIA A100-SXM4-40GB'
    # device.platform='gpu'
    print_device_info(X.device(), prefix="X.")

    # Some computations.
    evals, evecs = jnp.linalg.eigh(X.T @ X)
    print_device_info(evals.device(), prefix="evals.")
    print_device_info(evecs.device(), prefix="evecs.")

    ##for arr in [X, evecs, evals]:
    ##    assert arr.device().platform == "gpu", f"{arr.device().platform=}"

    # Bring back GPU arrays to host, transform to numpy and put in database.
    # Arrays can be large, so storing this info in the database is not ideal,
    # better use the database only for psets and other metadata. Write arrays
    # to disk directly (np.savez_compressed()) or store in something like HDF5
    # or zarr (needs parallel write access, haven't tested that so far).
    ##return dict(evals=jax.device_get(evals), evecs=jax.device_get(evals))

    print(f"<<< leave {pset=}")

    # https://github.com/dask/dask-jobqueue/issues/299#issuecomment-609002077
    sys.stdout.flush()
    os.fsync(sys.stdout.fileno())

    return dict(
        worker_name=worker.name,
        devs_avail=str(jax.devices()),
        dev_input=str(X.device()),
        dev_output1=str(evecs.device()),
        dev_output2=str(evals.device()),
    )


if __name__ == "__main__":
    cluster = SLURMCluster(
        queue="some_queue,some_other_queue",
        ##account="some_account",
        processes=1,
        cores=1,
        job_extra_directives=["--gres gpu:1"],
        memory="10GiB",
        walltime="01:00:00",
        local_directory="dask_tmp",
        log_directory="dask_log",
        scheduler_options={"dashboard_address": ":3333"},
    )

    nn = ps.plist("nn", ps.intspace(100, 300, num=100))
    ##params = ps.pgrid([nn])
    params = nn

    cluster.scale(jobs=3)
    client = Client(cluster)
    df = ps.run(func, params, dask_client=client)
