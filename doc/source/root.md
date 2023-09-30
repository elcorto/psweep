**psweep**

Welcome to the `psweep` docs. We hope that you'll soon loop like us!

```{tableofcontents}
```

**Install**

```sh
$ pip install psweep
```

Dev install of this repo:

```sh
$ pip install -e .
```

To install testing tools, use

```sh
$ pip install -e ".[test]"
```

To also install `dask` tools which we support (`dask.distributed` Client) but
are not hard dependencies, use

```sh
$ pip install -e ".[dask]"
# or to install test and dask deps
##$ pip install -e ".[test,dask]"
```

See also <https://github.com/elcorto/samplepkg>.

**Tests**

```sh
# Run in parallel, e.g. use 4 cores
$ pytest -vx -n4
```

**Legal**

The package and all its docs are created by Steve Schmerler under BSD 3-Clause
license. The psweep logo is created using https://www.textstudio.co.
