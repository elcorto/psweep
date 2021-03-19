# About

This package helps you to set up and run parameter studies.

Mostly, you'll start with a script and a for-loop and ask "why do I
need a package for that"? Well, soon you'll want housekeeping tools
and a database for your runs and results. This package exists because
sooner or later, everyone doing parameter scans arrives at roughly the
same workflow and tools.

This package deals with commonly encountered boilerplate tasks:

* write a database of parameters and results automatically
* make a backup of the database and all results when you repeat or
  extend the study
* append new rows to the database when extending the study
* simulate a parameter scan
* **experimental**: support for managing batch runs, e.g. on remote HPC
  systems, including basic `git` workflow support

Otherwise, the main goal is to not constrain your flexibility by
building a complicated framework -- we provide only very basic building
blocks. All data structures are really simple (dicts), as are the
provided functions. The database is a normal pandas DataFrame.


# Getting started

A simple example: Loop over two parameters 'a' and 'b' in a nested loop (grid),
calculate and store the result of a calculation for each parameter combination.

```py
import random
import psweep as ps


def func(pset):
    return {'result': random.random() * pset['a'] * pset['b']}


if __name__ == '__main__':
    a = ps.plist('a', [1,2,3])
    b = ps.plist('b', [77,88])
    params = ps.pgrid(a,b)
    df = ps.run_local(func, params)
    print(df)
```

`pgrid` produces a list `params` of parameter sets (dicts
`{'a': ..., 'b': ...}`) to loop over:

```
[{'a': 1, 'b': 77},
 {'a': 1, 'b': 88},
 {'a': 2, 'b': 77},
 {'a': 2, 'b': 88},
 {'a': 3, 'b': 77},
 {'a': 3, 'b': 88}]
```

and a database of results (pandas DataFrame `df`, pickled file
`calc/database.pk` by default):

```
                               a   b                               _run_id  \
2021-03-19 22:58:17.015223265  1  77  bb32aade-6ac9-4ac2-9019-3093c5f00965
2021-03-19 22:58:17.020342588  1  88  bb32aade-6ac9-4ac2-9019-3093c5f00965
2021-03-19 22:58:17.023407698  2  77  bb32aade-6ac9-4ac2-9019-3093c5f00965
2021-03-19 22:58:17.025991917  2  88  bb32aade-6ac9-4ac2-9019-3093c5f00965
2021-03-19 22:58:17.028507948  3  77  bb32aade-6ac9-4ac2-9019-3093c5f00965
2021-03-19 22:58:17.030908823  3  88  bb32aade-6ac9-4ac2-9019-3093c5f00965

                                                           _pset_id _calc_dir  \
2021-03-19 22:58:17.015223265  e6fbfe50-8c14-49f3-b9f7-83268c73f34d      calc
2021-03-19 22:58:17.020342588  d8599b90-8475-4bfb-96b5-4e7068195057      calc
2021-03-19 22:58:17.023407698  85948244-46a3-4b19-bd1c-2d59e32328a6      calc
2021-03-19 22:58:17.025991917  a775be5e-bc53-4c3a-a237-7151b279f1bd      calc
2021-03-19 22:58:17.028507948  b0ad166f-6bd8-4d64-a82a-7e8e8ac91b20      calc
2021-03-19 22:58:17.030908823  e91309d4-21b4-4d45-9fb6-cb84ffb31e24      calc

                                                  _time_utc  \
2021-03-19 22:58:17.015223265 2021-03-19 22:58:17.015223265
2021-03-19 22:58:17.020342588 2021-03-19 22:58:17.020342588
2021-03-19 22:58:17.023407698 2021-03-19 22:58:17.023407698
2021-03-19 22:58:17.025991917 2021-03-19 22:58:17.025991917
2021-03-19 22:58:17.028507948 2021-03-19 22:58:17.028507948
2021-03-19 22:58:17.030908823 2021-03-19 22:58:17.030908823

                                                             _pset_sha1  \
2021-03-19 22:58:17.015223265  281c595edf7ffa124ac3ef82f26de3f8eb8517f2
2021-03-19 22:58:17.020342588  088056f830758f949823ddc52bf8527f3e727b45
2021-03-19 22:58:17.023407698  b59bc3ae98e8b93de34135d751ed5090c3f087c5
2021-03-19 22:58:17.025991917  52434b66dd37763d09280480ff449d66ffcdcd4f
2021-03-19 22:58:17.028507948  f6295cdfab520d9ede7e709bc68d4d65ffc45848
2021-03-19 22:58:17.030908823  35af18a51c425b09097af78d55ebdd91588c6f3c

                               _pset_seq  _run_seq     result
2021-03-19 22:58:17.015223265          0         0  35.276105
2021-03-19 22:58:17.020342588          1         0  30.183059
2021-03-19 22:58:17.023407698          2         0  50.545904
2021-03-19 22:58:17.025991917          3         0  59.443434
2021-03-19 22:58:17.028507948          4         0  78.027465
2021-03-19 22:58:17.030908823          5         0  62.461030
```

You see the columns 'a' and 'b', the column 'result' (returned by
`func`) and a number of reserved fields for book-keeping such as

```
_run_id
_pset_id
_run_seq
_pset_seq
_pset_sha1
_calc_dir
_time_utc
```

as well as the `df.index` also holding a time stamp.

Observe that one call `ps.run_local(func, params)` creates one `_run_id` -- a
UUID identifying this run, where by "run" we mean one loop over all parameter
combinations. Inside that, each call `func(pset)` creates a UUID `_pset_id` and
a new row in the DataFrame (the database). In addition we also add sequential
integer IDs `_run_seq` and `_pset_seq` for convenience, as well as an
additional hash `_pset_sha1` over the input dict (`pset` in the example) to
`func()`.


# Concepts

The basic data structure for a param study is a list of "parameter sets" or
short "psets", each of which is a dict.


```py
params = [{'a': 1, 'b': 'lala'},  # pset 1
          {'a': 2, 'b': 'zzz'},   # pset 2
          ...                     # ...
         ]
```

Each pset contains values of parameters ('a' and 'b') which are varied during
the parameter study.

You need to define a callback function `func`, which takes exactly one
pset such as:

```py
{'a': 1, 'b': 'lala'}
```

and runs the workload for that pset. `func` must return a
dict, for example:

```py
{'result': 1.234}
```

or an updated 'pset':

```py
{'a': 1, 'b': 'lala', 'result': 1.234}
```

We always merge (`dict.update`) the result of `func` with the pset, which gives
you flexibility in what to return from `func`. In particular, you are free to
also return an empty dict if you record results in another way (see the
`save_data_on_disk` example later).

The psets form the rows of a pandas `DataFrame`, which we use to store the
pset and the result from each `func(pset)`.

The idea is now to run `func` in a loop over all psets in `params`. You do this
using the `ps.run_local()` helper function. The function adds some special
columns such as `_run_id` (once per `ps.run_local()` call) or `_pset_id` (once
per pset). Using `ps.run_local(... poolsize=...)` runs `func` in parallel on
`params` using `multiprocessing.Pool`.

This package offers some very simple helper functions which assist in
creating `params`. Basically, we define the to-be-varied parameters
('a' and 'b') and then use something like `itertools.product` to
loop over them to create `params`, which is passed to `ps.run_local()` to
actually perform the loop over all psets.

```py
>>> from itertools import product
>>> import psweep as ps
>>> a=ps.plist('a', [1,2,3])
>>> b=ps.plist('b', ['xx', 'yy'])
>>> a
[{'a': 1}, {'a': 2}, {'a': 3}]
>>> b
[{'b': 'xx'}, {'b': 'yy'}]
>>> ps.itr2params(product(a,b))
[{'a': 1, 'b': 'xx'},
 {'a': 1, 'b': 'yy'},
 {'a': 2, 'b': 'xx'},
 {'a': 2, 'b': 'yy'},
 {'a': 3, 'b': 'xx'},
 {'a': 3, 'b': 'yy'}]
```

The last pattern is so common, that we have a function for it:
`pgrid()`.

```py
>>> ps.pgrid(a,b)
[{'a': 1, 'b': 'xx'},
 {'a': 1, 'b': 'yy'},
 {'a': 2, 'b': 'xx'},
 {'a': 2, 'b': 'yy'},
 {'a': 3, 'b': 'xx'},
 {'a': 3, 'b': 'yy'}]
```

The logic of the param study is entirely contained in the creation of
`params`. E.g., if parameters shall be varied together (say a and b),
then instead of

```py
>>> product(a,b,c)
```

use

```py
>>> product(zip(a,b), c)
```

The nesting from `zip()` is flattened in `itr2params()` and `pgrid()`

```py
>>> c=ps.plist('c', [None, 1.2, 'X'])
>>> ps.pgrid(zip(a,b),c)
[{'a': 1, 'b': 'xx', 'c': None},
 {'a': 1, 'b': 'xx', 'c': 1.2},
 {'a': 1, 'b': 'xx', 'c': 'X'},
 {'a': 2, 'b': 'yy', 'c': None},
 {'a': 2, 'b': 'yy', 'c': 1.2},
 {'a': 2, 'b': 'yy', 'c': 'X'}]
```

If you want a parameter which is constant, use a list of length one:

```py
>>> const=ps.plist('const', [1.23])
>>> ps.pgrid(zip(a,b), c, const)
[{'a': 1, 'b': 'xx', 'c': None, 'const': 1.23},
 {'a': 1, 'b': 'xx', 'c': 1.2,  'const': 1.23},
 {'a': 1, 'b': 'xx', 'c': 'X',  'const': 1.23},
 {'a': 2, 'b': 'yy', 'c': None, 'const': 1.23},
 {'a': 2, 'b': 'yy', 'c': 1.2,  'const': 1.23},
 {'a': 2, 'b': 'yy', 'c': 'X',  'const': 1.23}]
```

So, as you can see, the general idea is that we do all the loops
*before* running any workload, i.e. we assemble the parameter grid to be
sampled before the actual calculations. This has proven to be very
practical as it helps detecting errors early.

You are, by the way, of course not restricted to use simple nested loops
over parameters using `pgrid()` (which uses `itertools.product`). You
are totally free in how to create `params`, be it using other fancy
stuff from `itertools` or explicit loops. Of course you can also define
a static `params` list

```py
params = [
    {'a': 1,    'b': 'xx', 'c': None},
    {'a': 1,    'b': 'yy', 'c': 1.234},
    {'a': None, 'b': 'xx', 'c': 'X'},
    ...
    ]
```

or read `params` in from an external source such as a database from a
previous study, etc.

The point is: you generate `params`, we run the study.


# The database

By default, `ps.run_local()` writes a database `calc/database.pk` (a pickled
DataFrame) with the default `calc_dir='calc'`. You can turn that off using
`save=False` if you want. If you run `ps.run_local()` again

```py
>>> ps.run_local(func, params)
>>> ps.run_local(func, other_params)
```

it will read and append to that file. The same happens in an interactive
session when you pass in `df` again, in which case we don't read it from disk:

```py
# default is df=None -> create empty df
# save=False: don't write db to disk, optional
>>> df_run_0 = ps.run_local(func, params, save=False)
>>> df_run_0_and_1 = ps.run_local(func, other_params, save=False, df=df_run_0)
```


# `_pset_id`, `_pset_seq`, `_run_id`, `_run_seq`, `_pset_sha1` and repeated runs

See `examples/*repeat.py`.

It is important to get the difference between the two special fields
`_run_id` and `_pset_id`, the most important one being `_pset_id`.

Both are random UUIDs. They are used to uniquely identify things.

Once per `ps.run_local()` call, a `_run_id` and `_run_seq` is created. Which
means that when you call `ps.run_local()` multiple times *using the same
database* as just shown, you will see multiple (in this case two) `_run_id`
and `_run_seq` values.

```
                             _run_id                              _pset_id  _run_seq  _pset_seq
8543fdad-4426-41cb-ab42-8a80b1bebbe2  08cb5f7c-8ce8-451f-846d-db5ac3bcc746         0          0
8543fdad-4426-41cb-ab42-8a80b1bebbe2  18da3840-d00e-4bdd-b29c-68be2adb164e         0          1
8543fdad-4426-41cb-ab42-8a80b1bebbe2  bcc47205-0919-4084-9f07-072eb56ed5fd         0          2
969592bc-65e6-4315-9e6b-5d64b6eaa0b3  809064d6-c7aa-4e43-81ea-cebfd4f85a12         1          3
969592bc-65e6-4315-9e6b-5d64b6eaa0b3  ef5f06d4-8906-4605-99cb-2a9550fdd8de         1          4
969592bc-65e6-4315-9e6b-5d64b6eaa0b3  162a7b8c-3ab5-41bb-92cd-1e5d0db0842f         1          5
```

Each `ps.run_local()` call in turn calls `func(pset)` for each pset in `params`.
Each `func` invocation creates a unique `_pset_id` and increment the integer
counter `_pset_seq`. Thus, we have a very simple, yet powerful one-to-one
mapping and a way to refer to a specific pset.

An interesting special case (see `examples/vary_1_param_repeat_same.py`) is
when you call `ps.run_local()` multiple times using *the exact same* `params`,

```py
>>> ps.run_local(func, params)
>>> ps.run_local(func, params)
```

which is perfectly fine, e.g. in cases where you just want to sample more data
for the same psets in `params` over and over again. In this case, you will have
as above two unique `_run_id`s but *two sets of the same* `_pset_sha1`.

```
                             _run_id                              _pset_id  _run_seq  _pset_seq                                _pset_sha1  a    result
8543fdad-4426-41cb-ab42-8a80b1bebbe2  08cb5f7c-8ce8-451f-846d-db5ac3bcc746         0          0  e4ad4daad53a2eec0313386ada88211e50d693bd  1  0.381589
8543fdad-4426-41cb-ab42-8a80b1bebbe2  18da3840-d00e-4bdd-b29c-68be2adb164e         0          1  7b7ee754248759adcee9e62a4c1477ed1a8bb1ab  2  1.935220
8543fdad-4426-41cb-ab42-8a80b1bebbe2  bcc47205-0919-4084-9f07-072eb56ed5fd         0          2  9e0e6d8a99c72daf40337183358cbef91bba7311  3  2.187107
969592bc-65e6-4315-9e6b-5d64b6eaa0b3  809064d6-c7aa-4e43-81ea-cebfd4f85a12         1          3  e4ad4daad53a2eec0313386ada88211e50d693bd  1  0.590200
969592bc-65e6-4315-9e6b-5d64b6eaa0b3  ef5f06d4-8906-4605-99cb-2a9550fdd8de         1          4  7b7ee754248759adcee9e62a4c1477ed1a8bb1ab  2  1.322758
969592bc-65e6-4315-9e6b-5d64b6eaa0b3  162a7b8c-3ab5-41bb-92cd-1e5d0db0842f         1          5  9e0e6d8a99c72daf40337183358cbef91bba7311  3  1.639455
```

This is a very powerful tool to filter the database for calculations that used
the same pset, e.g. an exact repetition of one experiment. But since we use
UUIDs for `_pset_id`, those calculations can still be distinguished.


# Best practices

The following workflows and practices come from experience. They are, if
you will, the "framework" for how to do things. However, we decided to
not codify any of these ideas but to only provide tools to make them
happen easily, because you will probably have quite different
requirements and workflows.

Please also have a look at the `examples/` dir where we document these
and more common workflows.


## Save data on disk, use UUIDs

Assume that you need to save results from a `func()` call not only in the
returned dict from `func` (or even not at all!) but on disk, for instance when
you call an external program which saves data on disk. Consider this toy
example (`examples/save_data_on_disk/10run.py`):

```py
#!/usr/bin/env python3

import os
import subprocess
import psweep as ps


def func(pset):
    fn = os.path.join(pset["_calc_dir"], pset["_pset_id"], "output.txt")
    cmd = (
        f"mkdir -p $(dirname {fn}); "
        f"echo {pset['a']} {pset['a']*2} {pset['a']*4} > {fn}"
    )
    subprocess.run(cmd, shell=True)
    return {"cmd": cmd}


if __name__ == "__main__":
    params = ps.plist("a", [1, 2, 3, 4])
    df = ps.run_local(func, params)
    print(df)
```

In this case, you call an external program (here a dummy shell command) which
saves its output on disk. Note that we don't return any output from the
external command in `func`'s `return` statement. We only update the database
row added for each call to `func` by returning a dict `{"cmd": cmd}` with the
shell `cmd` we call in order to have that in the database.

Also note how we use the special fields `_pset_id` and `_calc_dir`, which are
added in `ps.run_local()` to `pset` *before* `func` is called.

After the run, we have four dirs for each pset, each simply named with
`_pset_id`:

```
calc
├── 63b5daae-1b37-47e9-a11c-463fb4934d14
│   └── output.txt
├── 657cb9f9-8720-4d4c-8ff1-d7ddc7897700
│   └── output.txt
├── d7849792-622d-4479-aec6-329ed8bedd9b
│   └── output.txt
├── de8ac159-b5d0-4df6-9e4b-22ebf78bf9b0
│   └── output.txt
└── database.pk
```

This is a useful pattern. History has shown that in the end, most naming
conventions start simple but turn out to be inflexible and hard to adapt
later on. I have seen people write scripts which create things like:

    calc/param_a=1.2_param_b=66.77
    calc/param_a=3.4_param_b=88.99

i.e. encode the parameter values in path names, because they don't have
a database. Good luck parsing that. I don't say this cannot be done --
sure it can (in fact the example above easy to parse). It is just not
fun -- and there is no need to. What if you need to add a "column"
for parameter 'c' later? Impossible (well, painful at least). This
approach makes sense for very quick throw-away test runs, but gets out
of hand quickly.

Since we have a database, we can simply drop all data in
`calc/<_pset_id>` and be done with it. Each parameter set is identified
by a UUID that will never change. This is the only kind of naming
convention which makes sense in the long run.


## Post-processing

An example of a simple post-processing script that reads data from disk
(`examples/save_data_on_disk/20eval.py`):

```py
df = ps.df_read("calc/database.pk")

# Filter database
df = df[df.a > 0 & ~df.a.isna()]

arr = np.array(
    [np.loadtxt(f"calc/{pset_id}/output.txt") for pset_id in df._pset_id.values]
)

df["mean"] = arr.mean(axis=1)

cols = ["a", "mean", "_pset_id"]
ps.df_print(df[cols])

ps.df_write(df, "calc/database_eval.pk")
```


## Iterative extension of a parameter study

See `examples/multiple_local_1d_scans/` and `examples/*repeat*`.

You can backup old calc dirs when repeating calls to `ps.run_local()` using the
`backup` keyword.

```py
df = ps.run_local(func, params, backup=True)
```

This will save a copy of the old `calc_dir` to something like

```
calc.bak_2021-03-19T23:20:33.621004Z_run_id_d309c2c6-e4ba-4ef4-934c-2a4b2df07941
```

That way, you can track old states of the overall study, and recover from
mistakes, e.g. by just

```sh
$ rm -r calc
$ mv calc.bak_2021-03-19T2* calc
```

For any non-trivial work, you won't use an interactive session. Instead, you
will have a driver script (say `run.py`) which defines `params` and starts
`ps.run_local()`. Also in a common workflow, you won't define `params` and run
a study once. Instead you will first have an idea about which parameter values
to scan. You will start with a coarse grid of parameters and then inspect the
results and identify regions where you need more data (e.g. more dense
sampling). Then you will modify `params` and run the study again. You will
modify `run.py` multiple times, as you refine your study.


## Use git

Instead or in addition to using `backup`, we recommend a `git`-based workflow
to at least track changes to `run.py` (instead of manually creating backups
such as `run.py.0`, `run.py.1`, ...). We recommend to create a `.gitignore`
such as

```
# ignore backups
calc.bak*

# ignore simulate runs
calc.simulate

# ignore the whole calc/ dir, track only scripts
##calc/

# or just ignore potentially big files coming from a simulation you run
calc/*/*.bin
```

You can even go crazy with [`git-lfs`](https://git-lfs.github.com) here. Again,
we don't enforce a specific workflow. We just provide basic building blocks to
create your own.


## Simulate / Dry-Run: look before you leap

See `examples/vary_1_param_simulate.py`.

When you fiddle with finding the next good `params` and even when using
`backup` and/or `git`, appending to the old database might be
a hassle if you find that you made a mistake when setting up `params`.
You need to abort the current run, copy the backup over or use `git` to go
back.

Instead, while you tinker with `params`, use another
`calc_dir`, e.g.

```sh
# only needed so that we can copy the old database over
$ mkdir -p calc.simulate
$ cp calc/database.pk calc.simulate/
```

```py
df = ps.run_local(func, params, calc_dir='calc.simulate')
```

But what's even better: keep everything as it is and just set
`simulate=True`, which performs exactly the two steps above.

```py
df = ps.run_local(func, params, simulate=True)
```

It will copy only the database, not all the (possible large) data in `calc/` to
`calc.simulate/` and run the study there. Additionally , it *will not call* call
`func()` to run any workload. So you still append to your old database as in a
real run, but in a safe separate dir which you can delete later.


## Advanced: Give runs names for easy post-processing

See `examples/vary_1_param_study_column.py`.

Post-processing is not the scope of the package. The database is a
pandas DataFrame and that's it. You can query it and use your full pandas
Ninja skills here, e.g. "give me all psets where parameter 'a' was
between 10 and 100, while 'b' was constant, ...". You get the idea.

To ease post-processing, it can be useful practice to add a constant parameter
named "study" or "scan" to label a certain range of runs. If you, for instance,
have 5 runs (meaning 5 calls to `ps.run_local()`) where you scan values for
parameter 'a' while keeping parameters 'b' and 'c' constant, you'll have 5
`_run_id` values. When querying the database later, you could limit by
`_run_id` if you know the values:

```py
>>> df_filtered = df[(df._run_id=='afa03dab-071e-472d-a396-37096580bfee') |
                     (df._run_id=='e813db52-7fb9-4777-a4c8-2ce0dddc283c') |
                     ...
                     ]
```

This doesn't look like fun. It shows that the UUIDs (`_run_id` and
`_pset_id`) are rarely meant to be used directly, but rather to
programmatically link psets and runs to other data (as shown above in the
"Save data on disk" example). You can also use the integer IDs `_run_seq` and
`_pset_seq` instead. But still, you need to know to which parameter values they
correspond to.

When possible, you could limit by the constant values of the other parameters:

```py
>>> df_filtered = df[(df.b==10) & (df.c=='foo')]
```

Much better! This is what most post-processing scripts will do. In fact, we
have a shortcut function

```py
>>> conds = [df.b==10, df.c=='foo']
>>> df_filtered = ps.df_filter_conds(df, conds)
```

which is useful in post-processing scripts where `conds` is
created programmatically.

But when you have a column "study" which has the value `'a'` all the
time, it is just

```py
>>> df = df[df.study=='a']
```

You can do more powerful things with this approach. For instance, say
you vary parameters 'a' and 'b', then you could name the "study"
field 'scan=a:b' and encode which parameters (thus column names) you
have varied. Later in the post-processing

```py
>>> study = 'scan=a:b'
# cols = ['a', 'b']
>>> cols = study.split('=')[1].split(':')
>>> values = df[cols].values
```

So in this case, a naming convention *is* useful in order to bypass
possibly complex database queries. But it is still flexible -- you can
change the "study" column at any time, or delete it again.

Pro tip: You can manipulate the database at any later point and add the
"study" column after all runs have been done.

Super Pro tip: Make a backup of the database first!


# Batch runs

We have experimental support for managing calculations on remote systems such
as HPC clusters with a batch system like SLURM. It is basically a modernized
and stripped-down version of
[`pwtools.batch`](https://elcorto.github.io/pwtools/written/background/param_study.html).
Note that we don't use any method like DRMAA to automatically dispatch jobs to
clusters. Our design revolves around maximal user control of each step of the
workflow.

See `ps.prep_batch()` and `examples/batch/`. The workflow is based on
**template files**. In the templates, we use (for now) the standard library's
`string.Template`, where each `$foo` is replaced by a value contained in a
pset, so `$param_a`, `$param_b`, as well as `$_pset_id` and so forth.

We piggy-back on the `run_local()` workflow from above to use all it's power
and flexibility to create batch scripts using template files. Again, you have
full control over every part of the workflow. We just automate the boring
stuff.

Workflow summary:

* define `params` to be varied as shown above (probably in a script, say
  `input.py`)
* in that script, call `ps.prep_batch(params)`, which does
  * use `templates/calc/*`: scripts that you want to run in each batch job
  * use `templates/machines/<mycluster>/jobscript`: batch job script
  * read `templates/machines/<mycluster>/info.yaml`: machine-specific info
    (e.g. command to submit the `jobscript`)
  * define `func()` that will create a dir named `calc/<_pset_id>` for each
    batch job, **replace placeholders** such as `$param_a` from psets
    (including special ones such as `$_pset_id`)
  * call `run_local(func, params)`
  * create a script `calc/run_<mycluster>.sh` to submit all jobs

We replace running jobs locally (i.e. what `ps.run_local()` would do) with some
small helper scripts:

* use `bin/psweep-push <mycluster>` to `rsync` `calc/` to a cluster
* ssh to cluster; execute `calc/run_<mycluster>.sh`, wait ...
* use `bin/psweep-pull <mycluster>` to `rsync` results back

Now suppose that each of our batch jobs produces a output file, then we have
the same post-processing setup as in `save_data_on_disk`, namely

```
calc
├── 63b5daae-1b37-47e9-a11c-463fb4934d14
│   └── output.txt
├── 657cb9f9-8720-4d4c-8ff1-d7ddc7897700
│   └── output.txt
├── d7849792-622d-4479-aec6-329ed8bedd9b
│   └── output.txt
├── de8ac159-b5d0-4df6-9e4b-22ebf78bf9b0
│   └── output.txt
└── database.pk
```

The only difference is that the outputs were not calculated and written locally
but `rsync`ed from the cluster.

And then post-processing is (almost) as before:

* analyze results, run post-processing script(s) on `calc/database.pk`, read in
  `output.txt` for each `_pset_id`
* when extending the study (modify `params`, call `input.py` again), we use the
  same features shown above
  * append to database
  * create new unique `calc/<_pset_id>` without overwriting anything
  * additionally: write a new `calc/run_<mycluster>.sh` with old submit
    commands still in there, but commented out


## git support

Use `prep_batch(..., git=True)` to have some basic git support such as
automatic commits in each call. Make sure to create `.gitignore` first, else
we'll track `calc/` as well, which you may safely do when data in `calc` is
small.


# Install

```sh
$ pip install psweep
```

Dev install of this repo:

```sh
$ pip install -e .
```

See also <https://github.com/elcorto/samplepkg>.

# Tests

```sh
cd src/psweep/tests
pytest test.py
```
