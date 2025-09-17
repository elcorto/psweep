# Manual

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
* `git` support to track progress of your work and recover from mistakes
* optional: don't repeat already performed calculations based on parameter set
  hashes
* support for managing batch runs, e.g. on remote HPC infrastructure, using
  either [`dask.distributed`][dask_dist] or a template file workflow

Otherwise, the main goal is to not constrain your flexibility by
building a complicated framework -- we provide only very basic building
blocks. All data structures are really simple (dicts), as are the
provided functions. The database is a normal `pandas` DataFrame.

```{note}
We assume that you run experiments on a local machine (laptop, workstation).
See the [section on distributed computing](s:distributed) for how you can scale
out to more advanced compute infrastructure.
```

## Getting started

A simple example: Loop over two parameters `a` and `b` in a nested loop (grid),
calculate and store the result of a calculation for each parameter combination.

```py
>>> import random
>>> import psweep as ps


>>> def func(pset):
...    return {"result_": random.random() * pset["a"] * pset["b"]}

>>> a = ps.plist("a", [1,2,3])
>>> b = ps.plist("b", [88,99])
>>> params = ps.pgrid(a,b)
>>> df = ps.run(func, params)
```

{func}`~psweep.psweep.pgrid` produces a list `params` of parameter sets (dicts `{'a': ..., 'b':
...}`) to loop over:

```
[{'a': 1, 'b': 88},
 {'a': 1, 'b': 99},
 {'a': 2, 'b': 88},
 {'a': 2, 'b': 99},
 {'a': 3, 'b': 88},
 {'a': 3, 'b': 99}]
```

{func}`~psweep.psweep.run` returns a database of results (`pandas` DataFrame `df`) and saves a
pickled file `calc/database.pk` by default:

```py
>>> import pandas as pd
>>> pd.set_option("display.max_columns", None)
>>> print(df)

   a   b                                _pset_hash  \
0  1  88  2580bf27aca152e5427389214758e61ea0e544e0
1  1  99  f2f17559c39b416483251f097ac895945641ea3a
2  2  88  010552c86c69e723feafb1f2fdd5b7d7f7e46e32
3  2  99  b57c5feac0608a43a65518f01da5aaf20a493535
4  3  88  719b2a864450534f5b683a228de018bc71f4cf2d
5  3  99  54baeefd998f4d8a8c9524c50aa0d88407cabb46

                                _run_id                              _pset_id  \
0  ab95c9a9-05ba-4619-b060-d3a81feeee40  d47ae513-9d70-4844-87f7-f821c9ac124b
1  ab95c9a9-05ba-4619-b060-d3a81feeee40  6cd79932-4dc2-4e16-b98a-dcc17c188796
2  ab95c9a9-05ba-4619-b060-d3a81feeee40  b1420236-86cb-4c37-8a07-ba625ee90c4f
3  ab95c9a9-05ba-4619-b060-d3a81feeee40  5d0165d8-2099-4ad3-92ac-d1d6b08d7125
4  ab95c9a9-05ba-4619-b060-d3a81feeee40  19500d3e-3b37-4815-8cd9-d0c3405269a3
5  ab95c9a9-05ba-4619-b060-d3a81feeee40  8f3c8e8c-77a3-42c1-9ab1-7e6a67eb8845

  _calc_dir                     _time_utc  _pset_seq  _run_seq _exec_host  \
0      calc 2023-01-20 19:54:09.541669130          0         0    deskbot
1      calc 2023-01-20 19:54:09.543383598          1         0    deskbot
2      calc 2023-01-20 19:54:09.544652700          2         0    deskbot
3      calc 2023-01-20 19:54:09.545840979          3         0    deskbot
4      calc 2023-01-20 19:54:09.546977043          4         0    deskbot
5      calc 2023-01-20 19:54:09.548082113          5         0    deskbot

      result_  _pset_runtime
0    3.629665       0.000004
1   59.093600       0.000002
2   84.056801       0.000002
3   79.200365       0.000002
4  240.718717       0.000002
5   37.220296       0.000002
```

You see the columns `a` and `b`, the column `result_` (returned by
`func`) and a number of reserved fields for book-keeping such as

```
_run_id
_pset_id
_run_seq
_pset_seq
_pset_hash
_pset_runtime
_calc_dir
_time_utc
_exec_host
```

Observe that one call `ps.run(func, params)` creates one `_run_id` -- a
UUID identifying this run, where by "run" we mean one loop over all parameter
combinations. Inside that, each call `func(pset)` creates a UUID `_pset_id` and
a new row in the DataFrame (the database). In addition we also add sequential
integer IDs `_run_seq` and `_pset_seq` for convenience, as well as an
additional hash `_pset_hash` over the input dict (`pset` in the example) to
`func()`. `_pset_runtime` is the time of one `func()` call. `_pset_seq` is the
same as the integer index `df.index`. Hashes are calculated using the excellent
[joblib](https://joblib.readthedocs.io) package.


## Concepts

The basic data structure for a param study is a list of "parameter sets" or
short "`pset`s", each of which is a dict.


```py
params = [{"a": 1, "b": 88},  # pset 1
          {"a": 1, "b": 99},  # pset 2
          ...                 # ...
         ]
```

Each pset contains values of parameters which are varied during the parameter
study.

You need to define a callback function `func`, which takes exactly one
pset such as:

```py
{'a': 1, 'b': 88}
```

and runs the workload for that pset. `func` must return a
dict, for example:

```py
{'result_': 1.234}
```

or an updated 'pset':

```py
{'a': 1, 'b': 88, 'result_': 1.234}
```

We always merge (`dict.update()`) the result of `func` with the pset, which gives
you flexibility in what to return from `func`. In particular, you are free to
also return an empty dict if you record results in another way (see the
`save_data_on_disk` example later).

The `pset`s form the rows of a `pandas` DataFrame, which we use to store the
pset and the result from each `func(pset)`.

The idea is now to run `func` in a loop over all `pset`s in `params`. You do this
using the {func}`~psweep.psweep.run` helper function. The function adds some special
columns such as `_run_id` (once per {func}`~psweep.psweep.run` call) or `_pset_id` (once
per pset). Using `ps.run(... poolsize=...)` runs `func` in parallel on
`params` using `multiprocessing.Pool`.

## Naming of database fields

{func}`~psweep.psweep.run` will add book-keeping fields starting with an underscore prefix
(e.g. `_pset_id`). By doing that, they can be distinguished from `pset` fields
`a` and `b`. We *recommend* but not require you to name all fields (dict keys)
generated in `func()` such as `result_` with a trailing or *postfix*
underscore. That way you can in the database clearly distinguish between
book-keeping (`_foo`), `pset` (`a`, `b`) and result-type fields (`bar_`). This
is only a suggestion, you can name the fields in a `pset` and the ones created
in `func()` any way you like. However, we rely on this convention for all
functionality based on `pset` hashes. See [this section for more
details](s:more-on-db-field-names).


## Building parameter grids

This package offers some very simple helper functions which assist in creating
`params`. Basically, we define the to-be-varied parameters and then use
something like `itertools.product()` to loop over them to create `params`,
which is passed to {func}`~psweep.psweep.run` to actually perform the loop over all
`pset`s.

```py
>>> from itertools import product
>>> import psweep as ps
>>> a=ps.plist("a", [1, 2])
>>> b=ps.plist("b", ["xx", "yy"])
>>> a
[{'a': 1}, {'a': 2}]
>>> b
[{'b': 'xx'}, {'b': 'yy'}]

>>> list(product(a,b))
[({'a': 1}, {'b': 'xx'}),
 ({'a': 1}, {'b': 'yy'}),
 ({'a': 2}, {'b': 'xx'}),
 ({'a': 2}, {'b': 'yy'})]

>>> ps.itr2params(product(a,b))
[{'a': 1, 'b': 'xx'},
 {'a': 1, 'b': 'yy'},
 {'a': 2, 'b': 'xx'},
 {'a': 2, 'b': 'yy'}]
```

Here we used the helper function {func}`~psweep.psweep.itr2params` which accepts an iterator
that represents the loops over params. It merges dicts to `pset`s and also deals
with nesting when using `zip()` (see below).

The last pattern is so common that we have a short-cut function
{func}`~psweep.psweep.pgrid`, which basically does `itr2params(product(a,b))`.

```py
>>> ps.pgrid(a,b)
[{'a': 1, 'b': 'xx'},
 {'a': 1, 'b': 'yy'},
 {'a': 2, 'b': 'xx'},
 {'a': 2, 'b': 'yy'}]
```

{func}`~psweep.psweep.pgrid` accepts either a sequence or individual args (but please
check the "pgrid gotchas" section below for some corner cases).

```py
>>> ps.pgrid([a,b])
>>> ps.pgrid(a,b)
```

So the logic of the param study is entirely contained in the creation of
`params`. For instance, if parameters shall be varied together (say `a` and
`b`), then use `zip`. The nesting from `zip()` is flattened in {func}`~psweep.psweep.itr2params`
and {func}`~psweep.psweep.pgrid`.

```py
>>> ##ps.itr2params(zip(a, b))
>>> ps.pgrid([zip(a, b)])
[{'a': 1, 'b': 'xx'},
 {'a': 2, 'b': 'yy'}]
```

Let's add a third parameter to vary. Of course, in general, plists can have
different lengths.

```py
>>> c=ps.plist("c", [88, None, "Z"])
>>> ##ps.itr2params(product(zip(a, b), c))
>>> ##ps.pgrid([zip(a, b), c])
>>> ps.pgrid(zip(a, b), c)
[{'a': 1, 'b': 'xx', 'c': 88},
 {'a': 1, 'b': 'xx', 'c': None},
 {'a': 1, 'b': 'xx', 'c': 'Z'},
 {'a': 2, 'b': 'yy', 'c': 88},
 {'a': 2, 'b': 'yy', 'c': None},
 {'a': 2, 'b': 'yy', 'c': 'Z'}]
```

If you want to add a parameter which is constant, use a list of length one.

```py
>>> const=ps.plist("const", [1.23])
>>> ps.pgrid(zip(a, b), c, const)
[{'a': 1, 'b': 'xx', 'c': 88,   'const': 1.23},
 {'a': 1, 'b': 'xx', 'c': None, 'const': 1.23},
 {'a': 1, 'b': 'xx', 'c': 'Z',  'const': 1.23},
 {'a': 2, 'b': 'yy', 'c': 88,   'const': 1.23},
 {'a': 2, 'b': 'yy', 'c': None, 'const': 1.23},
 {'a': 2, 'b': 'yy', 'c': 'Z',  'const': 1.23}]
```

Besides {func}`~psweep.psweep.pgrid`, we have another convenience function {func}`~psweep.psweep.stargrid`, which
creates a specific param sampling pattern, where we vary params in a "star"
pattern (and not a full pgrid) around constant values (middle of the "star").

```py
>>> const=dict(a=1, b=77, c=11)
>>> a=ps.plist("a", [1,2,3,4])
>>> b=ps.plist("b", [77,88,99])
>>> ps.stargrid(const, vary=[a, b])
[{'a': 1, 'b': 77, 'c': 11},
 {'a': 2, 'b': 77, 'c': 11},
 {'a': 3, 'b': 77, 'c': 11},
 {'a': 4, 'b': 77, 'c': 11},
 {'a': 1, 'b': 88, 'c': 11},
 {'a': 1, 'b': 99, 'c': 11}]
```

This is useful in cases where you know that parameters are independent and you
want to do just a "line scan" for each parameter around known good values.

So, as you can see, the general idea is that we do all the loops
*before* running any workload, i.e. we assemble the parameter grid to be
sampled before the actual calculations. This has proven to be very
practical as it helps detecting errors early.

You are, by the way, of course not restricted to use simple nested loops
over parameters using {func}`~psweep.psweep.pgrid` (which uses `itertools.product()`). You
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

### How to define default params

There are two ways to do this. You can use a list of length 1 as mentioned
above with the `const` param. This will end up in the database, which is
(probably) what you want. So, let's define a default value for "c" this way.

```py
a = ps.plist("a", [1, 2, 3])
b = ps.plist("b", [88, 99])
c = ps.plist("c", [1.23])
params = ps.pgrid(a, b, c)
```

Another solution is to define defaults in `func`, like so:

```py
def func(pset, c_default=1.23):
    a_val = pset["a"]
    b_val = pset["b"]
    # Use c_default if "c" is not in pset.
    c_val = pset.get("c", c_default)
    return {"result_": random.random() * a_val * b_val * c_val, "c": c_val}
```

In this case, the default value of param "c" won't end up in the database,
unless we add it to the returned dict, which we did in the example above.

### Gotchas

#### `pgrid`

tl;dr: `pgrid(a,b,...)` is a convenience API. It can't handle all corner
cases. If in doubt, use `pgrid([a,b,...])` (or even `itr2params(product(...))`
directly).

Note that for a single param we have

```py
>>> a=ps.plist("a", [1,2])
>>> a
[{'a': 1}, {'a': 2}]
>>> ps.pgrid([a])
[{'a': 1}, {'a': 2}]
```

i.e. the loop from `itertools.product()` is over `[a]` which returns `a`
itself. You can leave off `[...]` if you have at least two args, say `a` and
`b` as in

```py
>>> pgrid([a,b])
>>> pgrid(a,b)
```

For a single arg calling `pgrid(a)` is wrong since then `itertools.product()`
will be called on the entries of `a` which is not what you want. In fact doing
so will raise an error.

Also, in case

```py
>>> pgrid([zip(a,b)])
```

the list `[zip(a,b)]` is what you want to loop over and `pgrid(zip(a,b))` will
raise an error, just as in case `pgrid(a)` above.

And as before, if you have more plists, then `[...]` is optional, e.g.

```py
>>> pgrid([zip(a,b), c])
>>> pgrid(zip(a, b), c)
```


#### `zip`

When using `zip(a,b)`, make sure that `a` and `b` have the same length, else `zip`
will return an iterator whose length is `min(len(a), len(b))`.


## The database

By default, {func}`~psweep.psweep.run` writes a database `calc/database.pk` (a pickled
DataFrame) with the default `calc_dir='calc'`. You can turn that off using
`save=False` if you want. If you run {func}`~psweep.psweep.run` again

```py
>>> ps.run(func, params)
>>> ps.run(func, other_params)
```

it will read and append to that file. The same happens in an interactive
session when you pass in `df` again, in which case we don't read it from disk:

```py
# default is df=None -> create empty df
# save=False: don't write db to disk, optional
>>> df_run_0 = ps.run(func, params, save=False)
>>> df_run_0_and_1 = ps.run(func, other_params, save=False, df=df_run_0)
```

## Special database fields and repeated runs

See `examples/*repeat.py`.

It is important to get the difference between the two special fields
`_run_id` and `_pset_id`, the most important one being `_pset_id`.

Both are random UUIDs. They are used to uniquely identify things.

Once per {func}`~psweep.psweep.run` call, a `_run_id` and `_run_seq` is created. Which
means that when you call {func}`~psweep.psweep.run` multiple times *using the same
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

Each {func}`~psweep.psweep.run` call in turn calls `func(pset)` for each pset in `params`.
Each `func` invocation creates a unique `_pset_id` and increment the integer
counter `_pset_seq`. Thus, we have a very simple, yet powerful one-to-one
mapping and a way to refer to a specific pset.

An interesting special case (see `examples/vary_1_param_repeat_same.py`) is
when you call {func}`~psweep.psweep.run` multiple times using *the exact same* `params`,

```py
>>> ps.run(func, params)
>>> ps.run(func, params)
```

which is perfectly fine, e.g. in cases where you just want to sample more data
for the same `pset`s in `params` over and over again. In this case, you will
have as above two unique `_run_id`s, unique `_pset_id`s, but *two sets of the
same* `_pset_hash`.

```
                             _run_id                              _pset_id  _run_seq  _pset_seq                                _pset_hash  a   result_
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


### Filter by `pset` hashes

As mentioned, we create a hash for each `pset`, which is stored in the
`_pset_hash` database column. This unlocks powerful database filter options.

As shown above, when calling {func}`~psweep.psweep.run` twice
with the same `params` you get a second set of calculations. But suppose you
have a script where you keep modifying the way you create `params` and you just
want to add some more scans *without* removing the code that generated the old
`params` that you already have in the database. In that case use

```py
>>> ps.run(func, params, skip_dups=True)
```

This will skip all `pset`s already in the database based on their hash and
only add calculations for new `pset`s.


(s:more-on-db-field-names)=
### More details on naming database fields

We implement the convention to ignore fields starting and ending in an
underscore in {func}`~psweep.psweep.pset_hash` (and all functions that use it, in particular
{func}`~psweep.psweep.run`) to ensure that the hash includes only `pset` variables. For example
should you ever want to re-calculate the hash, as in

```py
>>> for idx, row_dct in enumerate(ps.df_extract_dicts(df)):
...     df.at[idx, "_pset_hash_new"] = ps.pset_hash(row_dct)

>>> df
   a                                _pset_hash                              _pset_id  ...   result_                            _pset_hash_new
0  1  64846e128be5c974d6194f77557d0511542835a8  61f899a8-314b-4a19-a359-3502e3e2d009  ...  0.880328  64846e128be5c974d6194f77557d0511542835a8
1  2  e746f91e51f09064bd7f1e516701ba7d0d908653  cd1dc05b-0fab-4e09-9798-9de94a5b3cd3  ...  0.815945  e746f91e51f09064bd7f1e516701ba7d0d908653
2  3  96da150761c9d66b31975f11ef44bfb75c2fdc11  6612eab6-5d5a-4fbf-ae18-fdb4846fd459  ...  0.096946  96da150761c9d66b31975f11ef44bfb75c2fdc11
3  4  79ba178b3895a603bf9d84dea82e034791e8fe30  bf5bf881-3273-4932-a3f3-9c117bca921b  ...  2.606486  79ba178b3895a603bf9d84dea82e034791e8fe30
```

then the hash goes only over the `a` field, ignoring `_pset_id`, any other
prefix field, as well as `result_`. Thus, `_pset_hash` and `_pset_hash_new`
must be the same.

Note that you can do the above using
{func}`~psweep.psweep.df_update_pset_hash`, which updates the `_pset_hash`
field. We do this automatically in {func}`~psweep.psweep.run` so that when
params come in that add new pset variables (extend a study case), the hashes
are kept up-to-date.

If you don't use any functionality based on pset hashes, such as filtering
duplicate `pset`s, then this doesn't affect you at all. For example you
can have prefix or postfix names in your params.

```py
>>> a=ps.plist("_a", [1,2])
>>> b=ps.plist("b_", [77,88])
>>> df=ps.run(func=lambda pset: {"result": pset["_a"] + pset["b_"]},
...           params=ps.pgrid(a, b),
...           save=False)
>>> ps.df_print(df, cols=["_a", "b_", "result"])
 _a  b_  result
  1  77      78
  1  88      89
  2  77      79
  2  88      90
```

However in this case, the hash calculation got handed an empty `pset` (since `_a`
and `b_` are ignored), so the hash is the same for each `pset`.

```py
>>> ps.pset_hash({})
'62977be8e45d8a56a5537c11dfd5d2fd8dda69e0'

>>> df._pset_hash
0    62977be8e45d8a56a5537c11dfd5d2fd8dda69e0
1    62977be8e45d8a56a5537c11dfd5d2fd8dda69e0
2    62977be8e45d8a56a5537c11dfd5d2fd8dda69e0
3    62977be8e45d8a56a5537c11dfd5d2fd8dda69e0
```


## Best practices

The following workflows and practices come from experience. They are, if
you will, the "framework" for how to do things. However, we decided to
not codify any of these ideas but to only provide tools to make them
happen easily, because you will probably have quite different
requirements and workflows.

Please also have a look at the `examples/` dir where we document these
and more common workflows.


### Save data on disk, use UUIDs

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
    df = ps.run(func, params)
    print(df)
```

In this case, you call an external program (here a dummy shell command) which
saves its output on disk. Note that we don't return any output from the
external command in `func`'s `return` statement. We only update the database
row added for each call to `func` by returning a dict `{"cmd": cmd}` with the
shell `cmd` we call in order to have that in the database.

Also note how we use the special fields `_pset_id` and `_calc_dir`, which are
added in {func}`~psweep.psweep.run` to `pset` *before* `func` is called.

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
`calc/<pset_id>` and be done with it. Each parameter set is identified
by a UUID that will never change. This is the only kind of naming
convention which makes sense in the long run.


### Post-processing

An example of a simple post-processing script that reads data from disk
(`examples/save_data_on_disk/20eval.py`):

```py
df = ps.df_read("calc/database.pk")

# Filter database
df = df[df.a > 0 & ~df.a.isna()]

arr = np.array(
    [np.loadtxt(f"calc/{pset_id}/output.txt") for pset_id in df._pset_id.values]
)

# Add new column to database, print and write new eval database
df["mean"] = arr.mean(axis=1)

cols = ["a", "mean", "_pset_id"]
ps.df_print(df[cols])

ps.df_write("calc/database_eval.pk", df)
```


### Iterative extension of a parameter study

See `examples/multiple_local_1d_scans/` and `examples/*repeat*`.

You can backup old calc dirs when repeating calls to {func}`~psweep.psweep.run` using the
`backup` keyword.

```py
df = ps.run(func, params, backup=True)
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
will have a driver script (say `input.py`, or a jupyter notebook, or ...) which
defines `params` and starts {func}`~psweep.psweep.run`. Also in a common workflow, you
won't define `params` and run a study once. Instead you will first have an idea
about which parameter values to scan. You will start with a coarse grid of
parameters and then inspect the results and identify regions where you need
more data (e.g. more dense sampling). Then you will modify `params` and run the
study again. You will modify `input.py` multiple times, as you refine your
study.


### Use git

Instead or in addition to using

```py
>>> ps.run(..., backup=True)
```

we recommend a `git`-based workflow to at least track changes to `input.py`
(instead of manually creating backups such as `input.py.0`, `input.py.1`, ...). You
can manually `git commit` at any point of course, or use

```py
>>> ps.run(..., git=True)
```

This will commit any changes made to e.g. `input.py` itself and create a commit
message containing the current `_run_id` such as

```
psweep: batch_templates_git: run_id=68f5d9b7-efa6-4ed8-9384-4ffccab6f7c5
```

We **strongly recommend** to create a `.gitignore` such as

```
## ignore backups
calc.bak*

# ignore simulate runs
calc.simulate

# ignore the whole calc/ dir, track only scripts
##calc/

# or just ignore potentially big files coming from a simulation you run
calc/*/*.bin
```

#### How to handle large files when using git

The first option is to `.gitignore` them. Another is to use
[`git-lfs`][git-lfs] (see the section on that later). That way you track their
changes but only store the most recent version. Or you leave those files on
another local or remote storage and store only the path to them (and maybe a
hash) in the database. It's up to you.

Again, we don't enforce a specific workflow but instead just provide basic
building blocks.

### Simulate / Dry-Run: look before you leap

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
df = ps.run(func, params, calc_dir='calc.simulate')
```

But what's even better: keep everything as it is and just set
`simulate=True`, which performs exactly the two steps above.

```py
df = ps.run(func, params, simulate=True)
```

It will copy only the database, not all the (possible large) data in `calc/` to
`calc.simulate/` and run the study there. Additionally , it *will not call* call
`func()` to run any workload. So you still append to your old database as in a
real run, but in a safe separate dir which you can delete later.


### Advanced: Give runs names for easy post-processing

See `examples/vary_1_param_study_column.py`.

Post-processing is not the scope of the package. The database is a
`pandas` DataFrame and that's it. You can query it and use your full `pandas`
Ninja skills here, e.g. "give me all `pset`s where parameter 'a' was
between 10 and 100, while 'b' was constant, ...". You get the idea.

To ease post-processing, it can be useful practice to add a constant parameter
named "study" or "scan" to label a certain range of runs. If you, for instance,
have 5 runs (meaning 5 calls to {func}`~psweep.psweep.run`) where you scan values for
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
programmatically link `pset`s and runs to other data (as shown above in the
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

(s:wrap-failed-func)=
### How to handle failing workloads

If the code in `func()` fails and raises a Python exception, it will take
down the whole run (the call `run(func, params, ...)`).

Sometimes we instead want to catch that, log the event, carry on and later
analyze the failed `pset`s. One way to do this is by wrapping `func` in a
`try`...`except` block, for example:

```py
import traceback

def func(pset):
    ... here be code ...
    return dict(...)

def safe_func(pset):
    try:
        ret = func(pset)
        ret.update(_failed=False, _exc_txt=None)
    except:
        txt = traceback.format_exc()
        print(f"{pset=} failed, traceback:\n{txt}")
        ret = dict(_failed=True, _exc_txt=txt)
    return ret

df = ps.run(safe_func, params)
```

This will add a bool field `_failed` to the database, as well as a text field
`_exc_txt` which stores the exception's traceback message.

We don't implement this as a feature and only provide examples, which keeps
things flexible. Maybe you want `_failed` to be called `_crashed` instead, or you want
to log more data.

For post-processing, you would then do something like:

```py
df = ps.df_read("calc/database.pk")

# Only successful psets
df_good = df[~df._failed]

# All failed pset
df_fail = df[df._failed]

for pset_id, txt in zip(df_fail._pset_id, df_fail._exc_txt):
    print(f"{pset_id=}\n{txt}")
```

See also [this discussion](https://github.com/elcorto/psweep/issues/1) for
more.

### Text logging per `pset`

All text sent to the terminal (`sys.stdout` and `sys.stderr`) by any code in
`func()` can be redirected to a database field `_logs`, a file
`<calc_dir>/<pset_id>/logs.txt`, or both, by using

```py
ps.run(..., capture_logs="db")
ps.run(..., capture_logs="file")
ps.run(..., capture_logs="db+file")
```

For `print()`-style logging, this is very convenient.

```{note}
This feature may have side effects since it dynamically redirects
`sys.stdout` and `sys.stderr`, so code in `func()` making advanced use of
those may not work as intended.
```

To also log all text from shell commands that you call, make sure to capture
this on the Python side and print it. For example

```py
import subprocess

def func(pset):
    input_file_txt = f"""
param_a={pset["a"]}
param_b={pset["b"]}
"""
    pset_id = pset["_pset_id"]
    ps.file_write(f"calc/{pset_id}/input.txt", input_file_txt)
    txt = subprocess.run(
        f"cd calc/{pset_id}; my_fortran_simulation_code.x < input.txt | tee output.txt",
        shell=True,
        check=False,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
    ).stdout.decode()
    print(txt)
    return dict()

ps.run(func, params, capture_logs="db+file")
```

This will log all text output and errors from the command executed by
`subprocess.run()`. In this example we use `subprocess.run(..., check=False)`
which prevents raising an exception in case of a shell error (exit code != 0).
To detect and log a fail, you may look into `txt` for signs of an error,
either directly in `func()`, or later in post-processing, for example using
regular expressions (see also [this
discussion](https://github.com/elcorto/psweep/issues/1)). But you can also use
`check=True` and wrap the function [as described earlier](s:wrap-failed-func).

```py
def func(pset):
    ...
    # Same as above only that check=True
    print(subprocess.run(..., check=True).stdout.decode())
    return dict()


def safe_func(pset):
    """Simple safe_func that doesn't create new database fields."""
    try:
        return(func(pset))
    except:
        print(traceback.format_exc())
        return dict()

ps.run(safe_func, params, capture_logs="db+file")
```

See also `examples/capture_logs.py` for a worked example.


(s:distributed)=
## Running on (HPC) compute infrastructure

We have support for managing calculations on remote systems such
as HPC clusters with a batch system like SLURM.

You can use `dask` tooling or a template-based workflow. Both have their
pros and cons (no free lunch!).

### `dask`

Overview of `dask`'s architecture (from [the `dask`
docs](https://docs.dask.org/en/stable/deploying.html)).

```{figure} https://docs.dask.org/en/stable/_images/dask-cluster-manager.svg
:width: 80%
:align: center
```

#### Overview

The `dask` ecosystem provides [`dask.distributed`][dask_dist] which
lets you spin up a dask cluster on distributed infrastructure.
[`dask_jobqueue`][dask_jq] supports doing this on HPC machines.

With these tools, you have fine-grained control over how many batch jobs you
start. This is useful for cases where you have many small workloads (each
`func(pset)` runs only a few seconds, say). Then one batch job per `pset`, as
in the [templates workflow](s:templates), would
be overhead and using many dask workers living in only a few (or one!) batch
job is more efficient.

From the `psweep` side of things, you just need to provide a `dask` client (see
below) and that's it. The challenge is to determine how to distribute the work
and to define a matching `dask` client, which depends entirely on the type of
workload and the compute infrastructure you wish to run on.

See `examples/batch_dask/slurm_cluster_settings.py` for a detailed example
which covers the most important `SLURMCluster` settings.

#### API

The `psweep` API to use `dask` is `df=ps.run(..., dask_client=client)`.
First, let's look at an example for using `dask` locally.

```py
from dask.distributed import Client, LocalCluster

# The type of cluster depends on your compute infastructure. Replace
# LocalCluster with e.g. dask_jobqueue.SLURMCluster when running on a HPC
# machine.
cluster = LocalCluster()
client = Client(cluster)

# The same as always, only pass dask_client in addition
ps.run(..., dask_client=client)
```

`LocalCluster` uses all cores by default, which has the same effect as using

```py
from multiprocessing import cpu_count

ps.run(..., poolsize=cpu_count())
```

but using `dask` instead of `multiprocessing`.

If you run on a chunky workstation with many cores instead of a full-on HPC
machine, you may still want to use `dask` + `LocalCluster` instead of
`multiprocessing` since then you can access `dask`'s dashboard (see below).
Also check [this tutorial](https://www.youtube.com/watch?v=N_GqzcuGLCY) and
others by Matthew Rocklin.

Note: `LocalCluster` is actually default, so

```py
client = Client()
```

is sufficient for running locally.


#### HPC machine example

The example below uses the SLURM workload manager typically found in HPC
centers.

Create a script that

* spins up a dask cluster on your (HPC) infrastructure (`SLURMCluster`), this
  is the only additional step compared to running locally
* defines parameters to sweep over, runs workloads and collects results in a
  database (`ps.pgrid(...)`, `ps.run(...)`)

`run_psweep.py`

```{literalinclude} ../../../examples/batch_dask/run_psweep.py
```

To start calculations

 * run `python run_psweep.py` directly from the HPC cluster head
   node (if the cluster is yours and/or there is not runtime quota on the head
   node) or better yet
 * create a SLURM job script, say `dask_control.slurm`, that starts a dask
   control process, in our case this is `python run_psweep.py`, run `sbatch
   dask_control.slurm`, make sure that this job has large time limit; see example
   below (recommended)

`dask_control.slurm`

```{literalinclude} ../../../examples/batch_dask/dask_control.slurm
   :language: sh
```

After submitting the job, the SLURM `squeue` output could look like this:

```sh
hpc$ sbatch dask_control.slurm
hpc$ squeue -l -u $USER -O JobID,Name,NumNodes,NumTasks,NumCPUs,TimeLimit,State,ReasonList

JOBID     NAME         NODES  TASKS  CPUS  TIME_LIMIT  STATE    NODELIST(REASON)
6278655   dask_control 1      1      2     2-00:00:00  RUNNING  node042
6278656   dask-worker  1      1      10    10:00       RUNNING  node123
6278657   dask-worker  1      1      10    10:00       RUNNING  node124
```

The `dask_control` process runs on `node042` while the two jobs that hold
the dask cluster with 10 dask workers each run `node123` and `node124`.

#### Access to the `dask` dashboard

`dask` has a [nice dashboard](https://docs.dask.org/en/stable/dashboard.html)
to visualize the state of all workers. Say you set the dashboard port to 3333
in `SLURMCluster`. If the `dask_control` process runs on the head node, you
only need to forward that port to your local machine (`mybox`):

```sh
mybox$ ssh -L 2222:localhost:3333 hpc.machine.edu
mybox$ browser localhost:2222
```

If `dask_control` runs on a compute node, you will need a second tunnel:

```sh
mybox$ ssh -L 2222:localhost:3333 hpc.machine.edu
hpc$ ssh -L 3333:localhost:3333 node42
mybox$ browser localhost:2222
```

#### Jupyter

Many HPC centers offer a [JupyterHub](https://jupyter.org/hub) service, where a
small batch job is started that runs a JupyterLab. When using that, you can
skip the `dask_control` batch job part and run the content of `run_psweep.py`
in Jupyter. JupyterLab also has [an
extension](https://github.com/dask/dask-labextension) that gives you access to
the dask dashboard and more.


#### How to request GPUs

See `examples/batch_dask/run_psweep_jax_gpu.py`.


#### Pros and Cons

```{admonition} Pros
:class: hint

* Simple API, same workflow as if running locally.
* Fine-grained control over mapping of workloads to compute resources.
* You can run on all the different compute infrastructures that
  `dask.distributed` (+ `dask_jobqueue`) support, such as Kubernetes.
* `dask`'s JupyterLab integration.
```

```{admonition} Cons
:class: attention

* Assigning different resources (GPUs or not, large or small memory) to different
  dask workers is hard or even not possible with `dask_jobqueue` since the
  design assumes that you create one `FooCluster` with fixed resource
  specifications. See these discussions for more:
  https://github.com/dask/dask-jobqueue/issues/378
  https://github.com/dask/dask-jobqueue/issues/378.
* HPC cluster time limits might be a problem if some jobs are waiting in the
  queue for a long time and meanwhile the `dask_control` batch job gets
  terminated due to its time limit. The latter should be set to the longest
  available on the system, e.g. if you have a queue for long-running jobs, use
  that. Also the batch jobs holding workers have a time limit. See [this part
  of the `dask_jobqueue`
  docs][dask_time_limits]
  for how to handle the latter and [this
  comment](https://github.com/elcorto/psweep/issues/11#issuecomment-1757564391)
  for more.
* More software to install: On the HPC machine, you need `psweep`,
  `dask.distributed` and `dask_jobqueue`.
* `dask` uses [`cloudpickle`](https://github.com/cloudpipe/cloudpickle) to
  serialize data before sending it to workers, which may fail in some cases
  such as data generated by
  [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html)
```

(s:templates)=
### Templates

This template-based workflow is basically a modernized
and stripped-down version of
[`pwtools.batch`](https://elcorto.github.io/pwtools/written/background/param_study.html).
Note that we don't use any method like DRMAA to automatically dispatch jobs to
clusters. Instead we write out a shell script to submit jobs. The design
revolves around maximal user control of each step of the workflow.

```{note}
This workflow, while being the most general, might not be ideal for all
workloads. See the [Pros and Cons](s:template-pro-con) section below.

```

The workflow is based on **template files**. They are processed using
[jinja](https://github.com/pallets/jinja) to replace each `{{foo}}` or `{{ foo }}`
by a value contained in a pset, so `{{param_a}}`,
`{{param_b}}`, as well as `{{_pset_id}}` and so forth.

```{note}
We support an older template mechanism where we use the standard
library's
[`string.Template`](https://docs.python.org/3/library/string.html#template-strings)
with "dollar" placeholder syntax (`$foo`, `$param_a`, `$_pset_id`). You can
switch this on with `ps.prep_batch(..., template_mode="dollar")` but the default
is `template_mode="jinja"`.

If your template files are shell scripts that contain variables like `$foo`,
you need to escape the `$` with `$$foo`, else they will be treated as
placeholders.
```

We piggy-back on the {func}`~psweep.psweep.run` workflow from above to, instead
of running jobs with it, create batch scripts using template files. We
replace {func}`~psweep.psweep.run` by {func}`~psweep.psweep.prep_batch` to
process `params`, so call `ps.prep_batch(params)` instead of `ps.run(func,
params)`. The rest of the setup code that creates `params` stays the same.

See `examples/batch_templates` for a full example.

Except for `func`, you can pass all keywords that {func}`~psweep.psweep.run`
accepts also to {func}`~psweep.psweep.prep_batch`, in particular `skip_dups`
when extending a study, as well as `backup` and/or `simulate`. The latter will
also create a dir `calc.simulate` and copy over the database. Keywords that
deal with executing `func` such as `poolsize`, `dask_client` or `capture_logs`
are not relevant here, though.

You can use the proposed workflow below directly on the remote machine (need to
install `psweep` there) or run it locally and use a copy-to-cluster workflow.
Since we actually don't start jobs or talk to the batch system, you have full
control over every part of the workflow. We just automate the boring stuff.

#### Workflow summary

* define `params` to be varied as shown above (probably in a script, say
  `input.py`)
* in that script, call `ps.prep_batch(params)` (instead of `ps.run(func, params)`);
  this performs these steps for you
  * use `templates/calc/*`: scripts that you want to run in each batch job
  * use `templates/machines/<mycluster>/jobscript`: batch job script
  * read `templates/machines/<mycluster>/info.yaml`: machine-specific info
    (e.g. command to submit the `jobscript`)
  * define `func()` that will create a dir named `calc/<pset_id>` for each
    batch job, **replace placeholders** such as `{{param_a}}` from `pset`s
    (including special ones such as `{{_pset_id}}`)
  * call `run(func, params)`
  * create a script `calc/run_<mycluster>.sh` to submit all jobs
* if running locally
  * use `scp` or `rsync` or the helper script `bin/psweep-push <mycluster>` (uses
    `rsync`) to copy `calc/` to a cluster
* ssh to cluster
* execute `calc/run_<mycluster>.sh`, wait ...
* if running locally
  * use `scp` or `rsync` or the helper script use `bin/psweep-pull <mycluster>`
    (uses `rsync`) to copy results back

Now suppose that each of our batch jobs produces an output file, then we have
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

Post-processing is (almost) as before:

* analyze results, run post-processing script(s) on `calc/database.pk`, read in
  `output.txt` for each `_pset_id`
* when extending the study (modify `params`, call `input.py` again which calls
  `prep_batch(params)`), we use the same features shown above
  * append to database
  * create new unique `calc/<pset_id>` without overwriting anything
  * **additionally**: write a new `calc/run_<mycluster>.sh` with old submit
    commands still in there, but commented out

#### Templates layout and written files

An example template dir, based on ``examples/batch_templates``:

```
templates
├── calc
│   └── run.py
└── machines
    ├── cluster
    │   ├── info.yaml
    │   └── jobscript
    └── local
        ├── info.yaml
        └── jobscript
```

The template workflow is very generic. One aspect of this design is that each
template file is treated as a simple text file, be it a Python script, a shell
script, a config file or anything else. Above we use a small Python script
`run.py` for demonstration purposes and communicate `pset` content (parameters
to vary) by replacing placeholders in there. See [this
section](s:template-tips) for other ways to improve this in the Python
script case.

##### calc templates

Each file in `templates/calc` such as `run.py` will be treated as
template, goes thru the file template machinery and ends up in
`calc/<pset_id>/`.

##### machine templates

The example above has machine templates for 2 machines, "local" and a
remote machine named "cluster". `psweep` will generate `run_<machine>.sh`
for both. Also you must provide a file `info.yaml` to store
machine-specific info. ATM this is only `subcmd`, e.g.

```yaml
# templates/machines/cluster/info.yaml
---
subcmd: sbatch
```

All other SLURM stuff can go into `templates/machines/cluster/jobscript`, e.g.

```sh
#SBATCH --time 00:20:00
#SBATCH -o out.log
#SBATCH -J foo_{{_pset_seq}}_{{_pset_id}}
#SBATCH -p bar
#SBATCH -A baz

echo "hostname=$(hostname)"

module purge
module load python

python3 run.py
```

For the "local" machine we'd just use `sh` (or `bash` or ...) as "submit
command".

```yaml

# templates/machines/local/info.yaml
---
subcmd: sh
```


The files written are:

```
run_cluster.sh                              # submit script for each machine
run_local.sh

calc/3c4efcb7-e37e-4ffe-800d-b05db171b41b   # one dir per pset
├── jobscript_cluster                       # jobscript for each machine
├── jobscript_local
└── run.py                                  # from templates/calc/run.py
calc/11967c0d-7ce6-404f-aae6-2b0ea74beefa
├── jobscript_cluster
├── jobscript_local
└── run.py
calc/65b7b453-96ec-4694-a0c4-4f71c53d982a
...
```

In each `run_<machine>.sh` we use the `subcmd` from `info.yaml`.

```sh
#!/bin/sh
here=$(pwd)
cd 3c4efcb7-e37e-4ffe-800d-b05db171b41b; sbatch jobscript_cluster; cd $here  # run_seq=0 pset_seq=0
cd 11967c0d-7ce6-404f-aae6-2b0ea74beefa; sbatch jobscript_cluster; cd $here  # run_seq=0 pset_seq=1
...
```

#### git support

Use `prep_batch(..., git=True)` to have some basic git support such as
automatic commits in each call. It uses `run(..., git=True)` when
creating batch scripts, so all best practices for that apply here as well. In
particular, make sure to create `.gitignore` first, else we'll track `calc/` as
well, which you may safely do when data in `calc` is small. Else use `git-lfs`,
for example.

(s:template-tips)=
#### Tips and tricks

The template workflow will create one batch job per `pset`. If the workload in
`run.py` is small, then this creates some overhead. Use this only if you have
sizable workloads. For many small lightweight workloads, better start one batch
job manually (ideally on a node with many cores), replace `run.py`'s content
with a function `func(pset)` and then use something like `ps.run(func, params,
poolsize=n_cores)` in that job.

The communication of the `pset` content via filling templates is suited for
non-Python workloads, such as simulation software with text input files and no
Python interface, for instance


`templates/calc/input_file`:

```
# Input file for super fast bzzrrr simulation code.

param_a = {{param_a}}
param_b = {{param_b}}
```

`templates/machines/cluster/jobscript`:

```sh
#!/bin/bash

#SBATCH --time 00:20:00
#SBATCH -o out.log
#SBATCH -J foo_{{_pset_seq}}_{{_pset_id}}
#SBATCH -p bar
#SBATCH -A baz
#SBATCH -n 1 -c 8

module purge
module load bzzrrr/1.2.3

mpirun -np 8 bzzrrr input_file
```

For Python workloads (the one we have in
`examples/batch_templates/templates/calc/run.py`), using placeholders like
`{{param_a}}` is actually a bit of an anti-pattern, since we would ideally like to
pass the `pset` to the workload using Python. With templates, we have two ways
of dealing with that.

##### Write pset content to disk

First we can use
`prep_batch(..., write_pset=True)` which writes
`<calc_dir>/<pset_id>/pset.pk`. In `run.py` you can then do

```py
import psweep as ps

def func(pset):
    ... here be code ...
    return result

# Even better (less code):
##from my_utils_module import func

# write <calc_dir>/<pset_id>/result.pk
ps.pickle_write("result.pk", func(ps.pickle_read("pset.pk")))
```

Note that from all the `<calc_dir>/<pset_id>/result.pk` files, you can actually
reconstruct a `database_eval.pk` since each `result.pk` is a dict with results,
so a row in the DataFrame.

```py
df_input = ps.df_read("calc/database.pk")

dicts = [
    ps.pickle_read(f"calc/{pset_id}/result.pk") for pset_id in
    df_input._pset_id.values
    ]

df_eval = pd.DataFrame(dicts)
ps.df_write("db/database_eval.pk", df_eval)
```

##### Read pset content from the database


```py
import psweep as ps

def func(pset):
    ... here be code ...
    return result

# Even better (less code):
##from my_utils_module import func

# Read pset from the database. We only pass in pset_id via template
# replacement.
pset = ps.df_extract_pset(ps.df_read("../database.pk"), "{{_pset_id}}")

# write <calc_dir>/<pset_id>/result.pk
ps.pickle_write("result.pk", func(pset))
```

See also `examples/batch_templates_read_from_db`.

This saves writing `pset.pk`, but still, both approaches are not ideal
since we communicate by writing (potentially many) small files, namely
`run.py`, `pset.pk` (when using `write_pset=True)`, `result.pk`
`jobscript_<cluster>` for each `<calc_dir>/<pset_id>/` dir. Also, when using
`write_pset=True`, `run.py` would be the exact same file for each job.

On the other hand, the templates workflow is the most simple and low-tech
solution, esp. if you are already familiar with HPC systems, whereas `dask` +
`dask_jobqueue` has a bit of a learning curve.

So we recommend using templates if

* `dask_jobqueue` doesn't have something that matches your compute infrastructure
* your workload doesn't have a Python interface / it is cumbersome to create a
  Python function that runs your workload (for example your `run.py` needs to
  run in a docker / apptainer container and you need to run shell setup code in
  your SLURM script and inside the container before calling `python run.py`)
* you have a moderate number of jobs (say 100 .. 1000) and each one contains a
  huge workload (runs for hours or days, say)

(s:template-pro-con)=
#### Pros and Cons

```{admonition} Pros
:class: hint
* Works for all workloads, also ones with no Python interface; for instance you
  can sweep over SLURM settings easily (number of cores, memory, GPU yes or
  no, ...).
* No `dask_control` process. Once you submitt jobs and assuming that each job
  has a time limit that is large enough, jobs may wait in the queue for days.
  The workflow leverages the more non-interactive nature of HPC systems.
  You can once in a while run a post-processing script and collect
  results that are already written (e.g. `<calc_dir>/<pset_id>/result.pk`) and
  start to analyse this (partial) data.
* Uses only batch system tooling, so essentially shell scripts. If you copy
  (`rsync`!) generated files to and from the HPC machine, you don't even need to
  install `psweep` there.
* Easy debugging, since all files of one job sit in the job's
  `<calc_dir>/<pset_id>/`. For example you can copy this dir, make a change to
  a file, `sbatch jobscript_<cluster>` again to repeat just that job.
```

```{admonition} Cons
:class: attention

* More manual multi-step workflow comparted to local and `dask` where we can
  just do `df=ps.run(...)` and have results in `df`.
* Always one batch job per `pset`.
* The workflow is suited for more experienced HPC users. You may need to work
  a bit more directly with your batch system (`squeue`, `scancel`, ... in case
  of SLURM). Also some shell scripting skills are useful.
```


## Special topics

### How to migrate a normal `git` repo to `git-lfs`

First we'll quickly mention how to set up LFS in a *new* repo. In this case we
just need to configure `git lfs` to track certain files. We'll use dirs
`_pics/` and `calc/` as examples.

```sh
$ git lfs track "_pics/**" "calc/**"
```

where `**` means recursive. This will write the config to `.gitattributes`.

```sh
$ cat .gitattributes
_pics/** filter=lfs diff=lfs merge=lfs -text
calc/** filter=lfs diff=lfs merge=lfs -text
```

Please refer to the [git lfs docs][git-lfs] for more info.

Note: LFS can be tricky to get right the first time around. We actually
recommend to fork the upstream repo, call that remote something like
`lfsremote` and experiment with that before force-pushing LFS content to
`origin`. Anyhow, let's continue.

Now we like to migrate an existing git repo to LFS. Here we don't need to call
`git lfs track` because we'll use `git lfs migrate import` to convert the repo.
We will use the `-I/--include=` option to specify which files we would like to
convert to LFS. Those patterns will end up in `.gitattributes` and the file
will even be created of not present already.

We found that only one `-I/--include=` at a time works, but we can separate
patterns by "," to include multiple ones.

```sh
$ git lfs migrate import -I '_pics/**,calc/**' --include-ref=master

$ cat .gitattributes
_pics/** filter=lfs diff=lfs merge=lfs -text
calc/** filter=lfs diff=lfs merge=lfs -text
```

Now after the migrate, all LFS files in the working tree (files on disk) have
been converted from their real content to text stub files.

```sh
$ cat _pics/foo.png
version https://git-lfs.github.com/spec/v1
oid sha256:de0a80ff0fa13a3e8cf8662c073ce76bfc986b64b3c079072202ecff411188ba
size 28339
```

The following will not change that.

```sh
$ git push lfsremote master -f
$ git lfs fetch lfsremote --all
```

Their real content is however still contained in the `.git` dir. A simple

```sh
$ git lfs checkout .

$ cat _pics/foo.png
<<binary foo>>
```

will bring the content back to the working dir.


## Scope and related projects

This project aims to be agnostic to the field of study. We target problems that
can be formulated as **computational experiments**: "let's vary X,Y and
analyze the results".

Unsurprisingly, there is a huge pile of similar tools. This project is super
small and as such of course lacks a lot of features that other packages offer.
We only

* provide simple helpers to set up the params ({func}`~psweep.psweep.plist`, {func}`~psweep.psweep.pgrid`,
{func}`~psweep.psweep.stargrid`)
* hook into the `concurrent.futures` API (`multiprocessing` or `dask`) for
  parallel (HPC) runs / provide a template workflow for HPC runs
* give you a `DataFrame` with useful metadata

Otherwise we stay out of the way. In particular we are not a workflow engine,
a tool that lets you define chains of interdependent tasks (but see [this
section](s:task-deps)). We just attempt to scratch some particular itches here:

* simulate runs
* backups
* git support
* simple local database (no db server to set up, no Mongo, etc)
* interactive (Python REPL) and script-driven runs
* local runs, also in parallel
* tooling for remote runs (on HPC and other infrastructure)
* minimal naming conventions, rely on UUIDs
* hash-based database filtering
* no yaml-ish DSLs, just Python please, thank you :)
* no CLIs, just Python please, thank you :)
* no config files, just Python please, thank you :)
* not application specific (e.g. machine learning)

Here is a small list of related projects that we have looked at so far. We try
to roughly classify each tool, based on its *main* use case, as best as we can.

tool | workflow | param sweep | exp. tracking
-|-|-|-
https://materialsproject.github.io/fireworks/ | * ||
https://www.aiida.net/ | * ||
https://luigi.readthedocs.io | * ||
https://snakemake.readthedocs.io | * ||
https://github.com/eviatarbach/parasweep ||*|
https://github.com/SmokinCaterpillar/pypet ||*|
https://github.com/pharmbio/sciluigi |*||
https://github.com/open-research/sumatra |||*
https://www.nist.gov/programs-projects/simulation-management-tools |||*
https://github.com/IDSIA/sacred |||*
https://www.wandb.ai/ |||*
https://github.com/maiot-io/zenml |*||
https://github.com/LLNL/maestrowf |*||
https://mlflow.org/ |||*
https://metaflow.org/ |*||*
https://www.nextflow.io/ |*||
https://dvc.org/ |||*
https://apps.fz-juelich.de/jsc/jube/jube2/docu/index.html ||*|*
https://www.prefect.io/ |*||
https://hydra.cc/ |*||

See also [this long list of workflow
engines](https://github.com/meirwah/awesome-workflow-engines) and [this even
longer list of MLOps
tools](https://neptune.ai/blog/mlops-tools-platforms-landscape).

(s:task-deps)=
### Handling task dependencies

In `psweep` we assume that workloads are independent and "embarrassingly
parallel" if using `multiprocessing`, `dask` or templates.

So while `psweep` is not a workflow engine where you can model task
dependencies as DAGs, one way to handle (simple linear) task dependencies is by
running things in order manually, say `10prepare.py`, `20production.py`,
`30eval.py`, where the first two can use `psweep` to compute stuff, update the
database and store intermediate data on disk, which the next script would pick
up. The "workflow" is to run all scripts in order.

A more advanced version of this approach is `examples/tree_study.py`, where we
show how to handle tree-like task dependencies with `psweep` tooling. The use
case is a multi-step study, where parameter sweeps depend on each other:
the sweep in step N depends on results of sweep N-1 and we read results
from the previous sweep's database.

This is still super low tech, simple, but of course also a bit brittle. For
more challenging dependencies and more reproducibility, look into using one of
the workflow frameworks above.


[git-lfs]: https://git-lfs.github.com
[dask]: https://dask.org
[dask_dist]: https://distributed.dask.org
[dask_jq]: https://jobqueue.dask.org
[dask_time_limits]: https://jobqueue.dask.org/en/latest/clusters-advanced-tips-and-tricks.html#how-to-handle-job-queueing-system-walltime-killing-workers
