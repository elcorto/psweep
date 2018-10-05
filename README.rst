=====================================================
psweep -- loop like a pro, make parameter studies fun
=====================================================

About
=====

This package helps you to set up and run parameter studies.

Mostly, you'll start with a script and a for-loop and ask "why do I need a
package for that"? Well, soon you'll want housekeeping tools and a database for
your runs and results. This package exists because sooner or later, everyone
doing parameter scans arrives at roughly the same workflow and tools.

This package deals with commonly encountered boilerplate tasks:

* write a database of parameters and results automatically
* make a backup of the database and all results when you repeat or extend the
  study
* append new rows to the database when extending the study
* simulate a parameter scan

Otherwise, the main goal is to not constrain your flexibility by building a
complicated framework -- we provide only very basic building blocks. All data
structures are really simple (dicts), as are the provided functions. The
database is a normal pandas DataFrame.


Getting started
===============

A trivial example: Loop over two parameters 'a' and 'b':

.. code-block:: python

    #!/usr/bin/env python3

    import random
    from itertools import product
    from psweep import psweep as ps


    def func(pset):
        return {'result': random.random() * pset['a'] * pset['b']}


    if __name__ == '__main__':
        a = ps.seq2dicts('a', [1,2,3,4])
        b = ps.seq2dicts('b', [8,9])
        params = ps.loops2params(product(a,b))
        df = ps.run(func, params)
        print(df)

This produces a list ``params`` of parameter sets (dicts ``{'a': ..., 'b': ...}``) to loop
over::

    [{'a': 1, 'b': 8},
     {'a': 1, 'b': 9},
     {'a': 2, 'b': 8},
     {'a': 2, 'b': 9},
     {'a': 3, 'b': 8},
     {'a': 3, 'b': 9},
     {'a': 4, 'b': 8},
     {'a': 4, 'b': 9}]


and a database of results (pandas DataFrame ``df``, pickled file ``calc/results.pk``
by default)::

                               _calc_dir                              _pset_id  \
    2018-07-22 20:06:07.401398      calc  99a0f636-10b3-438c-ab43-c583fda806e8
    2018-07-22 20:06:07.406902      calc  6ec59d2b-7562-4262-b8d6-8f898a95f521
    2018-07-22 20:06:07.410227      calc  d3c22d7d-bc6d-4297-afc3-285482e624b5
    2018-07-22 20:06:07.412210      calc  f2b2269b-86e3-4b15-aeb7-92848ae25f7b
    2018-07-22 20:06:07.414637      calc  8e1db575-1be2-4561-a835-c88739dc0440
    2018-07-22 20:06:07.416465      calc  674f8a2c-bc21-40f4-b01f-3702e0338ae8
    2018-07-22 20:06:07.418866      calc  b4d3d11b-0f22-4c73-a895-7363c635c0c6
    2018-07-22 20:06:07.420706      calc  a265ca2f-3a9f-4323-b494-4b6763c46929

                                                             _run_id  \
    2018-07-22 20:06:07.401398  3e09daf8-c3a7-49cb-8aa3-f2c040c70e8f
    2018-07-22 20:06:07.406902  3e09daf8-c3a7-49cb-8aa3-f2c040c70e8f
    2018-07-22 20:06:07.410227  3e09daf8-c3a7-49cb-8aa3-f2c040c70e8f
    2018-07-22 20:06:07.412210  3e09daf8-c3a7-49cb-8aa3-f2c040c70e8f
    2018-07-22 20:06:07.414637  3e09daf8-c3a7-49cb-8aa3-f2c040c70e8f
    2018-07-22 20:06:07.416465  3e09daf8-c3a7-49cb-8aa3-f2c040c70e8f
    2018-07-22 20:06:07.418866  3e09daf8-c3a7-49cb-8aa3-f2c040c70e8f
    2018-07-22 20:06:07.420706  3e09daf8-c3a7-49cb-8aa3-f2c040c70e8f

                                                _time_utc  a  b     result
    2018-07-22 20:06:07.401398 2018-07-22 20:06:07.401398  1  8   2.288036
    2018-07-22 20:06:07.406902 2018-07-22 20:06:07.406902  1  9   7.944922
    2018-07-22 20:06:07.410227 2018-07-22 20:06:07.410227  2  8  14.480190
    2018-07-22 20:06:07.412210 2018-07-22 20:06:07.412210  2  9   3.532110
    2018-07-22 20:06:07.414637 2018-07-22 20:06:07.414637  3  8   9.019944
    2018-07-22 20:06:07.416465 2018-07-22 20:06:07.416465  3  9   4.382123
    2018-07-22 20:06:07.418866 2018-07-22 20:06:07.418866  4  8   2.713900
    2018-07-22 20:06:07.420706 2018-07-22 20:06:07.420706  4  9  27.358240

You see the columns 'a' and 'b', the column 'result' (returned by ``func``) and
a number of reserved fields for book-keeping such as

::

    _run_id
    _pset_id
    _calc_dir
    _time_utc

and a timestamped index.

Observe that one call ``ps.run(func, params)`` creates one ``_run_id`` -- a
UUID identifying this run. Inside that, each call ``func(pset)`` creates a
unique ``_pset_id``, a timestamp and a new row in the DataFrame (the database).

Concepts
========

The basic data structure for a param study is a list ``params`` of dicts
(called "parameter sets" or short `pset`).

.. code-block:: python

    params = [{'a': 1, 'b': 'lala'},  # pset 1
              {'a': 2, 'b': 'zzz'},   # pset 2
              ...                     # ...
             ]

Each `pset` contains values of parameters ('a' and 'b') which are varied
during the parameter study.

You need to define a callback function ``func``, which takes exactly one `pset`
such as::

    {'a': 1, 'b': 'lala'}

and runs the workload for that `pset`. ``func`` must return a dict, for example::

    {'result': 1.234}

or an updated `pset`::

    {'a': 1, 'b': 'lala', 'result': 1.234}

We always merge (``dict.update``) the result of ``func`` with the `pset`,
which gives you flexibility in what to return from ``func``.

The `psets` form the rows of a pandas ``DataFrame``, which we use to store
the `pset` and the result from each ``func(pset)``.

The idea is now to run ``func`` in a loop over all `psets` in ``params``. You
do this using the ``ps.run`` helper function. The function adds some special
columns such as ``_run_id`` (once per ``ps.run`` call) or ``_pset_id`` (once
per `pset`). Using ``ps.run(... poolsize=...)`` runs ``func`` in parallel on
``params`` using ``multiprocessing.Pool``.

This package offers some very simple helper functions which assist in creating
``params``. Basically, we define the to-be-varied parameters ('a' and 'b')
and then use something like ``itertools.product`` to loop over them to create
``params``, which is passed to ``ps.run`` to actually perform the loop over all
`psets`.

.. code-block:: python

    >>> from itertools import product
    >>> from psweep import psweep as ps
    >>> x=ps.seq2dicts('x', [1,2,3])
    >>> y=ps.seq2dicts('y', ['xx','yy','zz'])
    >>> x
    [{'x': 1}, {'x': 2}, {'x': 3}]
    >>> y
    [{'y': 'xx'}, {'y': 'yy'}, {'y': 'zz'}]
    >>> ps.loops2params(product(x,y))
    [{'x': 1, 'y': 'xx'},
     {'x': 1, 'y': 'yy'},
     {'x': 1, 'y': 'zz'},
     {'x': 2, 'y': 'xx'},
     {'x': 2, 'y': 'yy'},
     {'x': 2, 'y': 'zz'},
     {'x': 3, 'y': 'xx'},
     {'x': 3, 'y': 'yy'},
     {'x': 3, 'y': 'zz'}]

The logic of the param study is entirely contained in the creation of ``params``.
E.g., if parameters shall be varied together (say x and y), then instead of

.. code-block:: python

    >>> product(x,y,z)

use

.. code-block:: python

    >>> product(zip(x,y), z)

The nesting from ``zip()`` is flattened in ``loops2params()``.

.. code-block:: python

    >>> z=ps.seq2dicts('z', [None, 1.2, 'X'])
    >>> ps.loops2params(product(zip(x,y),z))
    [{'x': 1, 'y': 'xx', 'z': None},
     {'x': 1, 'y': 'xx', 'z': 1.2},
     {'x': 1, 'y': 'xx', 'z': 'X'},
     {'x': 2, 'y': 'yy', 'z': None},
     {'x': 2, 'y': 'yy', 'z': 1.2},
     {'x': 2, 'y': 'yy', 'z': 'X'},
     {'x': 3, 'y': 'zz', 'z': None},
     {'x': 3, 'y': 'zz', 'z': 1.2},
     {'x': 3, 'y': 'zz', 'z': 'X'}]

If you want a parameter which is constant, use a list of length one:

.. code-block:: python

    >>> c=ps.seq2dicts('c', ['const'])
    >>> ps.loops2params(product(zip(x,y),z,c))
    [{'a': 1, 'c': 'const', 'y': 'xx', 'z': None},
     {'a': 1, 'c': 'const', 'y': 'xx', 'z': 1.2},
     {'a': 1, 'c': 'const', 'y': 'xx', 'z': 'X'},
     {'a': 2, 'c': 'const', 'y': 'yy', 'z': None},
     {'a': 2, 'c': 'const', 'y': 'yy', 'z': 1.2},
     {'a': 2, 'c': 'const', 'y': 'yy', 'z': 'X'},
     {'a': 3, 'c': 'const', 'y': 'zz', 'z': None},
     {'a': 3, 'c': 'const', 'y': 'zz', 'z': 1.2},
     {'a': 3, 'c': 'const', 'y': 'zz', 'z': 'X'}]

So, as you can see, the general idea is that we do all the loops *before*
running any workload, i.e. we assemble the parameter grid to be sampled before
the actual calculations. This has proven to be very practical as it helps
detecting errors early.

You are, by the way, of course not restricted to use ``itertools.product``. You
can use any complicated manual loop you can come up with. The point is: you
generate ``params``, we run the study.


_pset_id, _run_id and repeated runs
-----------------------------------

See ``examples/vary_2_params_repeat.py``.

It is important to get the difference between the two special fields
``_run_id`` and ``_pset_id``, the most important one being ``_pset_id``.

Both are random UUIDs. They are used to uniquely identify things.

By default, ``ps.run()`` writes a database ``calc/results.pk`` (a pickled
DataFrame) with the default ``calc_dir='calc'``. If you run ``ps.run()``
again

.. code-block:: python

    df = ps.run(func, params)
    df = ps.run(func, other_params)

it will read and append to that file. The same happens in an interactive
session when you pass in ``df`` again:

.. code-block:: python

    df = ps.run(func, params) # default is df=None -> create empty df
    df = ps.run(func, other_params, df=df)


Once per ``ps.run`` call, a ``_run_id`` is created. Which means that when you
call ``ps.run`` multiple times *using the same database* as just shown, you
will see multiple (in this case two) ``_run_id`` values.

::

    _run_id                               _pset_id
    afa03dab-071e-472d-a396-37096580bfee  21d2185d-b900-44b3-a98d-4b8866776a77
    afa03dab-071e-472d-a396-37096580bfee  3f63742b-6457-46c2-8ed3-9513fe166562
    afa03dab-071e-472d-a396-37096580bfee  1a812d67-0ffc-4ab1-b4bb-ad9454f91050
    afa03dab-071e-472d-a396-37096580bfee  995f5b0b-f9a6-45ee-b4d1-5784a25be4c6
    e813db52-7fb9-4777-a4c8-2ce0dddc283c  7b5d8f76-926c-44e2-a0e3-2e68deb86abb
    e813db52-7fb9-4777-a4c8-2ce0dddc283c  f46bb714-4677-4a11-b371-dd2d41a83d19
    e813db52-7fb9-4777-a4c8-2ce0dddc283c  5fdcc88b-d467-4117-aa03-fd256656299b
    e813db52-7fb9-4777-a4c8-2ce0dddc283c  8c5c07ca-3862-4726-a7d0-15d60e281407

Each ``ps.run`` call in turn calls ``func(pset)`` for each `pset` in
``params``. Each ``func`` invocation created a unique ``_pset_id``. Thus, we
have a very simple, yet powerful one-to-one mapping and a way to refer to a
specific `pset`.


Best practices
==============

The following workflows and practices come from experience. They are, if you
will, the "framework" for how to do things. However, we decided to not codify
any of these ideas but to only provide tools to make them happen easily,
because you will probably have quite different requirements and workflows.

Please also have a look at the ``examples/`` dir where we document these and
more common workflows.

Save data on disk, use UUIDs
----------------------------

See ``examples/save_data_on_disk.py``.

Assume that you need to save results from a run not only in the returned dict
from ``func`` (or even not at all!) but on disk, for instance when you call an
external program which saves data on disk. Consider this example:

.. code-block:: python

    import os
    import subprocess
    from psweep import psweep as ps


    def func(pset):
        fn = os.path.join(pset['_calc_dir'],
                          pset['_pset_id'],
                          'output.txt')
        cmd = "mkdir -p $(dirname {fn}); echo {a} > {fn}".format(a=pset['a'],
                                                                 fn=fn)
        pset['cmd'] = cmd
        subprocess.run(cmd, shell=True)
        return pset


In this case, you call an external program (here a dummy shell command) which
saves its output on disk. Note that we don't return any output from the
external command from ``func``. We only update ``pset`` with the shell ``cmd``
we call to have that in the database.

Also note how we use the special fields ``_pset_id`` and ``_calc_dir``, which
are added in ``ps.run`` to ``pset`` *before* ``func`` is called.

After the run, we have four dirs for each `pset`, each simply named with
``_pset_id``::

    calc
    ├── 63b5daae-1b37-47e9-a11c-463fb4934d14
    │   └── output.txt
    ├── 657cb9f9-8720-4d4c-8ff1-d7ddc7897700
    │   └── output.txt
    ├── d7849792-622d-4479-aec6-329ed8bedd9b
    │   └── output.txt
    ├── de8ac159-b5d0-4df6-9e4b-22ebf78bf9b0
    │   └── output.txt
    └── results.pk

This is a useful pattern. History has shown that in the end, most naming
conventions start simple but turn out to be inflexible and hard to adapt later
on. I have seen people write scripts which create things
like::

    calc/param_a=1.2_param_b=66.77
    calc/param_a=3.4_param_b=88.99

i.e. encode the parameter values in path names, because they don't have a
database. Good luck parsing that. I don't say this cannot be done -- sure it
can (in fact the example above easy to parse). It is just not fun -- and there
is no need to. What if you need to add a "column" for parameter 'c' later?
Impossible (well, painful at least). This approach makes sense for very quick
throw-away test runs, but gets out of hand quickly.

Since we have a database, we can simply drop all data in ``calc/<_pset_id>``
and be done with it. Each parameter set is identified by a UUID that will never
change. This is the only kind of naming convention which makes sense in the
long run.


Iterative extension of a parameter study
----------------------------------------

See ``examples/{10,20}multiple_1d_scans_with_backup.py``.

We recommend to always use `backup_calc_dir`:

.. code-block:: python

    df = ps.run(func, params, backup_calc_dir=True)

`backup_calc_dir` will save a copy of the old
`calc_dir` to ``calc_<last_date_in_old_database>``, i.e. something like
``calc_2018-09-06T20:22:27.845008Z`` before doing anything else. That way, you
can track old states of the overall study, and recover from mistakes.

For any non-trivial work, you won't use an interactive session.
Instead, you will have a driver script which defines ``params`` and starts
``ps.run()``. Also in a common workflow, you won't define ``params`` and run a
study once. Instead you will first have an idea about which parameter values to
scan. You will start with a coarse grid of parameters and then inspect the
results and identify regions where you need more data (e.g. more dense
sampling). Then you will modify ``params`` and run the study again. You will
modify the driver script multiple times, as you refine your study. To save the
old states of that script, use `backup_script`:

.. code-block:: python

    df = ps.run(func, params, backup_calc_dir=True, backup_script=__file__)

`backup_script` will save a copy of the script which you use to drive your study
to ``calc/backup_script/<_run_id>.py``. Since each ``ps.run()`` will create a new
``_run_id``, you will have a backup of the code which produced your results for
this ``_run_id`` (without putting everything in a git repo, which may be
unpleasant if your study produces large amounts of data).

Simulate / Dry-Run: look before you leap
----------------------------------------

See ``examples/vary_1_param_simulate.py``.

When you fiddle with finding the next good ``params`` and even when using
`backup_calc_dir`, appending to the old database might be a hassle if you find
that you made a mistake when setting up ``params``. You need to abort the
current run, delete
`calc_dir` and copy the last backup back:

.. code-block:: sh

   $ rm -r calc
   $ mv calc_2018-09-06T20:22:27.845008Z calc

Instead, while you tinker with ``params``, use another `calc_dir`, e.g.

.. code-block:: python

    df = ps.run(func, params, calc_dir='calc_test')

But what's even better: keep everything as it is and just set ``simulate=True``

.. code-block:: python

    df = ps.run(func, params, backup_calc_dir=True, backup_script=__file__,
                simulate=True)

This will copy only the database, not all the (possible large) data in
``calc/`` to ``calc.simulate/`` and run the study there, but w/o actually
calling ``func()``. So you still append to your old database as in a real run,
but in a safe separate dir which you can delete later.


Give runs names for easy post-processing
----------------------------------------

See ``examples/vary_1_param_study_column.py``.

Post-processing is not the scope of the package. The database is a DataFrame
and that's it. You can query it and use your full pandas Ninja skills here
(e.g. "give me all psets where parameter 'a' was between 10 and 100, while 'b'
was constant, which were run last week and the result was not < 0" ... you get
the idea.

To ease post-processing, it is useful practice to add a constant parameter
named e.g. "study" or "scan" to label a certain range of runs. If you, for
instance, have 5 runs where you scan values for parameter 'a' while keeping
parameters 'b' and 'c' constant, you'll have 5 ``_run_id`` values. When
querying the database later, you could limit by ``_run_id`` if you know the
values:

.. code-block:: python

    >>> df = df[(df._run_id=='afa03dab-071e-472d-a396-37096580bfee') |
                (df._run_id=='e813db52-7fb9-4777-a4c8-2ce0dddc283c') |
                ...
                ]

This doesn't look like fun. It shows that the UUIDs (``_run_id`` and
``_pset_id``) are rarely ment to be used directly. Instead, you should (in this
example) limit by the constant values of the other parameters:

.. code-block:: python

    >>> df = df[(df.b==10) & (df.c=='foo')]

Much better! This is what most post-processing scripts will do.

But when you have a column "study" which has the value ``'a'`` all the time, it
is just

.. code-block:: python

    >>> df = df[df.study=='a']

You can do more powerful things with this approach. For instance, say you vary
parameters 'a' and 'b', then you could name the "study" field 'fine_scan=a:b'
and encode which parameters (thus column names) you have varied. Later in the
post-processing

.. code-block:: python

    >>> study = 'fine_scan=a:b'
    # cols = ['a', 'b']
    >>> cols = study.split('=')[1].split(':')
    >>> values = df[cols].values

So in this case, a naming convention *is* useful in order to bypass possibly
complex database queries. But it is still flexible -- you can change the
"study" column at any time, or delete it again.

Pro tip: You can manipulate the database at any later point and add the "study"
column after all runs have been done.

Super Pro tip: Make a backup of the database first!


Install
=======

::

    $ pip3 install psweep


Dev install of this repo::

    $ pip3 install -e .

See also https://github.com/elcorto/samplepkg.

Tests
=====

::

    # apt-get install python3-nose
    $ nosetests3
