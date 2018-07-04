psweep -- loop like a pro, make parameter studies fun
=====================================================

About
-----

This is a package with simple helpers to set up and run parameter studies.

Getting started
---------------

Loop over two parameters 'a' and 'b':

.. code-block:: python

    #!/usr/bin/env python3

    import random
    from itertools import product
    from psweep import psweep as ps


    def func(pset):
        return {'result': random.random() * pset['a'] * pset['b']}


    if __name__ == '__main__':
        params = ps.loops2params(product(
                ps.seq2dicts('a', [1,2,3,4]),
                ps.seq2dicts('b', [8,9]),
                ))
        df = ps.run(func, params)
        print(df)

produces a list of parameter sets to loop over (``params``)::

    [{'a': 1, 'b': 8},
     {'a': 1, 'b': 9},
     {'a': 2, 'b': 8},
     {'a': 2, 'b': 9},
     {'a': 3, 'b': 8},
     {'a': 3, 'b': 9},
     {'a': 4, 'b': 8},
     {'a': 4, 'b': 9}]


and a database of results (pandas DataFrame ``df``, file ``calc/results.json``
by default)::

      _calc_dir                              _pset_id  \
    0      calc  e5554177-ce31-4944-93ee-786dbaadacb7
    1      calc  c055661c-fc36-476b-b50a-ae5c101ce638
    2      calc  9f316933-5d46-42a7-aca8-14ccf3555ccc
    3      calc  9aff8f32-402b-4f3a-9040-449a8a7e23c6
    4      calc  0cf2a7f1-a5d6-4a23-8a9c-c14766b5d450
    5      calc  268ba704-8c32-4bd8-8006-59bc3bd3c234
    6      calc  c1732939-1668-4654-bb4f-8ad8c8391ef8
    7      calc  9f79b241-0ef1-408c-a538-dc588a11a0de

                                    _run_id  a  b      result
    0  f467d1f5-3db1-4fcb-8a3c-a9bb5ac18f4c  1  8     5.95035
    1  f467d1f5-3db1-4fcb-8a3c-a9bb5ac18f4c  1  9     3.74252
    2  f467d1f5-3db1-4fcb-8a3c-a9bb5ac18f4c  2  8     2.58442
    3  f467d1f5-3db1-4fcb-8a3c-a9bb5ac18f4c  2  9  0.00564436
    4  f467d1f5-3db1-4fcb-8a3c-a9bb5ac18f4c  3  8     15.9873
    5  f467d1f5-3db1-4fcb-8a3c-a9bb5ac18f4c  3  9     19.2371
    6  f467d1f5-3db1-4fcb-8a3c-a9bb5ac18f4c  4  8     17.0561
    7  f467d1f5-3db1-4fcb-8a3c-a9bb5ac18f4c  4  9     21.0376

See the ``examples`` dir for more.

Tests
-----

::

    # apt-get install python3-nose
    $ nosetests3

Concepts
--------

The basic data structure for a param study is a list ``params`` of dicts
(called "parameter sets" or short `pset`).

.. code-block:: python

    params = [{'a': 1, 'b': 'lala'},  # pset 1
              {'a': 2, 'b': 'zzz'},   # pset 2
              ...                         # ...
             ]

Each `pset` contains values of parameters ('a' and 'b') which are varied
during the parameter study.

These `psets` are the basis of a pandas ``DataFrame`` (much like an SQL table, 2D
array w/ named columns and in case of ``DataFrame`` also variable data types)
with columns 'a' and 'b'.

You only need to define a callback function ``func``, which takes exactly one `pset`
such as::

    {'a': 1, 'b': 'lala'}

and runs the workload for that `pset`. ``func`` must return a dict, for example::

    {'result': 1.234}

or an updated `pset`::
    
    {'a': 1, 'b': 'lala', 'result': 1.234}

which is the result of the run.

``func`` is called in a loop on all `psets` in ``params`` in the ``ps.run`` helper
function. The result dict (e.g. ``{'result': 1.234}`` from each call gets merged
with the current `pset` such that we have::

    {'a': 1, 'b': 'lala', 'result': 1.234}

That gets appended to a ``DataFrame``, thus creating a new column called
'result'. The ``ps.run`` function adds some special columns such as ``_run_id``
(once per ``ps.run`` call) or ``_pset_id`` (once per `pset`). Using ``ps.run(...
poolsize=...)`` runs ``func`` in parallel on ``params`` using
``multiprocessing.Pool``.

This package offers some very simple helper functions which assist in creating
``params``. Basically, we define the to-be-varied parameters ('a' and 'b')
and then use something like ``itertools.product`` to loop over them to create
``params``, which is passed to ``ps.run`` to actually perform the loop over all
`psets`.

.. code-block:: python

    >>> from itertools import product
    >>> from psweep import psweep as ps
    >>> x=ps.seq2dicts('a', [1,2,3])
    >>> x
    [{'x': 1}, {'x': 2}, {'x': 3}]
    >>> y=ps.seq2dicts('y', ['xx','yy','zz'])
    >>> y
    [{'y': 'xx'}, {'y': 'yy'}, {'y': 'zz'}]
    >>> list(product(x,y))
    [({'x': 1}, {'y': 'xx'}),
     ({'x': 1}, {'y': 'yy'}),
     ({'x': 1}, {'y': 'zz'}),
     ({'x': 2}, {'y': 'xx'}),
     ({'x': 2}, {'y': 'yy'}),
     ({'x': 2}, {'y': 'zz'}),
     ({'x': 3}, {'y': 'xx'}),
     ({'x': 3}, {'y': 'yy'}),
     ({'x': 3}, {'y': 'zz'})]

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

The nestings from ``zip()`` are flattened in ``loops2params()``.

.. code-block:: python

    >>> z=ps.seq2dicts('z', [None, 1.2, 'X'])
    >>> z
    [{'z': None}, {'z': 1.2}, {'z': 'X'}]
    >>> list(product(zip(x,y),z))
    [(({'x': 1}, {'y': 'xx'}), {'z': None}),
     (({'x': 1}, {'y': 'xx'}), {'z': 1.2}),
     (({'x': 1}, {'y': 'xx'}), {'z': 'X'}),
     (({'x': 2}, {'y': 'yy'}), {'z': None}),
     (({'x': 2}, {'y': 'yy'}), {'z': 1.2}),
     (({'x': 2}, {'y': 'yy'}), {'z': 'X'}),
     (({'x': 3}, {'y': 'zz'}), {'z': None}),
     (({'x': 3}, {'y': 'zz'}), {'z': 1.2}),
     (({'x': 3}, {'y': 'zz'}), {'z': 'X'})]

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

If you want a parameter which is constant, use a length one list and put it in
the loops:

.. code-block:: python

    >>> c=ps.seq2dicts('c', ['const'])
    >>> c
    [{'c': 'const'}]
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

We are aware of the fact that the data structures and functions used here are
so simple that it is almost not worth a package at all, but it is helpful to
have the ideas and the workflow packaged up in a central place.

Install
-------

::

    $ pip3 install psweep


Dev install of this repo::

    $ pip3 install -e .

See also https://github.com/elcorto/samplepkg.
