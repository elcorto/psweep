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

You see a number of reserved fields for book-keeping such as

::

    _run_id
    _pset_id
    _calc_dir
    _time_utc

and a timestamped index. See the ``examples`` dir for more.

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
