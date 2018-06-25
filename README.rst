psweep -- loop like a pro, make parameter studies fun
=====================================================

About
-----

This is a package with simple helpers to set up parameter studies.

Getting started
---------------

The most simple example one can think of: loop over a single variable.

.. code-block:: python

    #!/usr/bin/env python3

    import random
    from psweep import psweep as ps
    import pandas as pd


    def func(pset):
        """pset: dict such as {'a': 1}"""
        return {'timing': random.random() * pset['a']}
                
                    
    if __name__ == '__main__':
        # params = [{'a': 1}, {'a': 2}, {'a': 3}, {'a': 4}]
        params = ps.seq2dicts('a', [1,2,3,4])
        df = pd.DataFrame()
        df = ps.run(df, func, params)
        print(df.to_json(orient='split'))


prints::

       _run  a    timing
    0     0  1  0.034164
       _run  a    timing
    1     0  2  1.685145
       _run  a    timing
    2     0  3  1.612341
       _run  a    timing
    3     0  4  2.887937
    {"columns":["_run","a","timing"],"index":[0,1,2,3],"data":[[0,1,0.0341638266],[0,2,1.6851449492],[0,3,1.6123406636],[0,4,2.8879373119]]}

Tests
-----

::
    
    # apt-get install python3-nose
    $ nosetests3

Concepts
--------

The basic data structure for a param study is a list ``params`` of dicts
(called "parameter sets" or short psets).

.. code-block:: python

    params = [{'foo': 1, 'bar': 'lala'},  # pset 1
              {'foo': 2, 'bar': 'zzz'},   # pset 2
              ...                         # ...
             ]

Each pset contains values of parameters ('foo' and 'bar') which are varied
during the parameter study.

These psets are the basis of a pandas ``DataFrame`` (much like an SQL table, 2D
array w/ named columns and in case of ``DataFrame`` also variable data types)
with columns 'foo' and 'bar'. 

Then we define a callback function ``func``, which takes only one pset
such as::

    {'foo': 1, 'bar': 'lala'},

and runs the workload for that pset. ``func`` must return a dict, for example::

    {'timing': 1.234}, 

which is the result of the run.

``func`` is called in a loop on all psets in ``params`` in the ``run`` helper
function. The result dict (e.g. ``{'timing': 1.234}`` from each call gets merged
with the current pset such that we have::

    {'foo': 1, 'bar': 'lala', 'timing': 1.234}

That gets appended to a ``DataFrame``, thus creating a new column called
'timing'. The ``run`` function adds a ``_run`` column as well, which counts how
often the study has been performed.

This package offers some very simple helper functions which assist in creating
``params``. Basically, we define the to-be-varied parameters ('foo' and 'bar')
as "named sequences" (i.e. list of dicts) which are, in fact, the columns of
``params``. Then we use something like ``itertools.product`` to loop over them.

.. code-block:: python

    >>> from itertools import product
    >>> x=seq2dicts('a', [1,2,3])
    >>> x
    [{'x': 1}, {'x': 2}, {'x': 3}]
    >>> y=seq2dicts('y', ['xx','yy','zz'])
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

    >>> loops2params(product(x,y))
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

    >>> z=seq2dicts('z', [None, 1.2, 'X'])
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

    >>> loops2params(product(zip(x,y),z))
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

    >>> c=seq2dicts('c', ['const'])
    >>> c
    [{'c': 'const'}]
    >>> loops2params(product(zip(x,y),z,c))
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

You may have noticed that the data structures and functions used here are so
simple that is almost not worth a package at all, but it is helpful to have the
ideas and the workflow packaged up in a central place.

Install
-------

::

    $ pip3 install psweep


Dev install of this repo::

    $ pip3 install -e .

See also https://github.com/elcorto/samplepkg.
