#!/usr/bin/env python3

"""
We want to vary params a,b,c. The normal way to do this is to loop over all
values of a,b and c using pgrid(), which would give us these psets

├─ a=1 b=8 c=77
├─ a=1 b=8 c=88
├─ a=1 b=9 c=77
├─ a=1 b=9 c=88
├─ a=2 b=8 c=77
├─ a=2 b=8 c=88
├─ a=2 b=9 c=77
└─ a=2 b=9 c=88

and this result database

df
a b  c   result_ab_  result_abc_
1 8 77     1.430288    78.430288
1 8 88     1.430288    89.430288
1 9 77     3.148494    80.148494
1 9 88     3.148494    91.148494
2 8 77    15.298595    92.298595
2 8 88    15.298595   103.298595
2 9 77     3.704101    80.704101
2 9 88     3.704101    91.704101

But now suppose that to calculate result_abc_, we fist need to perform some
costly calculation with a and b, producing

    result_ab_ = func_ab(a,b)

We store that and do a final calculation with c to produce

    result_abc_ = func_abc(result_ab_, c)

In the case above we'd repeat func_ab() for each value of c, which is
suboptimal when func_ab() is slow.

Instead, we can build a "tree" of psets by doing leave calculations (study 2:
vary parameter c), based on each pset in study 1 ("nodes", vary a and b).

We reuse result_ab_ from study 1 as param result_ab in study 2. Here is the tree:

├─ a=1 b=8
│  ├─ c=77
│  └─ c=88
├─ a=1 b=9
│  ├─ c=77
│  └─ c=88
├─ a=2 b=8
│  ├─ c=77
│  └─ c=88
└─ a=2 b=9
   ├─ c=77
   └─ c=88

and here the corresponding databases:

df_1
a b  result_ab_
1 8    1.430288
1 9    3.148494
2 8   15.298595
2 9    3.704101

df_2
a b  c   result_ab_   result_abc_
1 8 77     1.430288     78.430288
1 8 88     1.430288     89.430288
1 9 77     3.148494     80.148494
1 9 88     3.148494     91.148494
2 8 77    15.298595     92.298595
2 8 88    15.298595    103.298595
2 9 77     3.704101     80.704101
2 9 88     3.704101     91.704101

We use run(..., save=False) to avoid writing and auto-reading calc/database.pk,
as we would want in an append-to-database type setting (repeat/extend study).
Here we have *two* studies which have their own database each, but we run all
in one directory for simplicity.
"""

import random
from pprint import pprint

import psweep as ps


def func_ab(pset):
    """First study function. Assumed to be a costly calculation."""
    return {"result_ab_": random.random() * pset["a"] * pset["b"]}


def func_abc(pset):
    """Second study function. For each row in df_1, do some calculation using
    result_ab_ and a value of "c" defined in the 2nd study.
    """
    return {"result_abc_": pset["result_ab"] + pset["c"]}


if __name__ == "__main__":
    params = ps.pgrid(
        ps.plist("a", [1, 2]),
        ps.plist("b", [8, 9]),
    )
    print("study 1 params:")
    pprint(params)
    # First run.
    df_1 = ps.run(func_ab, params, save=False)
    print("\ndf_1")
    ps.df_print(df_1)

    # Show how to extract the varied params easily from a df. Here of course we
    # could use "params" directly, but we show how to use df_extract_params()
    # since in a real world setting, calculations won't usually live in one
    # script, but instead read in df_1 from disk, for instance.
    params_1 = ps.df_extract_params(df_1)
    assert params_1 == params

    # Second run. Here we show an interesting case of using pgrid(). Usually we
    # give it plists
    #
    #   plist("a", [1, 2]) -> [{'a': 1}, {'a': 2}]
    #   plist("b", [8, 9]) -> [{'b': 8}, {'b': 9}]
    #
    # to loop over. In the plist case, each dict has length 1. But plists are
    # just lists of dicts and pgrid() doesn't care where they come from and
    # indeed how *long* the dicts are. We use that to link up params_1 from
    # study 1 and plist("c", [77, 88]) in study 2. Cool, eh? As an additional
    # trick, we zip() together params_1 and result_ab_, since df_extract_params()
    # extracts only pset-type columns, not prefix and postfix ones.
    params = ps.pgrid(
        zip(params_1, ps.plist("result_ab", df_1.result_ab_.values)),
        ps.plist("c", [77, 88]),
    )
    print("\nstudy 2 params:")
    pprint(params)
    df_2 = ps.run(func_abc, params, save=False)

    assert set(df_2.result_ab.values) == set(df_1.result_ab_.values)
    assert (df_2.result_abc_ == df_2.result_ab + df_2.c).all()

    print("\ndf_2")
    ps.df_print(df_2)
