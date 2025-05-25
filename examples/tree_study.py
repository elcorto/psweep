#!/usr/bin/env python3

"""
Build a "tree" of psets by doing leave calculations (study 2: vary parameter
c), based on each pset in study 1 ("nodes", vary a and b). This is useful if
calculations for psets of study 1 are costly and should not be repeated for
each study 2 pset with params a,b,c.

We reuse result_1_ from study 1 as param result_1 in study 2.

df_1
a b  result_1_
1 8   1.430288
1 9   3.148494
2 8  15.298595
2 9   3.704101

df_2
a b  c   result_1   result_2_
1 8 77   1.430288   78.430288
1 8 88   1.430288   89.430288
1 9 77   3.148494   80.148494
1 9 88   3.148494   91.148494
2 8 77  15.298595   92.298595
2 8 88  15.298595  103.298595
2 9 77   3.704101   80.704101
2 9 88   3.704101   91.704101

We use run(..., save=False) to avoid writing and auto-reading calc/database.pk,
as we would want in an append-to-database type setting (repeat/extend study).
Here we have *two* studies which have their own database each, but we run all
in one directory for simplicity.
"""

import random
import psweep as ps


def func_1(pset):
    """First study function. Assumed to be a costly calculation."""
    return {"result_1_": random.random() * pset["a"] * pset["b"]}


def func_2(pset):
    """Second study function. For each row in df_1, do some calculation using
    result_1_ and a value of "c" defined in the 2nd study.
    """
    return {"result_2_": pset["result_1"] + pset["c"]}


if __name__ == "__main__":
    params = ps.pgrid(
        ps.plist("a", [1, 2]),
        ps.plist("b", [8, 9]),
    )
    # First run.
    df_1 = ps.run(func_1, params, save=False)
    print("df_1")
    ps.df_print(df_1)

    # Show how to extract the varied params easily from a df. Here of course we
    # could use "params" directly, but we show how to use params_from_df()
    # since in a real world setting, calculations won't usually live in one
    # script, but instead read in df_1 from disk, for instance.
    params_1 = ps.params_from_df(df_1)
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
    # trick, we zip() together params_1 and result_1_, since params_from_df()
    # extracts only pset-type columns, not prefix and postfix ones.
    params = ps.pgrid(
        zip(params_1, ps.plist("result_1", df_1.result_1_.values)),
        ps.plist("c", [77, 88]),
    )
    df_2 = ps.run(func_2, params, save=False)

    assert set(df_2.result_1.values) == set(df_1.result_1_.values)
    assert (df_2.result_2_ == df_2.result_1 + df_2.c).all()

    print("\ndf_2")
    ps.df_print(df_2)
