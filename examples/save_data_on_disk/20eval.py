#!/usr/bin/env python3

import numpy as np
import psweep as ps


if __name__ == "__main__":
    df = ps.df_read("calc/database.pk")

    # Filter database
    df = df[df.a > 0 & ~df.a.isna()]

    arr = np.array(
        [
            np.loadtxt(f"calc/{pset_id}/output.txt")
            for pset_id in df._pset_id.values
        ]
    )

    df["mean_"] = arr.mean(axis=1)

    cols = ["a", "mean_", "_pset_id"]
    ps.df_print(df[cols])

    ps.df_write("calc/database_eval.pk", df)
