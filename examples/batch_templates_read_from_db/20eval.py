#!/usr/bin/env python3

import numpy as np
import psweep as ps

if __name__ == "__main__":
    df = ps.df_read("calc/database.pk")

    arr = np.array(
        [np.load(f"calc/{pset_id}/out.npy") for pset_id in df._pset_id.values]
    )

    df["mean_"] = arr.mean(axis=1)

    cols = ["param_a", "param_b", "mean_"]
    ps.df_print(df[cols])

    ps.df_write("calc/database_eval.pk", df)
