import numpy as np

import psweep as ps

# Read pset for the pset_id of this dir (calc/<pset_id>) from database. Use in
# code below.
pset = ps.df_extract_pset(ps.df_read("../database.pk"), "{{_pset_id}}")
print(f"Processing pset for pset_id={{_pset_id}}:\n{pset}")

result = np.array([pset["param_a"], pset["param_b"]]) * 3.21
np.save("out.npy", result)
