
import numpy as np

print("hello from run_id=$_run_id pset_id=$_pset_id")

print("param_a: $param_a")
print("param_b: $param_b")

result = np.array([$param_a, $param_b]) * 3.21
np.save("out.npy", result)
