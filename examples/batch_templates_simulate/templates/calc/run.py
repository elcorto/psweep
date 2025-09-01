import numpy as np

# dollar syntax
##print("hello from run_id=$_run_id pset_id=$_pset_id")
##print("param_a: $param_a")
##print("param_b: $param_b")

# jinja (recommended)
print("hello from run_id={{_run_id}} pset_id={{_pset_id}}")
print("param_a: {{param_a}}")
print("param_b: {{param_b}}")

result = np.array([{{param_a}}, {{param_b}}]) * 3.21  # noqa: F821
np.save("out.npy", result)
