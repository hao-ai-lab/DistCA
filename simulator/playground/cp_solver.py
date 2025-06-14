# %%
from ortools.sat.python import cp_model

# %%
m = cp_model.CpModel()
x = m.NewBoolVar("x")
y = m.NewBoolVar("y")
z = m.NewBoolVar("z")

# %%
# z = x XOR y
m.AddBoolXOr([x, y, z])
m.Maximize(z)         # objective just so the solver does some work

# %%
solver = cp_model.CpSolver()
solver.parameters.num_search_workers = 8  # 0 → all cores
solver.Solve(m)

# %%
print("x y z =", solver.Value(x), solver.Value(y), solver.Value(z))
# %%
