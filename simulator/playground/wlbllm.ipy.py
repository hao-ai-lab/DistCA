# %%
"""
WLB-LLM ILP Problem Solution

Problem:

    minimize max_j sum( (Wa[d_i] +  Wl[d_i]) * (x_ij * d_i) )

subject to
- sum_j(x_ij) = 1 for all i
- sum_i(x_ij * d_i) <= Lmax
- x_ij \in {0, 1} for all i, j
- i = 1 ... N = number of documents
- j = 1 ... M = number of workers
- Lmax = 16*K = maximum length of the document
"""

import pulp

def attn_time(x: int) -> float:
    """Calculate the attention time for a given document length.

    Args:
        x (int): The length of the document in tokens.

    Returns:
        float: The attention computation time in milliseconds.
    """

    return x ** 2
    pass

def mlp_time(x: int) -> float:
    """Calculate the MLP time for a given document length.

    Args:
        x (int): The length of the document in tokens.

    Returns:
        float: The MLP computation time in milliseconds.
    """
    return x



def wlbllm_ilp_solver(
    doc_lengths: list[int],
    max_length: int,
    max_micro_batches: int,
):
    """Solve the WLB-LLM ILP problem.

    Args:
        doc_lengths (list[int]): The lengths of the documents.
        max_length (int): The maximum length of the document.
        max_micro_batches (int): The maximum number of workers.
    
    Returns:
        pulp.LpProblem: The problem object.
    """
    # Create the problem
    problem = pulp.LpProblem("WLB-LLM ILP Problem", pulp.LpMinimize)

    # Create the decision variables
    x = pulp.LpVariable.dicts("x", (range(len(doc_lengths)), range(max_micro_batches)), cat="Binary")
    T = pulp.LpVariable("T", lowBound=0)

    # Create the objective function
    problem += T
        
    # Objective function
    for j in range(max_micro_batches):
        a = pulp.lpSum(x[i][j] * (attn_time(doc_lengths[i]) + mlp_time(doc_lengths[i])) for i in range(len(doc_lengths)))
        problem += a <= T, f"A_{j}"
        pass

    # Subject to sum(x_ij) = 1 for all i
    for i in range(len(doc_lengths)):
        X_i = pulp.lpSum(x[i][j] for j in range(max_micro_batches))
        problem += X_i == 1, f"X_{i}"

    # Subject to sum(x_ij * d_i) <= Lmax for all j
    for j in range(max_micro_batches):
        C_j = pulp.lpSum(x[i][j] * doc_lengths[i] for i in range(len(doc_lengths)))
        problem += C_j <= max_length, f"C_{j}"

    problem.solve()

    
    xs = dict()
    for i in range(len(doc_lengths)):
        for j in range(max_micro_batches):
            if x[i][j].varValue == 1:
                xs[i] = j

    return problem, xs, pulp.value(T)



def test_wlbllm_ilp_solver():
    doc_lengths = [1, 2, 3, 4]
    max_length = 16 # = Lmax
    max_micro_batches = 4 # = M
    problem, xs, T = wlbllm_ilp_solver(doc_lengths, max_length, max_micro_batches)
    batches = []
    for j in range(max_micro_batches):  
        batches.append([])
    for i in range(len(doc_lengths)):
        batches[xs[i]].append(doc_lengths[i])
    
    print(f"doc_lengths: {doc_lengths}")
    print(f"T: {T}")
    print(f"xs: {xs}")
    print(f"batches: {batches}")
    return batches, T



# %%
if __name__ == "__main__":
    test_wlbllm_ilp_solver()
# %%
