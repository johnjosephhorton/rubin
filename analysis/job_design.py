"""
dp_idiosyncratic_handoff.py

Solve a sequence-partitioning problem where each block (i..j) has a cost:
   (sum(C[i..j])) * (sum(t[i..j]))
and each new block starting at index m > 1 incurs a hand-off cost h[m].

We also reconstruct the partition (list of blocks) and the wage for each block.
"""


def solve_dp(C, t, h):
    """
    Solve for the minimal total cost using dynamic programming, and reconstruct the partition.

    Parameters
    ----------
    C : list or array of length T
        Training costs for tasks 1..T (0-based in Python).
    t : list or array of length T
        Times for tasks 1..T (0-based in Python).
    h : list or array of length T
        Idiosyncratic hand-off cost if we start a new block at task i (1-based),
        i.e., if the block starts at index i in 1-based indexing (or i-1 in 0-based),
        we pay h[i-1] (in code) as a hand-off.
        For convenience: h[i] corresponds to starting a block at i+1 in 1-based.

    Returns
    -------
    dp_value : float
        The minimal total cost to complete tasks 1..T under the given model.
    blocks : list of tuples (start, end, wage)
        The partition of tasks into blocks. Each tuple indicates:
          - start : int (1-based index of first task in block)
          - end   : int (1-based index of last task in block)
          - wage  : sum(C_k for k in block)
        in the order they are executed.
    """
    # Number of tasks
    T = len(C)

    # We'll use 1-based indexing internally, so pad at front:
    C_ = [0] + C
    t_ = [0] + t
    h_ = [0] + h  # h_[i] = cost if we start a block at task i (1-based)

    # 1) Build prefix sums for C and t for fast block cost calculation
    #    S_C[i] = sum of C_ up to index i
    #    S_t[i] = sum of t_ up to index i
    S_C = [0] * (T + 1)
    S_t = [0] * (T + 1)
    for i in range(1, T + 1):
        S_C[i] = S_C[i - 1] + C_[i]
        S_t[i] = S_t[i - 1] + t_[i]

    # Helper function: block i..j cost
    def block_cost(i, j):
        sumC = S_C[j] - S_C[i - 1]
        sumT = S_t[j] - S_t[i - 1]
        return sumC * sumT

    # 2) dp[i] = minimal cost to complete tasks i..T
    #    We'll also store choice[i] = j that achieves the minimum
    dp = [0] * (T + 2)
    choice = [0] * (T + 2)

    # Base case
    dp[T + 1] = 0  # no tasks => cost 0

    # 3) Fill dp[] from T down to 1
    for i in range(T, 0, -1):
        best_cost = float("inf")
        best_j = i
        # Try all possible j
        for j in range(i, T + 1):
            cost_ij = block_cost(i, j)

            # If j < T, we pay a hand-off cost if we continue
            if j < T:
                candidate = cost_ij + h_[j + 1] + dp[j + 1]
            else:
                # j == T => no further hand-off
                candidate = cost_ij

            if candidate < best_cost:
                best_cost = candidate
                best_j = j

        dp[i] = best_cost
        choice[i] = best_j

    # 4) dp[1] is the minimal total cost. Reconstruct the partition:
    blocks = []
    current = 1
    while current <= T:
        j = choice[current]  # endpoint of the block
        # wage for tasks current..j is sum of C
        wage = S_C[j] - S_C[current - 1]
        blocks.append((current, j, wage))
        current = j + 1  # move to the next block

    # Return the minimal cost and the partition
    return dp[1], blocks


def main():
    """
    Example usage of solve_dp with tasks 1..5.
    """
    # Example data
    # Let T=5
    C = [3, 1, 2, 2, 5]  # training costs
    t = [1, 4, 2, 1, 3]  # times
    h = [
        7 * x for x in [0.5, 2, 0, 1, 5]
    ]  # if we start at task i(1-based), cost is h[i-1]

    total_cost, partition = solve_dp(C, t, h)
    print(f"Minimal total cost = {total_cost:.2f}")
    print("Partition (start, end, wage):")
    for start, end, wage in partition:
        print(f"  - Tasks {start}..{end}, wage = {wage}")


if __name__ == "__main__":
    main()
