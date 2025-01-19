"""
handoff_solutions.py

Contains both the dynamic programming and brute force solutions to the 
sequence-partitioning problem with idiosyncratic handoff costs.
"""

# -------------- Dynamic Programming Solution --------------


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

    # 4) Reconstruct the partition
    blocks = []
    current = 1
    while current <= T:
        j = choice[current]
        wage = S_C[j] - S_C[current - 1]
        blocks.append((current, j, wage))
        current = j + 1

    return dp[1], blocks


# -------------- Brute Force Solution --------------


def calculate_block_cost(C, t, start, end):
    """Calculate the cost of a single block from start to end (0-based indices)."""
    sumC = sum(C[start : end + 1])
    sumT = sum(t[start : end + 1])
    return sumC * sumT


def generate_all_partitions(n):
    """
    Generate all possible partitions of sequence 0..n-1.
    Each partition is represented as a list of (start, end) pairs.
    """

    def recursive_partition(start, partitions):
        if start == n:
            yield partitions[:]
            return

        # Try ending current block at each possible position
        for end in range(start, n):
            partitions.append((start, end))
            yield from recursive_partition(end + 1, partitions)
            partitions.pop()

    yield from recursive_partition(0, [])


def solve_brute_force(C, t, h):
    """
    Solve for the minimal total cost by trying all possible partitions.

    Parameters are the same as in solve_dp:
    C : list of training costs
    t : list of times
    h : list of hand-off costs (h[i] = cost to start block at i+1)

    Returns (min_cost, best_partition) where best_partition is list of
    (start, end, wage) tuples in 1-based indexing.
    """
    T = len(C)
    min_cost = float("inf")
    best_partition = None

    # Try each possible partition
    for partition in generate_all_partitions(T):
        total_cost = 0

        # Calculate cost for each block
        for i, (start, end) in enumerate(partition):
            # Block cost
            block_cost = calculate_block_cost(C, t, start, end)
            total_cost += block_cost

            # Hand-off cost if not last block
            if end < T - 1:
                # Note: h[i] is cost to start at i+1, so we use end+1
                total_cost += h[end + 1]

        # Update best if this is better
        if total_cost < min_cost:
            min_cost = total_cost
            best_partition = partition

    # Convert partition to same format as DP solution (1-based indices)
    result_partition = []
    for start, end in best_partition:
        # Calculate wage (sum of C in block)
        wage = sum(C[start : end + 1])
        # Convert to 1-based indices
        result_partition.append((start + 1, end + 1, wage))

    return min_cost, result_partition


def validate_solutions(C, t, h):
    """Compare brute force and DP solutions."""
    bf_cost, bf_partition = solve_brute_force(C, t, h)
    dp_cost, dp_partition = solve_dp(C, t, h)

    print("Brute Force solution:")
    print(f"Cost: {bf_cost:.2f}")
    print("Partition:", bf_partition)
    print("\nDP solution:")
    print(f"Cost: {dp_cost:.2f}")
    print("Partition:", dp_partition)
    print("\nSolutions match?", abs(bf_cost - dp_cost) < 1e-10)


def main():
    """Run example cases to compare solutions."""
    print("Example case (T=5):")
    # Example data
    C = [3, 1, 2, 2, 5]  # training costs
    t = [1, 4, 2, 1, 3]  # times
    h = [7 * x for x in [0.5, 2, 0, 1, 5]]  # hand-off costs

    validate_solutions(C, t, h)

    print("\nSmall test case (T=3):")
    validate_solutions(C=[1, 2, 3], t=[1, 1, 1], h=[1, 1, 1])


if __name__ == "__main__":
    main()
