import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from itertools import product


class Job:
    """Represents a sequence of tasks to be assigned to either human or AI"""

    def __init__(self, cm: float, Q: List[float], CH: List[float]):
        self.cm = cm  # management cost
        self.Q = Q  # AI probabilities
        self.n = len(Q)
        self.CH = CH  # human costs
        self._solve_dp()  # Solve immediately upon initialization

    @classmethod
    def fromRandom(
        cls, cm: float, qmin: float, cmin: float, cmax: float, n: int
    ) -> "Job":
        """Create a Job instance with random parameters within given bounds"""
        Q = [random.uniform(qmin, 1.0) for _ in range(n)]
        CH = [random.uniform(cmin, cmax) for _ in range(n)]
        return cls(cm=cm, Q=Q, CH=CH)

    def _solve_dp(self) -> None:
        """Solve using dynamic programming approach"""
        n = self.n
        dp = [0.0] * (n + 1)
        self.choice = [None] * n

        prefix_q = [1.0] * (n + 1)
        for i in range(n):
            prefix_q[i + 1] = prefix_q[i] * self.Q[i]

        for i in reversed(range(n)):
            best_cost = self.CH[i] + dp[i + 1]
            best_mode = "H"
            best_j = i

            for j in range(i, n):
                if prefix_q[j + 1] == 0.0 or prefix_q[i] == 0.0:
                    continue

                prob_product = prefix_q[j + 1] / prefix_q[i]
                if prob_product <= 0:
                    continue

                ai_cost = self.cm / prob_product + dp[j + 1]
                if ai_cost < best_cost:
                    best_cost = ai_cost
                    best_mode = "A"
                    best_j = j

            dp[i] = best_cost
            self.choice[i] = (best_mode, best_j)

        self.min_cost = dp[0]
        self.notation = self._reconstruct_solution()
        self.human_count = sum(1 for mode, _ in self.choice if mode == "H")

    def _reconstruct_solution(self) -> str:
        blocks = []
        i = 0
        while i < self.n:
            mode, j = self.choice[i]
            if mode == "H":
                blocks.append(f"({i})")
                i += 1
            else:
                if i == j:
                    blocks.append(f"<{i}>")
                else:
                    chain = "|".join(str(k) for k in range(i, j + 1))
                    blocks.append(f"<{chain}>")
                i = j + 1
        return "".join(blocks)


class Simulation:
    def __init__(self, cm, qmin, cmin, cmax, n):
        self.cm = cm
        self.qmin = qmin
        self.cmin = cmin
        self.cmax = cmax
        self.n = n
        self.costs = []
        self.human_count = []

    def run(self, num_simulations):
        for _ in range(num_simulations):
            J = Job.fromRandom(
                cm=self.cm, qmin=self.qmin, cmin=self.cmin, cmax=self.cmax, n=self.n
            )
            self.costs.append(J.min_cost)
            self.human_count.append(J.human_count)

    @property
    def avg_human_count(self):
        return np.average(self.human_count)

    @property
    def avg_cost(self):
        return np.average(self.costs)

    @property
    def min_cost(self):
        return np.min(self.costs)

    @property
    def max_cost(self):
        return np.max(self.costs)

    @property
    def std_cost(self):
        return np.std(self.costs)

    def parameters_as_dictionary(self):
        return {
            "cm": self.cm,
            "qmin": self.qmin,
            "cmin": self.cmin,
            "cmax": self.cmax,
            "n": self.n,
        }

    def summary_stats(self):
        return {
            "avg_cost": self.avg_cost,
            "max_cost": self.max_cost,
            "min_cost": self.min_cost,
            "std_cost": self.std_cost,
            "avg_human_count": self.avg_human_count,
        }

    def results(self):
        params = self.parameters_as_dictionary()
        stats = self.summary_stats()
        return {**params, **stats}


if __name__ == "__main__":
    num_sim = 100
    results = []

    # Parameter ranges
    cmin_range = np.arange(0.1, 2, 0.1)
    qmin_range = np.arange(0.1, 1, 0.1)
    cm_range = np.arange(0.1, 2, 0.1)
    n_range = range(2, 11)

    # Run simulations for all parameter combinations
    for cmin in cmin_range:
        print(f"On cmin {cmin}")
        for qmin in qmin_range:
            for cm in cm_range:
                for n in n_range:
                    S = Simulation(cm=cm, qmin=qmin, cmin=cmin, cmax=2, n=n)
                    S.run(num_sim)
                    results.append(S.results())

    # Convert results to DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv("../computed_objects/simulation_results.csv")
    print("Results saved to simulation_results.csv")
