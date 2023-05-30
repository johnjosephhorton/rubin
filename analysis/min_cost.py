import random 
import numpy as np
import pandas as pd

def random_between(a, b):
    return a + (b-a)*random.random()

class Task:
    def __init__(self, node_type):
        self.node_type = node_type
        self.children = []
        self._parent = None
        self.total_cost = 0
        self.notation = "root" 
        self.human_count = 0

    def add_child(self, child):
        self.children.append(child)

    def add_parent(self, parent):
        self._parent = parent
    
    def __repr__(self):
        return 'Task(nodetype = %s, total_cost = %s, notation = %s)' % (
            self.node_type, self.total_cost, self.notation)
        
    @property
    def parent(self):
        return self._parent

def prod(list):
    "multiply each element of the list together"
    p = 1
    for i in list:
        p *= i
    return p

class Job:
    def __init__(self, cm, Q, CH):
        self.Q = Q
        self.n = len(Q)
        self.CH = CH
        self.cm = cm
        self.best_node = None
        self.min_cost = 1.0/prod(Q) + sum(CH)

    @classmethod
    def fromRandom(cls, cm, qmin, cmin, cmax, n):
        Q = [random_between(qmin, 1) for i in range(n)]
        CH = [random_between(cmin, cmax) for i in range(n)]
        return cls(cm = cm, Q = Q, CH = CH)

    def GrowTree(self, node = None, task_number = None):
        if node is None:
            node = Task("ROOT")
            task_number = 0
        if task_number == self.n:
            if node.total_cost < self.min_cost:
                self.min_cost = node.total_cost
                self.best_node = node
            return node
        else:
            ch = self.CH[task_number]
            q = self.Q[task_number]
            cm = self.cm
            children = []
            if ch < cm/q:
                task = Task("HUMAN")
                task.marginal_cost = ch
                task.human_count = node.human_count + 1 
                child_notation = f"({task_number})"
            else:
                task = Task("MACHINE")
                task.human_count = node.human_count
                task.marginal_cost = cm/q
                task.chain_product = q # we need this for the machine-chain calculation
                child_notation = f"<{task_number}>"
            task.add_parent(node)
        if node.parent:
            task.total_cost = node.total_cost + task.marginal_cost
            task.notation = node.notation + child_notation
        else:
            task.total_cost = task.marginal_cost
            task.notation = child_notation
        children.append(task)
        if node.parent: # if it's a child
            if node.node_type == "MACHINE" or node.node_type == "MACHINE-CHAIN":
                # you can only be a machine-chain if your parent is a machine or a machine-chain
                task = Task("MACHINE-CHAIN")
                task.add_parent(node)
                task.chain_product = q * node.chain_product
                task.marginal_cost = (1.0 / (node.chain_product)) * (1 - q)/(q)
                task.human_count = node.human_count
                # remove the last character from the sting, and add the new notation
                child_notation = f"| {task_number}>"
                task.notation = node.notation[:-1] + child_notation
                task.total_cost = node.total_cost + task.marginal_cost
                children.append(task)
        [self.GrowTree(child, task_number+1) for child in children]
        node.children = children
        return node    
 

class Simulation: 
    # n is how many tasks
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
                cm = self.cm, 
                qmin = self.qmin, 
                cmin = self.cmin, 
                cmax = self.cmax, n = self.n)
            J.GrowTree()
            self.costs.append(J.min_cost)
            self.human_count.append(J.best_node.human_count)

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
        # compute standard deviation of costs
        return np.std(self.costs) 

    def parameters_as_dictionary(self):
        return dict({"cm": self.cm, "qmin": self.qmin, "cmin": self.cmin, "cmax": self.cmax, "n": self.n})
        
    def summary_stats(self):
        return dict({"avg_cost": self.avg_cost,"max_cost": self.max_cost, "min_cost": self.min_cost, "std_cost": self.std_cost, 
                     "avg_human_count": self.avg_human_count})

    def results(self):
        params = self.parameters_as_dictionary()
        stats = self.summary_stats()
        return {**params, **stats}



if __name__ == "__main__":
    num_sim = 100
    results = [] 
    nrange = range(2, 11)
    # create range of values between start and stop, spaced by epsilon
    cmin_range = np.arange(0.1, 2, 0.1)
    qmin_range = np.arange(0.1, 1, 0.1)
    cm_range = np.arange(0.1, 2, 0.1)
    n_range = range(2, 11)
    for cmin in cmin_range:
        print(f"On cmin {cmin}")
        for qmin in qmin_range:
            for cm in cm_range:
                for n in n_range:
                    S = Simulation(cm = cm, qmin = qmin, cmin = cmin, cmax = 2, n = n)
                    S.run(num_sim)
                    results.append(S.results())

    # write a list of dictionaries to a pandas dataframe
    df = pd.DataFrame(results)
    # write the dataframe to a csv file
    df.to_csv("../computed_objects/simulation_results.csv")

    J = Job.fromRandom(cm = 1, qmin = 0.1, cmin = 0.1, cmax = 2, n = 3)
