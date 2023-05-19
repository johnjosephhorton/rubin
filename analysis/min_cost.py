import random 

def random_between(a, b):
    return a + (b-a)*random.random()

class Task:
    def __init__(self, node_type):
        self.node_type = node_type
        self.children = []
        self._parent = None
        self.total_cost = 0
        self.notation = "root" 

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
                child_notation = f"({task_number})"
            else:
                task = Task("MACHINE")
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
                # remove the last character from the sting, and add the new notation
                child_notation = f"| {task_number}>"
                task.notation = node.notation[:-1] + child_notation
                task.total_cost = node.total_cost + task.marginal_cost
                children.append(task)
        [self.GrowTree(child, task_number+1) for child in children]
        node.children = children
        return node    
 

import numpy as np 

if __name__ == "__main__":
    costs1 = []
    for _ in range(10000):
        J = Job.fromRandom(cm = 1, qmin = 0.6, cmin = 0.1, cmax = 2, n = 10)
        J.GrowTree()
        costs1.append(J.min_cost)
    costs2 = []
    for _ in range(10000):
        J = Job.fromRandom(cm = 1.1, qmin = 0.6, cmin = 0.1, cmax = 2, n = 10)
        J.GrowTree()
        costs2.append(J.min_cost)

    c1 = np.average(costs1)
    c2 = np.average(costs2)
    pct_change = (c2 - c1)/c1
    print(pct_change)
       