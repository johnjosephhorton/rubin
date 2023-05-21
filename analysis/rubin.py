from typing import Any

def GrowTree(node, i):
    if i == n:
        return
    else:
        node.add_child(i)
        GrowTree(node.children[-1], i+1)
        GrowTree(node, i+1)





import random 

class Task:
    def __init__(self, q, ch, cm, subtasks = None):
        self.q = q
        self.ch = ch
        self.cm = cm
        if subtasks:
            self.subtasks = subtasks
        else:
            self.subtasks = []
        self.status = "HUMAN"

    @classmethod
    def combine_tasks(cls, task1, task2):
        q = task1.q * task2.q
        ch = task1.ch + task2.ch
        cm = (task1.cm + task2.cm)/2 ## you sure about this? 
        return cls(q, ch, cm, [task1, task2])
    
    @classmethod 
    def random(cls):
        # q is a random float 
        q = random.random()
        ch = random.random()
        cm = random.random()
        return cls(q, ch, cm)

    def __repr__(self):
        return 'Task(%s, %s, %s)' % (self.q, self.ch, self.cm)
    
    @property
    def human_cost(self):
        return self.ch
    
    @property
    def machine_cost(self):
        return self.cm / self.q
    
class Job:
    def __init__(self, tasks):
        self.tasks = tasks
    
    def __repr__(self):
        return 'Job(%s)' % (self.tasks)
    
    @staticmethod
    def evaluate(task1, task2):
        cHH = task1.human_cost + task2.human_cost
        cHM = task1.human_cost + task2.machine_cost
        cMH = task1.machine_cost + task2.human_cost
        cMM = ((task1.cm + task2.cm) / 2) / (task1.q * task2.q)
        return dict({'HH': cHH, 'HM': cHM, 'MH': cMH, 'MM': cMM})
    
    def Q(self):
        return [task.q for task in self.tasks]

    # find the index of two successive values in Q such that the sum is highest
    def find_max(self):
        Q = self.Q()
        max_sum = 0
        max_index = 0
        for i in range(len(Q) - 1):
            if Q[i] + Q[i+1] > max_sum:
                max_sum = Q[i] + Q[i+1]
                max_index = i
        return max_index

    def optimal(self):
        for task in self.tasks:
            if task.human_cost > task.machine_cost:
                task.status = "MACHINE"
        Q = self.Q()
        max_index = self.find_max()
        costs = self.evaluate(self.tasks[max_index], self.tasks[max_index + 1])
        if costs['MM'] == min(costs.values):
            self.tasks[max_index].status = "MACHINE"
            self.tasks[max_index + 1].status = "MACHINE"
            new_task = Task.combine_tasks(self.tasks[max_index], self.tasks[max_index + 1])
            self.tasks[max_index] = new_task
            self.tasks.pop(max_index + 1)

if __name__ == "__main__":
    t1 = Task(1, 2, 3)
    t2 = Task(4, 5, 6)
    t3 = Task.random()
    j = Job([t1, t2, t3])
    print(j.evaluate(t1, t2))
    t4 = Task.combine_tasks(t1, t2)