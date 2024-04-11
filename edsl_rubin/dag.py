import copy
from textwrap import dedent

import networkx as nx
import matplotlib.pyplot as plt
from edsl.questions import QuestionCheckBox
from edsl import Scenario

def plot_dag(data):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges based on the dictionary structure
    for key, values in data.items():
        for value in values:
            G.add_edge(key, value)

    # Draw the DAG
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=15, width=2.5, edge_color='gray')
    plt.title("Directed Acyclic Graph (DAG)")
    plt.show()

def draw_dag_line(occupation, focal_task, all_tasks):
    tasks = copy.deepcopy(all_tasks)
    tasks.remove(focal_task)
    q = QuestionCheckBox(
        question_name = "dag",
        question_text = dedent("""\
            Consider this {{ occupation }}. 
            And consider this task: {{ task }}. 
            Of the following tasks, which task is this task an input to? 
            Check all that apply.
            """),
        question_options = tasks
    )
    scenario = Scenario({'occupation':occupation, 'task': focal_task})
    return q.by(scenario).run()

from occupations import get_single_scenarios

def draw_dag(index):
    scenarios = get_single_scenarios(index)
    occupation = scenarios[0]["occupation"]
    tasks = [scenario['task'] for scenario in scenarios]    
    dag = dict({})
    for task in tasks: 
        dag[task] = draw_dag_line(occupation, task, tasks).select("dag").first()
    return dag


