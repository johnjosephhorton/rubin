import sqlite3
from textwrap import dedent
import pandas as pd

from edsl import Scenario

conn = sqlite3.connect('data.db')
cursor = conn.cursor()

def get_tasks(title):
    query = dedent(f"""\
        SELECT title, task 
        FROM occupation_data as o
        JOIN task_statements as t
        ON o.onetsoc_code = t.onetsoc_code
        AND title = '{title}'
        AND task_type = 'Core'""")
    tables = pd.read_sql_query(query, conn)
    return tables['task'].tolist()

def get_titles():
    query = "SELECT title FROM occupation_data;"
    titles = pd.read_sql_query(query, conn)['title'].tolist()
    return titles

titles = get_titles()

def get_single_scenarios(index):
    if index < 0 or index >= len(titles):
        raise ValueError(f"Index {index} is out of range. There are only {len(titles)} occupations.")
    title = titles[index]
    tasks = get_tasks(title)
    return [Scenario({'occupation': title, 'task':task}) for task in tasks]

def get_scenarios(index_or_slice):
    all_scenarios = []
    if isinstance(index_or_slice, int):  # Single index
        indexes = [index_or_slice]  # Convert to list for uniform processing
    elif isinstance(index_or_slice, slice):  # Slice object (range)
        start, stop, step = index_or_slice.indices(len(titles))  # Get slice parameters within range
        indexes = range(start, stop, step)
    else:
        raise TypeError("Input must be an integer or a slice.")

    # Process each index in the list or range
    for index in indexes:
        all_scenarios.extend(get_single_scenarios(index))

    return all_scenarios
