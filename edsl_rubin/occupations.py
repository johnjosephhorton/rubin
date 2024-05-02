import sqlite3
from textwrap import dedent
import pandas as pd

from edsl import Scenario

conn = sqlite3.connect('data.db')
cursor = conn.cursor()

def search_title(partial_title):
    query = f"select * from occupation_data where title like '%{partial_title}%'"
    tables = pd.read_sql_query(query, conn)
    return tables

def get_index(title):
    titles = get_titles()
    if title not in titles: 
        raise Exception("Title not found")
    return titles.index(title)

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

def get_single_scenarios(title):
    tasks = get_tasks(title)
    return [Scenario({'occupation': title, 'task':task}) for task in tasks]

def get_scenarios(title_list):
    all_scenarios = []
    for title in title_list:
        all_scenarios.extend(get_single_scenarios(title))

    return all_scenarios
