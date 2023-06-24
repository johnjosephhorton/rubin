import sqlalchemy
import random
import json
from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from LLM import LanguageModel

# Connect to the database and reflect the schema
engine = create_engine('sqlite:///occupation-task.db')
Base = automap_base()
Base.prepare(engine, reflect=True)

def get_row(table_name):
    TableClass = Base.classes[table_name]
    session = Session(engine)
    row = session.query(TableClass).first()
    return TableClass(**row.__dict__)

class Object:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key.startswith('_'):
                pass
            else:
                setattr(self, key, value)
    
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

    def add_LLM(self, LLM):
        self.LLM = LLM
        self.call_llm = LLM.call_open_ai_apt_35

    def __repr__(self):
        return f'<Object {self.__class__.__name__}>'
    
class Task(Object):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.occupation = None
    
    def add_occupation(self, Occupations):
        for _, occupation in Occupations.items():
            if occupation.onetsoc_code == self.onetsoc_code:
                self.occupation = occupation
                break

    def how_does_llm_help(self):
        prompt = f"""Consider this occupational task
        TASK: {self.task}
        How helpful would a large language model like GPT4 be with the completion of this task?
        1 = Not helpful at all
        10 = The model could do this task entirely on its own.
        Answer in valid json with the following example format:
        {{"rating": 5, "explanation": "This task is easy to assess because it is a binary classification task."}} 
        """
        response = self.call_llm(prompt)
        return response

    def easy_to_asess_output(self): 
        prompt = f"""Consider this occupational task
        TASK: {self.task}
        Rate how easy it is for a non-expert to tell if the task is done correctly.
        1  = Very easy for a non-export  
        10 = Only an expert would know if done correctly. 
        Answer in valid json with the following example format:
        {{"rating": 5, "explanation": "This task is easy to assess because it is a binary classification task."}} 
        """
        response = self.call_llm(prompt)
        return response
       
class Occupation(Object):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tasks = []

    def add_tasks(self, Tasks):
        for _, task in Tasks.items():
            if task.onetsoc_code == self.onetsoc_code:
                self.tasks.append(task)

    def assess_tasks(self):
        self.scores = {}
        for task in self.tasks:
            print(f"Now working on task: {task.task}")
            task.add_LLM(self.LLM)
            score_rubin = task.easy_to_asess_output()
            score_llm = task.how_does_llm_help()
            self.scores[task.task] = {'score_rubin': score_rubin, 'score_llm': score_llm}

    def task_ordering(self, task1, task2):
        prompt = f"""Consider these two occupational tasks:
        TASK %: {task1.task}
        TASK #: {task2.task}
        What order should these tasks be completed in?
        1) Task % output feeds into Task #
        2) Task # output feeds into Task %
        3) Either order is fine."""
        response = self.call_llm(prompt)
        return response
    
    def task_related(self, task1, task2):
        prompt = f"""Consider these two occupational tasks:
        TASK %: {task1.task}
        TASK #: {task2.task}
        Are these tasks related to each other, in the sense that % feeds into # or vice versa? 
        """
        response = self.call_llm(prompt)
        return response

   
table_name = 'task_statements'
TableClass = Base.classes[table_name]
session = Session(engine)
rows = session.query(TableClass).all()
Tasks = {row.__dict__['task_id']:Task(**row.__dict__) for row in rows}

table_name = 'occupation_data'
TableClass = Base.classes[table_name]
session = Session(engine)
rows = session.query(TableClass).all()

Occupations = {row.__dict__['onetsoc_code']:Occupation(**row.__dict__) for row in rows}
[occupation.add_tasks(Tasks) for _, occupation in Occupations.items()]
[task.add_occupation(Occupations) for _, task in Tasks.items()]

index = int(input(f'Enter a number between 1 and {len(Occupations)}: '))
code, o = list(Occupations.items())[index]

params = dict({
    "model": "gpt-3.5-turbo",
    "family": "openai",
    "temperature": 1.0
})
L = LanguageModel(**params)
o.add_LLM(L)
task1 = Task(**{'task': "Mix ingredients"})
task2 = Task(**{'task': "Bake cake"})
task3 = Task(**{'task': "Design a nuclear reactor"})

#order = o.task_ordering(task1, task2)
#print(order)

relationship = o.task_related(task1, task3)
print(relationship)

while False:
    #index = random.choice(range(len(Occupations)))
    # ask for input between 1 and range(len(Occupations))
    if False:
        index = int(input(f'Enter a number between 1 and {len(Occupations)}: '))
        code, o = list(Occupations.items())[index]
        print(f'Occupation: {o.title}')
        params = dict({
        "model": "gpt-3.5-turbo",
        "family": "openai",
        "temperature": 1.0
        })
        L = LanguageModel(**params)
        o.add_LLM(L)
        print("Assesing tasks")
        o.assess_tasks()
        print(o.scores)



