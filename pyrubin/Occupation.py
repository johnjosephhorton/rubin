import sqlalchemy
import random
import json
from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from LLM import LanguageModel
import json

from jinja2 import Template
from jinja2 import Environment, FileSystemLoader

from Question import Question

class PromptLibrary:
    def __init__(self, template_dir = "prompt_templates"):
        self.template_dir = template_dir
        self.env = Environment(loader=FileSystemLoader(self.template_dir))

    def show_templates(self):
        return self.env.loader.list_templates()
    
    def get_template_string(self, template_name):
        return self.env.loader.get_source(self.env, template_name)[0]


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
    
    def ask(self, question, data = None):
        """Asks a question to the model using the data in self.__dict__ as paramters unless 
        passed a data paramter."""
        if data:
            prompt = question.ask(data)
        else:
            prompt = question.ask(self.__dict__)
        return self.call_llm(prompt)

    def add_LLM(self, LLM):
        self.LLM = LLM
        self.call_llm = LLM.call_open_ai_apt_35

    def add_prompt_library(self, prompt_library): 
        self.prompt_library = prompt_library

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
        q = Question(library.get_template_string("llm_usefulness.txt"))
        return self.ask(q)

    def easy_to_asess_output(self): 
        q = Question(library.get_template_string("llm_rubin.txt"))
        return self.ask(q)

def integer_to_base(n, base):
    if n == 0:
        return (0,)
    digits = []
    while n != 0:
        digit = n % base
        digits.append(digit)
        n //= base
    return tuple(reversed(digits))

def alpha_label_generator():
    index = 0
    letters = "abcdefghijklmnopqrstuvwxyz"
    while True:
        digits = integer_to_base(index, 26)
        label = "".join([letters[d] for d in digits])
        yield label
        index += 1

class Occupation(Object):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tasks = []

    def add_tasks(self, Tasks):
        for _, task in Tasks.items():
            if task.onetsoc_code == self.onetsoc_code:
                self.tasks.append(task)

    @staticmethod
    def append_labels(task_list):
        """Takes a list of tasks and returns a string with labels and a dictionary mapping labels to tasks.
        E.g., a. bake a cake\nb. make a cake\nc. make a pie\n
        """
        results = ""
        d = dict()
        g = alpha_label_generator()
        for task in task_list:
            label = next(g)
            results += f"{label}. {task}\n"
            d[label] = task
        return results, d
        
    def task_grouping(self):
        "Uses the ocupations assigned to tasks"
        tasks = [t.task for t in self.tasks]
        task_list, d = self.append_labels(tasks)
        groupings = self.get_groupings(task_list)
        return groupings, d

    def get_groupings(self, task_list):
        q = Question(library.get_template_string("task_grouping.txt"))
        groupings = self.ask(q, {"task_list": task_list})
        return groupings       

    @staticmethod
    def clean_groupings(groupings):
        j = json.loads(groupings)
        s = j["groupings"].split(")(")
        k = [l.replace("(", "").replace(")", "") for l in s]
        m = [l.split(",") for l in k]
        n = [[l.strip() for l in o] for o in m]
        return n
    
    def assess_tasks(self):
        self.scores = {}
        for task in self.tasks:
            print(f"Now working on task: {task.task}")
            task.add_LLM(self.LLM)
            score_rubin = task.easy_to_asess_output()
            score_llm = task.how_does_llm_help()
            self.scores[task.task] = {'score_rubin': score_rubin, 'score_llm': score_llm}

    def task_ordering(self, task1, task2):
        d = dict({"task1": task1, "task2": task2})
        q = Question(library.get_template_string("task_ordering.txt"))
        return self.ask(q, d)
    


if __name__ == "__main__":
    params = dict({
        "model": "gpt-3.5-turbo",
        "family": "openai",
        "temperature": 1.0
    })
    L = LanguageModel(**params)
    library = PromptLibrary()

    if False:
        task1 = Task(**{'task': "Mix ingredients"})
        task1.add_LLM(L)
        library = PromptLibrary()

        q1 = Question(library.get_template_string("llm_usefulness.txt"))
        q2 = Question(library.get_template_string("llm_rubin.txt"))
        q3 = Question(library.get_template_string("task_ordering.txt"))

        print(task1.ask(q1))
        print(task1.ask(q2))

    if True:
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
            "model": "gpt-4",
            "family": "openai",
            "temperature": 1.0
        })
        L = LanguageModel(**params)
        o.add_LLM(L)
        print("Task grouping")
        #print(o.get_task_list())
        groupings, d = o.task_grouping()
        cleaned = o.clean_groupings(groupings)
        print(cleaned)



    if False:
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

        order = o.task_ordering(task1, task2)
        print(order)

        #relationship = o.task_related(task1, task3)
        #print(relationship)

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



