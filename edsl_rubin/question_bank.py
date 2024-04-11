from textwrap import dedent
from edsl import QuestionMultipleChoice
from edsl.questions import QuestionLinearScale, QuestionNumerical

### About the good being produced

############################
## Task within an occupation
############################

q_info = QuestionMultipleChoice(
    question_name = "info_good",
    question_text = dedent("""\
        Consider this occupation {{ occupation }} and this task: {{ task }}.
        Is this task output an information good (i.e., writing/text, audio, a decision, an image, a video)? 
        """),
    question_options = ["Yes", "Partially", "No"]
)

q_gen_ai_help = QuestionLinearScale(
    question_name = "gen_ai_help",
    question_text = dedent("""\
    Consider this occupation: {{ occupation}} and this task: {{ task }}.
    How helpful would generative AI to be someone performing the task? 
    """), 
    question_options = list(range(0, 11)), 
    option_labels = {0:"No help at all.", 
                     10: "Generative AI could do this task."},
                     
)

q_trigger = QuestionMultipleChoice(
    question_text = """Consider the following task {{ task }} as part of the occupation {{ occupation }}.
    What best describes when this task is done?
    """, 
    question_options = [
        "Completion of a previous step in some productive process", 
        "External trigger like a customer request", 
        "External trigger a request from another part of the organization", 
        "Internal choice - can be scheduled whenever convenient"],
    question_name = "trigger")

q_easy_to_tell = QuestionLinearScale(
    question_name = "easy_to_tell",
    question_text = dedent("""\
        Consider the occupation {{ occupation }}. 
        And consider the task done by someone in this occupation: {{ task }}.
        Rate how easy it is for a non-expert to tell if the task is done correctly, on a scale of 1 to 10.
        
        The task: {{ task }}.
        1 = Very easy for a non-expert to tell if done correctly.
        10 = only an expert would know if done correctly. 
        """), 
    question_options = list(range(0, 11)), 
    option_labels = {0:"Very easy for a non-expert. Nearly any human could tell", 
                     10: "Impossible except for someone who is not an expert"},
                     
)

q_manager_task = QuestionMultipleChoice(
    question_text = """
    Consider this occupation: {{ occupation }}.
    Part of this occupation is this task: {{ task }}.
    
    Imagine a manager who was in this occupation is now asking a worker with the same occupation to do the task. 
    How much specific instruction is this task likely to require? 
    """,
    question_name = "manager_task", 
    question_options = ["None", "Almost none", "Some - less than an hour", "Great deal - mulitple hours"]
)

q_ratio_describe_to_do = q_describe = QuestionNumerical(
    question_text = """
    Consider this occupation: {{ occupation }}.
    Part of this occupation is this task: {{ task }}.
    
    Imagine a manager who was in this occupation is now asking a worker with the same occupation to do the task. 
    The worker is expert and does not need instruction in how to do the task generally. 
    They only need to be instructed on any specifics. 
    
    What is the ratio of time to describe this task to time to do it on their own?
    E.g., if it takes just as long to describe the task as to do it on their own, then 1.
    If the task could be described in 5 minutes but the task takes 10 minutes, then 0.5.
    """,
    question_name = "ratio_describe_to_do"
)

from edsl import Survey
task_survey = Survey([q_info, q_gen_ai_help, q_trigger, q_easy_to_tell, q_manager_task, q_ratio_describe_to_do])


## This is overall question about sequencing - takes occupations and tasks as input

q_task_sequence = QuestionLinearScale(
    question_name = "task_sequence",
    question_text = dedent("""\
        For this: \"{{occupation}}\"
        Consider the sequence of tasks: \"{{tasks}}\"
        On a scale of 1 to 10, to what extent would these tasks happen in a predictable sequence of steps vs. be done in arbitrary order?
        """), 
    question_options = list(range(0, 11)), 
    option_labels = {0:"Completely random order", 10: "Exact same sequence every time"}
)

