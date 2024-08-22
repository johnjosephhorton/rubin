def firstLastTask_DAG(GPT_input_occupation,
                      tasks,
                      firstLastTask_output_filename,
                      firstLastTask_DAG_output_filename,
                      conditioned_DAG_output_filename):

    # ### Set up the questions for GPT

    # In[142]:


    task_relationships_question_options = {'A first': "Worker working on task B needs to know the output of worker working ong task A",
                                            'B first': "Worker working on task A needs to know the output of worker working ong task B",
                                            'neither': "Neither worker needs to know the output of the other worker"
                                            }
    firstLastTask_question_text = dedent("""\
                                        Consider {{ occupation }} as an occupation.
                                        The first task (or set of tasks) to be completed for the job is: {{ first_task }}.
                                        The last task (or set of tasks) to be completed for the job is: {{ last_task }}. 
                                        Now consider these two tasks:
                                        A) {{ task_A }} 
                                        B) {{ task_B }}
                                        Imagine there are two workers, one working on task A and the other on task B.
                                        Does the worker working on task B need to know the output of the worker working on task A before getting started? What about the opposite?
                                        Avoid using words like "task A" and "task B" in the answer.
                                        Explain the reasoning behind your answer in a couple of sentences.
                                        """)


    # In[143]:


    task_relationships_question_options_list = list(task_relationships_question_options.values())


    # In[144]:


    def task_relationships_firstLast_included(occupation, tasks, first_task, last_task, question_text, question_options):
        # Modify the first task and last task to appear as a single string
        first_task = " And ".join(first_task)
        last_task = " And ".join(last_task)

        scenarios = [Scenario({"occupation": occupation, 
                            "task_A": task_A, "task_B": task_B,
                            "first_task": first_task, "last_task": last_task}) 
                            for task_A, task_B in combinations(tasks, 2)]

        q = QuestionMultipleChoice(
            question_name = "ordering",
            question_text = question_text,
            question_options = question_options
        )
        results = q.by(m4).by(scenarios).run()
        return results


    # ### Use One Step Method: Directly ask for pairwise comparison w/o giving the "either" option
    # ### Next determine first and last task/tasks to be done in the sequence and ask GPT to produce DAG

    # In[145]:


    def get_first_last_tasks(occupation, tasks):
        scenarios = [Scenario({"occupation": occupation, "tasks": tasks})]

        # First task
        q1 = QuestionCheckBox(
            question_name = "firstTask",
            question_text = dedent("""\
                Consider {{ occupation }} as an occupation.
                The tasks below are part of the job of a {{ occupation }}: {{ tasks }}.
                Among the following, which task or set of tasks would be done before all other tasks in order to compelete the job?
                """),
            question_options = tasks,
            min_selections = 1,
            max_selections = int(np.floor(len(tasks) / 2)) # an upper bound for how many tasks can be considered as first task
        )
        results1 = q1.by(m4).by(scenarios).run().to_pandas()
        first_task = results1['answer.firstTask'][0]
        first_task = ast.literal_eval(first_task) # convert from string resembling list format to actual list


        # Last task
        q2 = QuestionCheckBox(
            question_name = "lastTask",
            question_text = dedent("""\
                Consider {{ occupation }} as an occupation. 
                The tasks below are part of the job of {{ occupation }}: {{ tasks }}.
                Among the following, which task or set of tasks would be done after all other tasks are completed?
                """),
            question_options = tasks,
            min_selections = 1,
            max_selections = int(np.floor(len(tasks) / 2)) # an upper bound for how many tasks can be considered as last task
        )
        results2 = q2.by(m4).by(scenarios).run().to_pandas()
        last_task = results2['answer.lastTask'][0]
        last_task = ast.literal_eval(last_task) # convert from string resembling list format to actual list
        
        return first_task, last_task


    # In[146]:


    # Get first and last task(s) to be done in occupation
    first_task, last_task = get_first_last_tasks(GPT_input_occupation, tasks)


    # ### Compare pair of tasks

    # In[147]:


    results = task_relationships_firstLast_included(GPT_input_occupation, tasks, first_task, last_task, firstLastTask_question_text, task_relationships_question_options_list)
    #results.select("task_A", "task_B", "ordering", "comment.ordering_comment").print()
    GPT_firstLast_df_raw = results.select("task_A", "task_B", "ordering", "comment.ordering_comment").to_pandas()

    # Swap columns and subset only those that are part of the same task sequence 
    GPT_firstLast_df = GPT_firstLast_df_raw.copy()
    mask = GPT_firstLast_df['answer.ordering'] == task_relationships_question_options['B first']
    GPT_firstLast_df.loc[mask, ['scenario.task_A', 'scenario.task_B']] = GPT_firstLast_df.loc[mask, ['scenario.task_B', 'scenario.task_A']].values
    GPT_firstLast_df.loc[mask, 'answer.ordering'] = task_relationships_question_options['A first']
    GPT_firstLast_df = GPT_firstLast_df[GPT_firstLast_df['answer.ordering'] == task_relationships_question_options['A first']]
    GPT_firstLast_df = GPT_firstLast_df[['scenario.task_A', 'scenario.task_B', 'comment.ordering_comment']]

    # Change column names
    GPT_firstLast_df = GPT_firstLast_df.rename(columns={'scenario.task_A': 'source', 
                                                        'scenario.task_B': 'target', 
                                                        'comment.ordering_comment': 'comment'})


    # In[148]:



    # Check whether "artificial" last task is needed given DAG structure and last task(s) generated
    source_tasks = set(GPT_firstLast_df['source'].unique())
    target_tasks = set(GPT_firstLast_df['target'].unique())
    DAG_implied_last_task = list(target_tasks - source_tasks - set(last_task))

    firstLast_tasks_df = pd.DataFrame({'first_task': [first_task], 
                                       'last_task': [last_task],
                                       'implied_last_task': [DAG_implied_last_task]})
    firstLast_tasks_df.to_csv(firstLastTask_output_filename, index=False)


    # In[23]:


    # Add outgoing edges from last task(s) to "Target" node
    # first combine original last task(s) with implied last task(s)
    if len(DAG_implied_last_task) > 0:
        print(f'Warning: {len(DAG_implied_last_task)} DAG implied last task(s) found.')
        for task in DAG_implied_last_task:
            last_task.append(task)

    for task in last_task:
        aux_df = pd.DataFrame({'source': [task],
                            'target': ['"Target"'],
                            'comment': ['Job Completion Indicator']})
        GPT_firstLast_df = pd.concat([GPT_firstLast_df, aux_df], ignore_index=True)


    # In[149]:


    # Save output
    GPT_firstLast_df.to_csv(firstLastTask_DAG_output_filename, index=False)


    # ### Apply "conditioning" procedure to output

    # In[150]:


    # Get "conditioned" DAG
    condition_DAG(GPT_input_occupation, 
                  tasks, 
                  input_filename = firstLastTask_DAG_output_filename,
                  output_filename = conditioned_DAG_output_filename)

