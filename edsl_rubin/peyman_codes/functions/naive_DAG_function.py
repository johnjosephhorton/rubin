def naive_DAG(GPT_input_occupation,
              tasks,
              lastTask_output_filename,
              output_DAG_filename_naive,
              output_DAG_filename_naiveTwoStep,
              conditioned_DAG_output_filename):

    # Set output names
    output_filename_wo = output_DAG_filename_naive
    output_filename_w = output_DAG_filename_naiveTwoStep


    # ### Set up the questions for GPT

    task_relationships_question_options_wo = {'A first': "Worker working on task B needs to know the output of worker working on task A",
                                            'B first': "Worker working on task A needs to know the output of worker working on task B",
                                            'neither': "Neither worker needs to know the output of the other worker"
                                            }

    task_relationships_question_options_w = {'A first': "Worker working on task B needs to know the output of worker working on task A",
                                            'B first': "Worker working on task A needs to know the output of worker working on task B",
                                            'either': "Either worker can start first, but the output of one worker is needed by the other worker",
                                            'neither': "Neither worker needs to know the output of the other worker"
                                            }

    symmetric_edges_question_options = {'A first': "Worker working on task B needs to know the output of worker working on task A",
                                        'B first': "Worker working on task A needs to know the output of worker working on task B",
                                        }

    task_relationships_question_text = dedent("""\
                                            Consider {{ occupation }} as an occupation. 
                                            And consider these two tasks: 
                                            A) {{ task_A }} 
                                            B) {{ task_B }}
                                            Imagine there are two workers, one working on task A and the other on task B.
                                            Does the worker working on task B need to know the output of the worker working on task A before getting started? What about the opposite?
                                            Avoid using words like "task A" and "task B" in the answer.
                                            Explain the reasoning behind your answer in a couple of sentences.
                                            """)

    symmetric_edges_question_text = dedent("""\
                                        Consider {{ occupation }} as an occupation. 
                                        And consider these two tasks: 
                                        A) {{ task_A }} 
                                        B) {{ task_B }}
                                        Imagine there are two workers, one working on task A and the other on task B.
                                        Does the worker working on task B need to know the output of the worker working on task A before getting started? What about the opposite?
                                        Avoid using words like "task A" and "task B" in the answer.
                                        Explain the reasoning behind your answer in a couple of sentences.
                                        """)


    # In[208]:


    task_relationships_question_options_wo_list = list(task_relationships_question_options_wo.values())
    task_relationships_question_options_w_list = list(task_relationships_question_options_w.values())
    symmetric_edges_question_options_list = list(symmetric_edges_question_options.values())


    # In[209]:


    def get_last_tasks(occupation, tasks):
        scenarios = [Scenario({"occupation": occupation, "tasks": tasks})]

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
        
        return last_task


    # In[210]:


    # Get last task(s) to be done in occupation
    last_task = get_last_tasks(GPT_input_occupation, tasks)
    last_tasks_df = pd.DataFrame({'last_task': [last_task]})
    last_tasks_df.to_csv(lastTask_output_filename, index=False)


    # ### 2.1) One Step Method: Directly ask for pairwise comparison w/o giving the "either" option

    # In[211]:


    # Compare pair of tasks
    def task_relationships(occupation, tasks, question_text, question_options):
        scenarios = [Scenario({"occupation": occupation, "task_A": task_A, "task_B": task_B}) 
            for task_A, task_B in combinations(tasks, 2)]

        q = QuestionMultipleChoice(
            question_name = "ordering",
            question_text = question_text,
            question_options = question_options
        )
        results = q.by(m4).by(scenarios).run()
        return results

    results = task_relationships(GPT_input_occupation, tasks, task_relationships_question_text, task_relationships_question_options_wo_list)
    #results.select("task_A", "task_B", "ordering", "comment.ordering_comment").print()
    pairwise_relationships_wo_raw = results.select("task_A", "task_B", "ordering", "comment.ordering_comment").to_pandas()


    # In[212]:


    # Swap columns and subset only those that are part of the same task sequence 
    pairwise_relationships_wo = pairwise_relationships_wo_raw.copy()
    mask = pairwise_relationships_wo['answer.ordering'] == task_relationships_question_options_wo['B first']
    pairwise_relationships_wo.loc[mask, ['scenario.task_A', 'scenario.task_B']] = pairwise_relationships_wo.loc[mask, ['scenario.task_B', 'scenario.task_A']].values
    pairwise_relationships_wo.loc[mask, 'answer.ordering'] = task_relationships_question_options_wo['A first']
    pairwise_relationships_wo = pairwise_relationships_wo[pairwise_relationships_wo['answer.ordering'] == task_relationships_question_options_wo['A first']]
    pairwise_relationships_wo = pairwise_relationships_wo[['scenario.task_A', 'scenario.task_B', 'comment.ordering_comment']]

    # Change column names
    pairwise_relationships_wo = pairwise_relationships_wo.rename(columns={'scenario.task_A': 'source', 
                                                                        'scenario.task_B': 'target', 
                                                                        'comment.ordering_comment': 'comment'})


    # In[213]:


    # Add outgoing edges from last task(s) to "Target" node
    for task in last_task:
        aux_df = pd.DataFrame({'source': [task],
                            'target': ['"Target"'],
                            'comment': ['Job Completion Indicator']})
        pairwise_relationships_wo = pd.concat([pairwise_relationships_wo, aux_df], ignore_index=True)


    # In[214]:


    # Save one-step Naive output
    pairwise_relationships_wo.to_csv(output_filename_wo, index=False)


    # ### 2.2) Two Steps Method: Give option of "either" and then filter symmetric edges
    # ### Step 1:

    # In[215]:


    # Compare pair of tasks
    def task_relationships(occupation, tasks, question_text, question_options):
        scenarios = [Scenario({"occupation": occupation, "task_A": task_A, "task_B": task_B}) 
            for task_A, task_B in combinations(tasks, 2)]

        q = QuestionMultipleChoice(
            question_name = "ordering",
            question_text = question_text,
            question_options = question_options
        )
        results = q.by(m4).by(scenarios).run()
        return results

    results = task_relationships(GPT_input_occupation, tasks, task_relationships_question_text, task_relationships_question_options_w_list)
    #results.select("task_A", "task_B", "ordering", "comment.ordering_comment").print()
    pairwise_relationships_w_raw = results.select("task_A", "task_B", "ordering", "comment.ordering_comment").to_pandas()


    # ### Step 2:

    # In[216]:


    # subset symmetric edges
    both_edges = pairwise_relationships_w_raw[pairwise_relationships_w_raw['answer.ordering'] == task_relationships_question_options_w['either']]
    if len(both_edges) > 0:
        task_A_list = both_edges['scenario.task_A'].tolist()
        task_B_list = both_edges['scenario.task_B'].tolist()


        # Decide which one of symmetric edges to keep
        def pick_oneOf_symmetricEdges(occupation, task_A_list, task_B_list, question_text, question_options):
            scenarios = [Scenario({"occupation": occupation, "task_A": task_A, "task_B": task_B}) 
                for task_A, task_B in zip(task_A_list, task_B_list)]

            q = QuestionMultipleChoice(
                question_name = "ordering",
                question_text = question_text,
                question_options = question_options
            )
            results = q.by(m4).by(scenarios).run()
            return results

        results = pick_oneOf_symmetricEdges(GPT_input_occupation, task_A_list, task_B_list, task_relationships_question_text, symmetric_edges_question_options_list)
        #results.select("task_A", "task_B", "ordering", "comment.ordering_comment").print()
        which_symmetric_edge = results.select("task_A", "task_B", "ordering", "comment.ordering_comment").to_pandas()


    # In[217]:


    if len(both_edges) > 0:
        # Merge datasets
        pairwise_relationships_w = pairwise_relationships_w_raw[pairwise_relationships_w_raw['answer.ordering'].isin(symmetric_edges_question_options_list)]
        pairwise_relationships_w = pd.concat([pairwise_relationships_w, which_symmetric_edge], ignore_index=True)
    else:
        pairwise_relationships_w = pairwise_relationships_w_raw[pairwise_relationships_w_raw['answer.ordering'].isin(symmetric_edges_question_options_list)]

    # Swap columns
    mask = pairwise_relationships_w['answer.ordering'] == task_relationships_question_options_w['B first']
    pairwise_relationships_w.loc[mask, ['scenario.task_A', 'scenario.task_B']] = pairwise_relationships_w.loc[mask, ['scenario.task_B', 'scenario.task_A']].values
    pairwise_relationships_w.loc[mask, 'answer.ordering'] = task_relationships_question_options_w['A first']
    pairwise_relationships_w = pairwise_relationships_w[pairwise_relationships_w['answer.ordering'] == task_relationships_question_options_w['A first']]
    pairwise_relationships_w = pairwise_relationships_w[['scenario.task_A', 'scenario.task_B', 'comment.ordering_comment']]

    # Change column names
    pairwise_relationships_w = pairwise_relationships_w.rename(columns={'scenario.task_A': 'source', 
                                                                        'scenario.task_B': 'target', 
                                                                        'comment.ordering_comment': 'comment'})


    # In[218]:


    # Add outgoing edges from last task(s) to "Target" node
    for task in last_task:
        aux_df = pd.DataFrame({'source': [task],
                            'target': ['"Target"'],
                            'comment': ['Job Completion Indicator']})
        pairwise_relationships_w = pd.concat([pairwise_relationships_w, aux_df], ignore_index=True)


    # In[219]:


    # Save two-step Naive output
    pairwise_relationships_w.to_csv(output_filename_w, index=False)


    # ### Apply "conditioning" procedure to Naive (a.k.a. one-step) output

    # In[220]:


    # Get "conditioned" DAG
    condition_DAG(GPT_input_occupation, 
                  tasks, 
                  input_filename = output_DAG_filename_naive, 
                  output_filename = conditioned_DAG_output_filename)

