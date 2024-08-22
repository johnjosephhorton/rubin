def partitioned_DAG(GPT_input_occupation,
                    tasks,
                    lastTask_output_filename,
                    partitions_output_filename,
                    partitioned_DAG_output_filename, 
                    conditioned_partitioned_DAG_output_filename):

   
    partitions_relationship_question_text = dedent("""\
                                                Consider {{ occupation }} as an occupation. 
                                                And consider these two partitions of tasks: 
                                                A) {{ partition_A }} 
                                                B) {{ partition_B }}
                                                Imagine there are two groups of workers, one group are working on partition A tasks and the other are working on partition B tasks.
                                                Do the workers working on partition B tasks need to know the output of the workers working on partition A tasks before getting started?  What about the opposite?
                                                Avoid using words like "partition A" and "partition B" in the answer.
                                                Explain the reasoning behind your answer in a couple of sentences.
                                                """)
    partitions_relationship_question_options = {'A first': "Workers working on partition B tasks need to know the output of workers working on partition A tasks",
                                                'B first': "Workers working on partition A tasks need to know the output of workers working on partition B tasks",
                                                'neither': "Neither group of workers needs to know the output of the other group of workers"
                                                }

    within_partition_task_relationships_question_text = dedent("""\
                                                            Consider {{ occupation }} as an occupation. 
                                                            And consider these two tasks: 
                                                            A) {{ task_A }} 
                                                            B) {{ task_B }}
                                                            Imagine there are two workers, one working on task A and the other on task B.
                                                            Does the worker working on task B need to know the output of the worker working on task A before getting started? What about the opposite?
                                                            Avoid using words like "task A" and "task B" in the answer.Explain the reasoning behind your answer in a couple of sentences.
                                                            """)
    withinPartition_relationship_question_options = {'A first': "Worker working on task B needs to know the output of worker working on task A",
                                                        'B first': "Worker working on task A needs to know the output of worker working on task B",
                                                        'neither': "Neither worker needs to know the output of the other worker"
                                                        }

    between_partition_task_relationships_question_text = dedent("""\
                                                                Consider {{ occupation }} as an occupation. 
                                                                And consider these two partitions of tasks; partition 1: {{ tasks_partition1 }} and partition 2: {{ tasks_partition2 }}.
                                                                We know that tasks in partition 1 would be done before tasks in partition 2.
                                                                Now consider these two tasks:
                                                                A) {{ task_A }} 
                                                                B) {{ task_B }}
                                                                Imagine there are two workers, one working on task A and the other on task B.
                                                                Does the worker working on task B need to know the output of the worker working on task A before getting started? What about the opposite?
                                                                Avoid using words like "task A" and "task B" in the answer.
                                                                Explain the reasoning behind your answer in a couple of sentences.
                                                                """)
    betweenPartition_relationship_question_options = {'A first': "Worker working on task B needs to know the output of worker working on task A",
                                                        'B first': "Worker working on task A needs to know the output of worker working on task B",
                                                        'neither': "Neither worker needs to know the output of the other worker"
                                                        }


    # In[8]:


    partitions_relationship_question_options_list = list(partitions_relationship_question_options.values())
    withinPartition_relationship_question_options_list = list(withinPartition_relationship_question_options.values())
    betweenPartition_relationship_question_options_list = list(betweenPartition_relationship_question_options.values())


    # In[9]:


    def partition_relationships(occupation, partitions_dict, question_text, question_options):
        scenarios = [Scenario({"occupation": occupation, "partition_A": A, "partition_B": B}) 
            for A, B in itertools.combinations(partitions_dict.values(), 2)]

        q = QuestionMultipleChoice(
            question_name = "ordering",
            question_text = question_text,
            question_options = question_options
        )
        results = q.by(m4).by(scenarios).run()
        return results



    def task_relationships_within_partition(occupation, tasks, question_text, question_options):
        scenarios = [Scenario({"occupation": occupation, "task_A": task_A, "task_B": task_B}) 
            for task_A, task_B in combinations(tasks, 2)]

        q = QuestionMultipleChoice(
            question_name = "ordering",
            question_text = question_text,
            question_options = question_options
        )
        results = q.by(m4).by(scenarios).run()
        return results



    def task_relationships_between_partitions(occupation, tasks_partition1, tasks_partition2, question_text, question_options):
        scenarios = [Scenario({"occupation": occupation, 
                            "tasks_partition1": tasks_partition1, "tasks_partition2": tasks_partition2,
                            "task_A": task_A, "task_B": task_B}) 
            for task_A, task_B in itertools.product(tasks_partition1, tasks_partition2)]

        q = QuestionMultipleChoice(
            question_name = "ordering",
            question_text = question_text,
            question_options = question_options
        )
        results = q.by(m4).by(scenarios).run()
        return results



    # ## Break down the DAG into multiple minimally-connected subgraphs

    # In[10]:


    task_partitioning_question_text =  dedent("""\
                                            Consider {{ occupation }} as an occupation. 
                                            And consider these tasks: {{ tasks }}.
                                            Can these tasks be partitioned into separate, minimally connected groups of tasks?
                                            If so, give the number of groups and list tasks in each group. 
                                            Avoid using \n in the answer, and list groups in the following format: Group x: ['task1', 'task2', 'task3'].
                                            """)

    def partition_tasks(occupation, tasks, question_text):
        scenarios = [Scenario({"occupation": occupation, "tasks": tasks})]

        q = QuestionFreeText(
            question_name = "partition",
            question_text = question_text
        )
        results = q.by(m4).by(scenarios).run()
        return results

    results = partition_tasks(GPT_input_occupation, tasks, task_partitioning_question_text)
    #results.print()
    partition_tasks_output_str = results.select("answer.partition").to_pandas().iloc[0,0]


    # #### Group tasks into smaller partitions

    # In[11]:


    # Find all "Group x" occurrences in LLM output
    groups = re.findall(r'Group \d+', partition_tasks_output_str)

    # Split the text at each "Group x"
    parts = re.split(r'(Group \d+:)', partition_tasks_output_str)

    # Initialize a dictionary to hold the group texts
    partitions_dict = {}

    # Iterate through the parts and store the texts in the dictionary
    for i in range(1, len(parts), 2):
        group_name = parts[i].strip(': ')
        group_number = int(re.search(r'\d+', group_name).group())
        group_text = parts[i+1].strip().rstrip('.,')
        
        # Convert the string representation of the list to an actual list
        partitions_dict[group_number] = group_text

    # Save partitions
    partitions_df = pd.DataFrame.from_dict(partitions_dict, orient='index')
    partitions_df.to_csv(partitions_output_filename, index=False, header=False) 


    # ## Determine relation of partitions

    # In[12]:


    # Compare pair of partitions
    results = partition_relationships(GPT_input_occupation, partitions_dict, partitions_relationship_question_text, partitions_relationship_question_options_list)
    #results.print()
    partitions_ordering_df = results.select("partition_A", "partition_B", "ordering", "ordering_comment").to_pandas()


    # In[13]:


    partitions_ordering_df


    # In[14]:


    # Swap columns so that all partitions in first column are done earlier
    mask = partitions_ordering_df['answer.ordering'] == partitions_relationship_question_options['B first']
    partitions_ordering_df.loc[mask, ['scenario.partition_A', 'scenario.partition_B']] = partitions_ordering_df.loc[mask, ['scenario.partition_B', 'scenario.partition_A']].values
    partitions_ordering_df.loc[mask, 'answer.ordering'] = partitions_relationship_question_options['A first']
    partitions_ordering_df = partitions_ordering_df[partitions_ordering_df['answer.ordering'] == partitions_relationship_question_options['A first']]
    partitions_ordering_df


    # In[15]:


    # Add group numbers to data frame
    aux_dict = {v: k for k, v in partitions_dict.items()}
    partitions_ordering_df['partition_A_groupNum'] = partitions_ordering_df['scenario.partition_A'].map(aux_dict)
    partitions_ordering_df['partition_B_groupNum'] = partitions_ordering_df['scenario.partition_B'].map(aux_dict)
    partitions_ordering_df


    # ## Compare pair of tasks within each partition

    # In[16]:


    # Function to handle apastrophes and commas in the list string
    def clean_list_string(s):
        # Escape the apostrophe in specific problematic cases
        s = re.sub(r"(?<!\\)'s costs", r"\\'s costs", s)
        return s


    task_relationships_within_partition_df = pd.DataFrame()
    for key, value in partitions_dict.items():
        # Get list of tasks in the partition
        my_partition_tasks = clean_list_string(value)
        my_partition_tasks = ast.literal_eval(my_partition_tasks)
        if len(my_partition_tasks) < 2:
            continue

        # Run the function
        results = task_relationships_within_partition(GPT_input_occupation, my_partition_tasks, within_partition_task_relationships_question_text, withinPartition_relationship_question_options_list)
        aux_df = results.select("task_A", "task_B", "ordering", "comment.ordering_comment").to_pandas()
        aux_df['partition'] = key

        # Add to data frame
        task_relationships_within_partition_df = pd.concat([task_relationships_within_partition_df, aux_df], ignore_index=True)


    # In[17]:


    # Swap columns so that all tasks in first column are done earlier
    mask = task_relationships_within_partition_df['answer.ordering'] == withinPartition_relationship_question_options['B first']
    task_relationships_within_partition_df.loc[mask, ['scenario.task_A', 'scenario.task_B']] = task_relationships_within_partition_df.loc[mask, ['scenario.task_B', 'scenario.task_A']].values
    task_relationships_within_partition_df.loc[mask, 'answer.ordering'] = withinPartition_relationship_question_options['A first']


    # ## Compare pair of tasks between partitions

    # In[18]:


    task_relationships_between_partitions_df = pd.DataFrame()
    for (key1, value1), (key2, value2) in itertools.combinations(partitions_dict.items(), 2):
        # determine which partition is done first
        if len(partitions_ordering_df[(partitions_ordering_df['partition_A_groupNum'] == key1) & (partitions_ordering_df['partition_B_groupNum'] == key2)]) > 0:
            first_partition = key1
            second_partition = key2
        elif len(partitions_ordering_df[(partitions_ordering_df['partition_A_groupNum'] == key2) & (partitions_ordering_df['partition_B_groupNum'] == key1)]) > 0:
            first_partition = key2
            second_partition = key1
        else:
            continue

        # Get list of tasks in the partition
        tasks_partition1 = ast.literal_eval(clean_list_string(value1))
        tasks_partition2 = ast.literal_eval(clean_list_string(value2))
        
        # Run the function
        results = task_relationships_between_partitions(GPT_input_occupation, tasks_partition1, tasks_partition2, between_partition_task_relationships_question_text, betweenPartition_relationship_question_options_list)
        aux_df = results.select("task_A", "task_B", "ordering", "comment.ordering_comment").to_pandas()
        
        # Add to data frame
        task_relationships_between_partitions_df = pd.concat([task_relationships_between_partitions_df, aux_df], ignore_index=True)


    # In[19]:


    # Swap columns so that all tasks in first column are done earlier
    mask = task_relationships_between_partitions_df['answer.ordering'] == betweenPartition_relationship_question_options['B first']
    task_relationships_between_partitions_df.loc[mask, ['scenario.task_A', 'scenario.task_B']] = task_relationships_between_partitions_df.loc[mask, ['scenario.task_B', 'scenario.task_A']].values
    task_relationships_between_partitions_df.loc[mask, 'answer.ordering'] = betweenPartition_relationship_question_options['A first']


    # In[20]:


    # Get edges from within and between partitions data frames
    between_edges = task_relationships_between_partitions_df[task_relationships_between_partitions_df['answer.ordering'] == betweenPartition_relationship_question_options['A first']]
    between_edges = between_edges[['scenario.task_A', 'scenario.task_B', 'comment.ordering_comment']]

    within_edges = task_relationships_within_partition_df[task_relationships_within_partition_df['answer.ordering'] == withinPartition_relationship_question_options['A first']]
    within_edges = within_edges[['scenario.task_A', 'scenario.task_B', 'comment.ordering_comment']]


    # Combine edges from within and between partitions
    partitions_DAG_df = pd.concat([within_edges, between_edges], ignore_index=True)

    # Change column names
    partitions_DAG_df = partitions_DAG_df.rename(columns={'scenario.task_A': 'source', 
                                                        'scenario.task_B': 'target', 
                                                        'comment.ordering_comment': 'comment'})


    # ### Decide which task(s) should be done last

    # In[21]:


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


    # In[22]:


    # Get last task(s) to be done in occupation
    last_task = get_last_tasks(GPT_input_occupation, tasks)

    # Check whether "artificial" last task is needed given DAG structure and last task(s) generated
    source_tasks = set(partitions_DAG_df['source'].unique())
    target_tasks = set(partitions_DAG_df['target'].unique())
    DAG_implied_last_task = list(target_tasks - source_tasks - set(last_task))

    last_tasks_df = pd.DataFrame({'last_task': [last_task],
                                'implied_last_task': [DAG_implied_last_task]})
    last_tasks_df.to_csv(lastTask_output_filename, index=False)


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
        partitions_DAG_df = pd.concat([partitions_DAG_df, aux_df], ignore_index=True)


    # In[24]:


    # Save output
    partitions_DAG_df.to_csv(partitioned_DAG_output_filename, index=False)


    # ### Apply "conditioning" procedure to output

    # In[25]:


    # Get "conditioned" DAG
    condition_DAG(GPT_input_occupation, 
                  tasks, 
                  input_filename = partitioned_DAG_output_filename,
                  output_filename = conditioned_partitioned_DAG_output_filename)

