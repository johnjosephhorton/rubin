################# Conditioning #################
def condition_DAG(GPT_input_occupation, 
                  tasks, 
                  input_filename, 
                  output_filename):
    
    ### Set up questions and options
    triangle_check_question_text = dedent("""\
                                            Consider {{ occupation }} as an occupation. 
                                            And consider these three tasks: 
                                            A) {{ task_A }} 
                                            B) {{ task_B }}
                                            C) {{ task_C }} 
                                            Imagine there are two workers, one is working on task A and the other is working on task B.
                                            Suppose a third worker wants to start working on task C.
                                            Does the third worker need to wait for the workers working on tasks A and B to finish before starting on task C?
                                            Or can the third worker start working on task C right after worker B finishes without needing output of worker A?
                                            Avoid using words like "task A", "task B", or "task C" in the answer.
                                            Explain the reasoning behind your answer in a couple of sentences.
                                            """)
    triangle_check_question_options = {'keep AC': "Third worker can only start C after both the first and second workers have finished tasks A and B",
                                        'drop AC': "Third worker can start C after the second worker finishes B without having to wait for the first worker to finish A",
                                        'sanity check': "These are not part of the same task sequence"
                                        }

    AC_DC_question_question_text = dedent("""\
                                            Consider {{ occupation }} as an occupation.
                                            And consider these tasks:
                                            A) {{ task_A }}
                                            B) {{ task_B }}
                                            C) {{ task_C }}
                                            D) {{ task_D }}         
                                            As part of the steps leading up to completion of this job {{ task_C }} must be done.
                                            We know that tasks A, B, and D are inputs to task C.Moreover, task D is an input to task A while task A is an input to task B.
                                            Imagine there are three workers, one is working on task A, one is working on task B, and the other is working on task D.
                                            Suppose a fourth worker wants to start working on task C.
                                            Does the fourth worker need to wait for the workers working on tasks A, B, and D to finish before starting on task C?
                                            Or can the third worker start working on task C after workers B and D finish without needing output of worker A?
                                            How about after workers B and A finish without needing output of worker D?
                                            Avoid using words like "task A", "task B", or "task C" in the answer.
                                            Explain the reasoning behind your answer in a couple of sentences.
                                            """)
    AC_DC_question_options = {'drop AC, drop DC': "Given that worker B is finished, output of neither worker A nor worker D is needed for worker C",
                                'keep AC, drop DC': "Given that worker B is finished, output of worker A is needed for worker C but output of worker D is not",
                                'drop AC, keep DC': "Given that worker B is finished, output of worker D is needed for worker C but output of worker A is not",
                                'keep AC, keep DC': "Given that worker B is finished, output of both workers A and D are needed for worker C",
                                'sanity check': "These are not part of the same task sequence"
                                }

    triangle_check_question_options_list = list(triangle_check_question_options.values())
    AC_DC_question_options_list = list(AC_DC_question_options.values())



    def triangle_check(occupation, tasks, triangles_list, question_text, question_options_list):
        triangles = np.array(triangles_list)
        task_A_list = triangles[:, 0]
        task_B_list = triangles[:, 1]
        task_C_list = triangles[:, 2]
        scenarios = [Scenario({"occupation": occupation, "task_A": tasks[task_A], "task_B": tasks[task_B], "task_C": tasks[task_C]}) 
            for task_A, task_B, task_C in zip(task_A_list, task_B_list, task_C_list)]

        q = QuestionMultipleChoice(
            question_name = "ordering",
            question_text = question_text,
            question_options = question_options_list
        )
        results = q.by(m4).by(scenarios).run()
        return results



    q_AC_DC = QuestionMultipleChoice(
        question_name = "AC_DC",
        question_text = AC_DC_question_question_text,
        question_options = AC_DC_question_options_list
        )


    ### Step 1:
    #### Find all "triangles", defined as cases with:
    ##### A --> B --> C
    ##### A --> C

    # Read output of one step GPT DAG
    GPT_AM_df = pd.read_csv(input_filename)

    # Remove "Target" node for conditioning analysis. Will add later
    GPT_AM_df_targetNodes = GPT_AM_df[GPT_AM_df.target == '"Target"']
    GPT_AM_df = GPT_AM_df[GPT_AM_df.target != '"Target"']


    # Convert GPT AM data frame to adjacency matrix
    GPT_AM = pd.DataFrame(0, index=tasks, columns=tasks)
    for index, row in GPT_AM_df.iterrows():
        GPT_AM.at[row['source'], row['target']] = 1



    def find_triangles(matrix):
        # Ensure matrix is a numpy array
        if not isinstance(matrix, np.ndarray):
            matrix = matrix.to_numpy()
        
        # get length of matrix
        n = matrix.shape[0]

        # create list containing integers from 0 to n-1 for indexing
        numbers = list(range(n))

        # Find triangles
        triangles = []
        for x, y, z in itertools.permutations(numbers, 3):
            # get indices of destination nodes for outgoing edges of x
            out_edges_destination_x = np.where(matrix[x] == 1)[0]
            out_edges_destination_x = list(out_edges_destination_x)

            # check if x has outgoing edge to both y and z
            # if yes, check if y has outgoing edge to z
            if y in out_edges_destination_x and z in out_edges_destination_x:
                out_edges_destination_y = np.where(matrix[y] == 1)[0]
                out_edges_destination_y = list(out_edges_destination_y)
                
                # check if y has outgoing edge to z
                # if yes, we have a triangle
                if z in out_edges_destination_y:
                    triangles.append([x, y, z])
        
        return triangles

    # Find triangles
    GPT_AM_triangles_list = find_triangles(GPT_AM)
    #print(f'Examples of triangles: {GPT_AM_triangles_list[:5]}')
    print(f'Count of triangles: {len(GPT_AM_triangles_list)}')



    # If there are no triangles found, export input as output (as conditioning method doesn't change anything)
    if len(GPT_AM_triangles_list) == 0:
        # Add back "Target" node and save original input as output
        GPT_AM_df = pd.concat([GPT_AM_df, GPT_AM_df_targetNodes], ignore_index=True)
        GPT_AM_df.to_csv(output_filename, index=False)



    ### Step 2: 
    #### Ask GPT whether conditional on having B --> C we need A --> C
    results = triangle_check(GPT_input_occupation, tasks, GPT_AM_triangles_list, triangle_check_question_text, triangle_check_question_options_list)
    #results.select("task_A", "task_B", "task_C", "ordering", "comment.ordering_comment").print()
    GPT_trianglesCheck_output = results.select("task_A", "task_B", "task_C", "ordering", "comment.ordering_comment").to_pandas()
    GPT_trianglesCheck_output = GPT_trianglesCheck_output.sort_values(by=['scenario.task_A', 'scenario.task_C', 'scenario.task_B']).reset_index(drop=True)


    ### In cases where A --> C is shared among multiple triangles, only delete when all triangles say delete
    # Step 1: Find the count of triangles for each A --> C pair
    GPT_trianglesCheck_output['AC_pair_triangles_count'] = GPT_trianglesCheck_output.groupby(['scenario.task_A', 'scenario.task_C'])['scenario.task_A'].transform('count')


    # Step 2: Find if all triangles say delete
    aux_df = GPT_trianglesCheck_output.groupby(['scenario.task_A', 'scenario.task_C'])['answer.ordering'].apply(lambda x: (x == triangle_check_question_options['drop AC']).mean()*100).reset_index()
    aux_df.columns = ['scenario.task_A', 'scenario.task_C', 'fraction_triangles_say_delete']
    aux_df = aux_df[aux_df['fraction_triangles_say_delete'] == 100]

    # Merge aux_df with the original DataFrame to keep 'comment.ordering_comment'
    edges_to_remove = pd.merge(aux_df, GPT_trianglesCheck_output, on=['scenario.task_A', 'scenario.task_C'])
    edges_to_remove['comment.ordering_comment'] = (
        edges_to_remove['comment.ordering_comment'] 
        + '\n\n(Source):\n' + edges_to_remove['scenario.task_A'] 
        + '\n(Conditioned on):\n' + edges_to_remove['scenario.task_B'] 
        + '\n(Target):\n' + edges_to_remove['scenario.task_C']
    )
    edges_to_remove = edges_to_remove[['scenario.task_A', 'scenario.task_C', 'comment.ordering_comment']]

    # Step 3: Delete the rows where all triangles say delete
    modified_GPT_trianglesCheck = pd.merge(GPT_trianglesCheck_output, edges_to_remove[['scenario.task_A', 'scenario.task_C']], 
                                        how='left', 
                                        on=['scenario.task_A', 'scenario.task_C'], 
                                        indicator=True)
    modified_GPT_trianglesCheck = modified_GPT_trianglesCheck[modified_GPT_trianglesCheck['_merge'] == 'left_only'].drop(columns=['_merge', 'AC_pair_triangles_count'])
    modified_GPT_trianglesCheck = modified_GPT_trianglesCheck.reset_index(drop=True)


    #### Create a variable saying how many times each node appears as which node in a triangle
    ##### Purpose: find quanrangles
    # Initialize an empty DataFrame with unique values as columns and original columns as rows
    aux_df = pd.DataFrame(0, index=['scenario.task_A', 'scenario.task_B', 'scenario.task_C'], columns=tasks)

    # Fill the new DataFrame with counts
    for col in modified_GPT_trianglesCheck[['scenario.task_A', 'scenario.task_B', 'scenario.task_C']].columns:
        value_counts = modified_GPT_trianglesCheck[col].value_counts()
        aux_df.loc[col, value_counts.index] = value_counts.values
    aux_df = aux_df.T

    # Keep tasks which are sometimes node A of a triangle and sometimes node B of a triangle
    #aux_df = aux_df[(aux_df > 0).all(axis=1)]
    #print('Nodes stats as nodes A, B, C of a triangle:')
    aux_df

    # get list of pivotal tasks
    #pivotal_tasks = aux_df.index.tolist()




    ### Step 3: 
    ##### In cases where 
    # A --> B --> C and D --> A --> C 
    ##### the situation is different from when 
    # A --> B --> C and A --> D --> C
    ##### In such cases, edges A --> C and D --> C must be considered simultaneously as triangles are not totally "independent". 
    #### So we look for "quadrangles"

    # Iterate over the list of tuples and subset the DataFrame
    quadrangles_tasks = []
    for A, B, C, D in itertools.permutations(tasks, 4):
        # Initialize an empty list to collect the indices of desired rows
        quadrangle_indices = []

        # Find rows where triangle nodes are A, B, C
        condition1 = (modified_GPT_trianglesCheck['scenario.task_A'] == A) & (modified_GPT_trianglesCheck['scenario.task_B'] == B) & (modified_GPT_trianglesCheck['scenario.task_C'] == C)
        rows1 = modified_GPT_trianglesCheck[condition1]
        
        # Find rows where triangle nodes are D, A, C
        condition2 = (modified_GPT_trianglesCheck['scenario.task_A'] == D) & (modified_GPT_trianglesCheck['scenario.task_B'] == A) & (modified_GPT_trianglesCheck['scenario.task_C'] == C)
        rows2 = modified_GPT_trianglesCheck[condition2]
        
        # If both conditions are met, add the indices to the list
        if not rows1.empty and not rows2.empty:
            quadrangles_tasks.append((A, B, C, D))    
    print(f'Number of quadrilaterals: {len(quadrangles_tasks)}')




    quadrangles_df = pd.DataFrame()
    if len(quadrangles_tasks) > 0:
        scenarios = [Scenario({"occupation": GPT_input_occupation, "tasks": tasks,
                    "task_A": A, "task_B": B, "task_C": C, "task_D": D})
                    for A, B, C, D in quadrangles_tasks]
        results_AC_DC = q_AC_DC.by(m4).by(scenarios).run()
        #results_AC_DC.select(['answer.AC_DC', 'scenario.task_A', 'scenario.task_B', 'scenario.task_C', 'scenario.task_D', 'comment.AC_DC_comment']).print()
        quadrangles_df = results_AC_DC.select(['answer.AC_DC', 'scenario.task_A', 'scenario.task_B', 'scenario.task_C', 'scenario.task_D', 'comment.AC_DC_comment']).to_pandas()

        # decide whether to keep or drop AC and DC
        quadrangles_df['keep_AC'] = quadrangles_df['answer.AC_DC'].apply(lambda x: x in [AC_DC_question_options['keep AC, keep DC'], AC_DC_question_options['keep AC, drop DC']])
        quadrangles_df['keep_DC'] = quadrangles_df['answer.AC_DC'].apply(lambda x: x in [AC_DC_question_options['keep AC, keep DC'], AC_DC_question_options['drop AC, keep DC']])

        # Add node info to comments
        quadrangles_df['comment.AC_DC_comment'] = (
            quadrangles_df['comment.AC_DC_comment'] 
            + '\n\n(Source):\n' + quadrangles_df['scenario.task_A'] 
            + '\n(Conditioned on):\n' + quadrangles_df['scenario.task_B'] 
            + '\n(Target):\n' + quadrangles_df['scenario.task_C']
            + '\n(Other task -- Task D):\n' + quadrangles_df['scenario.task_D']
        )
    quadrangles_df.head()



    #### Drop extra AC and DC edges
    ACDC_edges_to_remove = pd.DataFrame()
    if len(quadrangles_tasks) > 0:
        # Step 1: Get list of unique edges found in all quadrangles
        pairs_AC = list(zip(quadrangles_df["scenario.task_A"], quadrangles_df["scenario.task_C"]))
        pairs_AC = [(task_A, task_C, 'AC') for (task_A, task_C) in pairs_AC]
        pairs_DC = list(zip(quadrangles_df["scenario.task_D"], quadrangles_df["scenario.task_C"]))
        pairs_DC = [(task_D, task_C, 'DC') for (task_D, task_C) in pairs_DC]
        all_pairs = pairs_AC + pairs_DC


        # Step 2: Get list of edges to keep
        aux_df = quadrangles_df[quadrangles_df['keep_AC']==True]
        pairs_AC_toKeep = list(zip(aux_df["scenario.task_A"], aux_df["scenario.task_C"]))
        pairs_AC_toKeep = [(task_A, task_C, 'AC') for (task_A, task_C) in pairs_AC_toKeep]

        aux_df = quadrangles_df[quadrangles_df['keep_DC']==True]
        pairs_DC_toKeep = list(zip(aux_df["scenario.task_D"], aux_df["scenario.task_C"]))
        pairs_DC_toKeep = [(task_D, task_C, 'DC') for (task_D, task_C) in pairs_DC_toKeep]

        pairs_toKeep = pairs_AC_toKeep + pairs_DC_toKeep


        # Step 3: Get list of edges to drop
        ACDC_edges_toDrop_list = [item for item in all_pairs if item not in pairs_toKeep]
        ACDC_edges_to_remove = pd.DataFrame(ACDC_edges_toDrop_list, columns=["scenario.task_A", "scenario.task_C", 'ID'])


        # Step 4: Match comments
        AC_indices = ACDC_edges_to_remove[ACDC_edges_to_remove['ID'] == 'AC'].index
        DC_indices = ACDC_edges_to_remove[ACDC_edges_to_remove['ID'] == 'DC'].index

        # Split ACDC_edges_to_remove into two DataFrames based on AC or DC edges
        aux_df_AC = ACDC_edges_to_remove.loc[AC_indices]#[['scenario.task_A', 'scenario.task_C']]
        aux_df_DC = ACDC_edges_to_remove.loc[DC_indices]#[['scenario.task_A', 'scenario.task_C']]

        # Merge comments depending on AC or DC edge
        merged_AC = pd.merge(aux_df_AC, quadrangles_df[['scenario.task_A', 'scenario.task_C', 'comment.AC_DC_comment']], 
                            on=['scenario.task_A', 'scenario.task_C'], 
                            how='left')
        merged_AC = merged_AC[['scenario.task_A', 'scenario.task_C', 'comment.AC_DC_comment']]
        merged_DC = pd.merge(aux_df_DC, quadrangles_df[['scenario.task_D', 'scenario.task_C', 'comment.AC_DC_comment']], 
                            left_on=['scenario.task_A', 'scenario.task_C'], 
                            right_on=['scenario.task_D', 'scenario.task_C'], 
                            how='left')
        merged_DC = merged_DC[['scenario.task_A', 'scenario.task_C', 'comment.AC_DC_comment']]

        # Concatenate merged DataFrames
        ACDC_edges_to_remove = pd.concat([merged_AC, merged_DC]).sort_index()
        ACDC_edges_to_remove = ACDC_edges_to_remove.drop_duplicates(subset=['scenario.task_A', 'scenario.task_C']).reset_index(drop=True)
        ACDC_edges_to_remove.columns = ['scenario.task_A', 'scenario.task_C', 'comment.ordering_comment'] # for consistency with previous edges_to_remove data frame


    # Create a DataFrame of edges to be dropped from this analysis and earlier analyses
    print(f'Number of AC-DC edges to remove: {len(ACDC_edges_to_remove)}')
    print(f'Number of AC edges to remove: {len(edges_to_remove)}')
    edges_to_remove = pd.concat([edges_to_remove, ACDC_edges_to_remove], ignore_index=True)
    print(f'Total number of edges to remove: {len(edges_to_remove)}')




    # Label edges to remove
    def adjust_graph_label_for_removed_edges(edges_to_remove, main_df):
        output = main_df.copy()
        for _, row in edges_to_remove.iterrows():
            task_A, task_C, comment = row['scenario.task_A'], row['scenario.task_C'], row['comment.ordering_comment']
            match = (main_df['source'] == task_A) & (main_df['target'] == task_C)
            if match.any():
                index_to_update = main_df.index[match][0]  # Get the index where the pair matches
                output.at[index_to_update, 'comment'] = comment + 'TriangleRemovedFlag'
        return output
    modified_GPT_AM_df = adjust_graph_label_for_removed_edges(edges_to_remove, GPT_AM_df)


    # Add back "Target" node edges
    modified_GPT_AM_df = pd.concat([modified_GPT_AM_df, GPT_AM_df_targetNodes], ignore_index=True)

    # Save output
    modified_GPT_AM_df.to_csv(output_filename, index=False)
