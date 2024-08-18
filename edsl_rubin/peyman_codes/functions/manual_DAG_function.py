def create_manual_DAG(occupation, 
                      input_filename,
                      output_filename):

    # Read manual adjacency matrix
    manual_AM = pd.read_csv(input_filename, index_col=0)

    # Initialize lists to store the source and target nodes
    sources = []
    targets = []

    # Iterate over the adjacency matrix to find ones and populate the lists
    for row_label, row in manual_AM.iterrows():
        for col_label, value in row.items():
            if value == 1:
                sources.append(row_label)
                targets.append(col_label)

    # Create data frame
    manual_DAG_df = pd.DataFrame({'source': sources, 'target': targets})

    # Save output
    manual_DAG_df.to_csv(output_filename, index=False)