# Functions

# Pick occupation
def pick_occupation(occupation):
    if occupation == 'travelAgents':
        GPT_input_occupation = 'travel agents'
        plot_title_occupation = 'Travel Agents'
        occupation_code = '41-3041'
    elif occupation == 'insuranceUnderwriters':
        GPT_input_occupation = 'insurance underwriters'
        plot_title_occupation = 'Insurance Underwriters'
        occupation_code = '13-2053'
    elif occupation == 'pileDriverOperators':
        GPT_input_occupation = 'pile dirver operators'
        plot_title_occupation = 'Pile Driver Operators'
        occupation_code = '47-2072'
    elif occupation == 'dredgeOperators':
        GPT_input_occupation = 'dredge operators'
        plot_title_occupation = 'Dredge Operators'
        occupation_code = '53-7031'
    elif occupation == 'gradersAndSortersForAgriculturalProducts':
        GPT_input_occupation = 'Graders and sorters for agricultural products'
        plot_title_occupation = 'Graders and Sorters for Agricultural Products'
        occupation_code = '45-2041'
    elif occupation == 'reinforcingIronAndRebarWorkers':
        GPT_input_occupation = 'Reinforcing iron and rebar working'
        plot_title_occupation = 'Reinforcing Iron and Rebar Workers'
        occupation_code = '47-2171'
    elif occupation == 'insuranceAppraisersForAutoDamage':
        GPT_input_occupation = 'Insurance appraisers for auto damage'
        plot_title_occupation = 'Insurance Appraisers, Auto Damage'
        occupation_code = '13-1032'
    elif occupation == 'floorSandersAndFinishers':
        GPT_input_occupation = 'Floor sanders and finishers'
        plot_title_occupation = 'Floor Sanders and Finishers'
        occupation_code = '47-2043'
    elif occupation == 'dataEntryKeyer':
        GPT_input_occupation = 'Data entry keyers'
        plot_title_occupation = 'Data Entry Keyers'
        occupation_code = '43-9021'
    elif occupation == 'athletesAndSportsCompetitors':
        GPT_input_occupation = 'Athletes and sports competitors'
        plot_title_occupation = 'Athletes and Sports Competitors'
        occupation_code = '27-2021'
    elif occupation == 'audiovisualEquipmentInstallerAndRepairers':
        GPT_input_occupation = 'Audiovisual equipment installer and repairers'
        plot_title_occupation = 'Audiovisual Equipment Installers and Repairers'
        occupation_code = '49-2097'
    elif occupation == 'hearingAidSpecialists':
        GPT_input_occupation = 'Hearing aid specialists'
        plot_title_occupation = 'Hearing Aid Specialists'
        occupation_code = '29-2092'
    elif occupation == 'personalCareAides':
        GPT_input_occupation = 'Personal care aides'
        plot_title_occupation = 'Personal Care Aides'
        occupation_code = '31-1122'
    elif occupation == 'proofreadersAndCopyMarkers':
        GPT_input_occupation = 'Proofreaders and copy markers'
        plot_title_occupation = 'Proofreaders and Copy Markers'
        occupation_code = '43-9081'
    elif occupation == 'chiropractors':
        GPT_input_occupation = 'Chiropractors'
        plot_title_occupation = 'Chiropractors'
        occupation_code = '29-1011'
    elif occupation == 'shippingReceivingAndInventoryClerks':
        GPT_input_occupation = 'Shipping, receiving, and inventory clerks'
        plot_title_occupation = 'Shipping, Receiving, and Inventory Clerks'
        occupation_code = '43-5071'
    elif occupation == 'cooksShortOrder':
        GPT_input_occupation = 'Cooks, short order'
        plot_title_occupation = 'Cooks, Short Order'
        occupation_code = '35-2015'
    elif occupation == 'orthodontists':
        GPT_input_occupation = 'Orthodontists'
        plot_title_occupation = 'Orthodontists'
        occupation_code = '29-1023'
    elif occupation == 'subwayAndStreetcarOperators':
        GPT_input_occupation = 'Subway and streetcar operators'
        plot_title_occupation = 'Subway and Streetcar Operators'
        occupation_code = '53-4041'
    elif occupation == 'packersAndPackagersHand':
        GPT_input_occupation = 'Packers and packagers (with hand)'
        plot_title_occupation = 'Packers and Packagers, Hand'
        occupation_code = '53-7064'
    elif occupation == 'hoistAndWinchOperators':
        GPT_input_occupation = 'Hoist and winch operators'
        plot_title_occupation = 'Hoist and Winch Operators'
        occupation_code = '53-7041'
    elif occupation == 'forgingMachineSettersOperatorsAndTenders':
        GPT_input_occupation = 'Forging machine setters, operators, and tenders, metal and plastic'
        plot_title_occupation = 'Forging Machine Setters, Operators, and Tenders, Metal and Plastic'
        occupation_code = '51-4022'
    elif occupation == 'avionicsTechnicians':
        GPT_input_occupation = 'Avionics technicians'
        plot_title_occupation = 'Avionics Technicians'
        occupation_code = '49-2091'
    elif occupation == 'dishwashers':
        GPT_input_occupation = 'Dishwashers'
        plot_title_occupation = 'Dishwashers'
        occupation_code = '35-9021'
    elif occupation == 'dispatchersExceptPoliceFireAndAmbulance':
        GPT_input_occupation = 'Dispatchers, except police, fire, and ambulance'
        plot_title_occupation = 'Dispatchers, Except Police, Fire, and Ambulance'
        occupation_code = '43-5032'
    elif occupation == 'familyMedicinePhysicians':
        GPT_input_occupation = 'Family medicine physicians'
        plot_title_occupation = 'Family Medicine Physicians'
        occupation_code = '29-1215'
    elif occupation == 'MachineFeedersAndOffbearers':
        GPT_input_occupation = 'Machine feeders and offbearers'
        plot_title_occupation = 'Machine Feeders and Offbearers'
        occupation_code = '53-7063'
    elif occupation == 'shampooers':
        GPT_input_occupation = 'shampooers'
        plot_title_occupation = 'Shampooers'
        occupation_code = '39-5093'
    elif occupation == 'sociologists':
        GPT_input_occupation = 'Sociologists'
        plot_title_occupation = 'Sociologists'
        occupation_code = '47-2041'
    elif occupation == 'carpetInstallers':
        GPT_input_occupation = 'Carpet installers'
        plot_title_occupation = 'Carpet Installers'
        occupation_code = '19-3041'
    elif occupation == 'dancers':
        GPT_input_occupation = 'Dancers'
        plot_title_occupation = 'Dancers'
        occupation_code = '27-2031'
    
    occupation_folder = f'{data_path}/daily_tasks_occupations_analysis/{occupation}'

    # create folder in directory if it doesn't exist
    if not os.path.exists(occupation_folder):
        os.makedirs(occupation_folder)

    return GPT_input_occupation, plot_title_occupation, occupation_code, occupation_folder



# For HTML graph positions
def node_positions(occupation):
    if occupation == 'travelAgents':
        fixed_positions = {
            'Collect payment for transportation and accommodations from customer.': 
            (0, -100),
            'Converse with customer to determine destination, mode of transportation, travel dates, financial considerations, and accommodations required.': 
            (-400, 200),
            'Compute cost of travel and accommodations, using calculator, computer, carrier tariff books, and hotel rate books, or quote package tours costs.': 
            (-200, -100),
            'Book transportation and hotel reservations, using computer or telephone.': 
            (100, 0),
            'Plan, describe, arrange, and sell itinerary tour packages and promotional travel incentives offered by various travel carriers.':
            (-300, 0),
            'Provide customer with brochures and publications containing travel information, such as local customs, points of interest, or foreign country regulations.':
            (-400, -200),
            'Print or request transportation carrier tickets, using computer printer system or system link to travel carrier.':
            (200, 200),
            'Record and maintain information on clients, vendors, and travel packages.':
            (200, -200),
            '"Target"':
            (300, 0)
        }
    elif occupation == 'insuranceUnderwriters':
        fixed_positions = {
            'Decline excessive risks.': 
            (200, -150),
            'Write to field representatives, medical personnel, or others to obtain further information, quote rates, or explain company underwriting policies.': 
            (-300, 0),
            'Evaluate possibility of losses due to catastrophe or excessive insurance.': 
            (-100, 300),
            'Decrease value of policy when risk is substandard and specify applicable endorsements or apply rating to ensure safe, profitable distribution of risks, using reference materials.': 
            (0, 0),
            'Review company records to determine amount of insurance in force on single risk or group of closely related risks.':
            (-100, -300),
            'Authorize reinsurance of policy when risk is high.':
            (200, 150),
            'Examine documents to determine degree of risk from factors such as applicant health, financial standing and value, and condition of property.':
            (-500, 0),
            '"Target"':
            (300, 0)
        }
    elif occupation == 'pileDriverOperators':
        fixed_positions = {
            'Move hand and foot levers of hoisting equipment to position piling leads, hoist piling into leads, and position hammers over pilings.':
            (100, -250),
            'Conduct pre-operational checks on equipment to ensure proper functioning.':
            (-300, 0),
            'Drive pilings to provide support for buildings or other structures, using heavy equipment with a pile driver head.':
            (300, 0),
            'Move levers and turn valves to activate power hammers, or to raise and lower drophammers that drive piles to required depths.':
            (100, -100),
            'Clean, lubricate, and refill equipment.':
            (-100, 0)
        }
    elif occupation == 'shampooers':
        fixed_positions = {
            'Massage, shampoo, and condition patrons hair and scalp to clean them and remove excess oil.':
            (100, 50),
            'Advise patrons with chronic or potentially contagious scalp conditions to seek medical treatment.':
            (0, 0),
            'Treat scalp conditions and hair loss, using specialized lotions, shampoos, or equipment such as infrared lamps or vibrating equipment.':
            (100, -50),
            'Maintain treatment records.':
            (200, 0)
        }
    return fixed_positions




# Plotting interactive graph - two functions: compare_graphs and plot_graphs
def compare_graphs(df1, df2):
    '''
    For getting common elements of both graphs + unique elements of each graph
    '''
    # Find entries in df1 where 'comment' ends with 'Flag'
    flag_entries_df1 = df1[df1['comment'].str.endswith('TriangleRemovedFlag')]
    
    # Find entries in df2 where 'comment' ends with 'Flag'
    flag_entries_df2 = df2[df2['comment'].str.endswith('TriangleRemovedFlag')]

    # Remove flagged entries from df1 and df2
    df1 = df1[~df1.index.isin(flag_entries_df1.index)]
    df2 = df2[~df2.index.isin(flag_entries_df2.index)]

    # Find common rows based on columns "source" and "target"
    common = pd.merge(df1, df2, 
                    on=['source', 'target'], 
                    suffixes=('_df1', '_df2'))

    # Find unique rows in df1
    merged = pd.merge(df1, common[['source', 'target']], 
                    on=['source', 'target'], 
                    how='outer', 
                    indicator=True)
    in1_notIn2 = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

    # Find unique rows in df2
    merged = pd.merge(df2, common[['source', 'target']], 
                    on=['source', 'target'], 
                    how='outer', 
                    indicator=True)

    # Keep only rows that are in df1 but not in df2
    in2_notIn1 = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

    return common, in1_notIn2, in2_notIn1, flag_entries_df1, flag_entries_df2


def insert_line_breaks(text, max_length=80):
    '''
    For breaking long GPT comment lines while keeping existing line breaks
    '''
    def break_line(line, max_length):
        words = line.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 > max_length:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word) + 1
            else:
                current_line.append(word)
                current_length += len(word) + 1

        if current_line:
            lines.append(" ".join(current_line))
        
        return "\n".join(lines)

    # Split the text by newline characters
    segments = text.split('\n')
    
    # Apply line breaking to each segment
    broken_segments = [break_line(segment, max_length) for segment in segments]
    
    # Reassemble the text with the original newlines
    return "\n".join(broken_segments)


def plot_graphs(occupation, 
                df1, df2, 
                df1_comment, 
                df2_comment,
                df1_unique_color, df2_unique_color, 
                graph_title,
                save_path,
                plot_removed_edges=True):
    # Compare two graphs
    common, unique_to_df1, unique_to_df2, flag_entries_df1, flag_entries_df2 = compare_graphs(df1, df2)

    # Break down long comments into multiple lines
    unique_to_df1['comment'] = unique_to_df1['comment'].apply(lambda x: insert_line_breaks(x))
    unique_to_df2['comment'] = unique_to_df2['comment'].apply(lambda x: insert_line_breaks(x))
    flag_entries_df1['comment'] = flag_entries_df1['comment'].apply(lambda x: insert_line_breaks(x[:-19])) # 19 is length of label 'TriangleRemovedFlag'
    flag_entries_df2['comment'] = flag_entries_df2['comment'].apply(lambda x: insert_line_breaks(x[:-19]))

    # Create a Pyvis network
    net = Network(notebook=True, directed=True, cdn_resources="remote",
                height = "800px",
                    width = "125%",
                    select_menu = False,
                    filter_menu = False,)

    # Add nodes with fixed positions and labels
    fixed_positions = node_positions(occupation)
    for node, (x, y) in fixed_positions.items():
        net.add_node(node, label=node.split(" ")[0], title=node, x=x, y=y, fixed=True, borderWidthSelected=10)

    # Add edges with corresponding labels
    for index, row in common.iterrows():
        net.add_edge(row['source'], row['target'], title='In both', color='darkgrey')
    for index, row in unique_to_df1.iterrows():
        net.add_edge(row['source'], row['target'], title=df1_comment + row['comment'], color=df1_unique_color)
    for index, row in unique_to_df2.iterrows():
        net.add_edge(row['source'], row['target'], title=df2_comment + row['comment'], color=df2_unique_color)

    # Add pseudo edges for removed triangle edges with dashed lines
    if plot_removed_edges == True:
        if not flag_entries_df1.empty:
            for index, row in flag_entries_df1.iterrows():
                net.add_edge(row['source'], row['target'], title='Triangle edge removed:\n' + row['comment'], color='green', dashes=True)
        if not flag_entries_df2.empty:
            for index, row in flag_entries_df2.iterrows():
                net.add_edge(row['source'], row['target'], title='Triangle edge removed:\n' + row['comment'], color='orange', dashes=True)

    # Save interactive graph
    net.save_graph(save_path)

    # Read the saved HTML file and insert the title
    with open(save_path, 'r') as file:
        html_content = file.read()

    # Insert title in the HTML content
    title_html = f'<center><h2 style="font-size: 30px;">{graph_title}</h2></center>'
    html_content = html_content.replace("<body>", f"<body>{title_html}", 1)

    # Write the updated HTML content back to the file
    with open(save_path, 'w') as file:
        file.write(html_content)