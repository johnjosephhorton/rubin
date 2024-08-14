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
        GPT_input_occupation = 'pile dirver operator'
        plot_title_occupation = 'Pile Driver Operators'
        occupation_code = '47-2072'
    elif occupation == 'dredgeOperators':
        GPT_input_occupation = 'dredge operators'
        plot_title_occupation = 'Dredge Operators'
        occupation_code = '53-7031'
    elif occupation == 'gradersAndSorters':
        GPT_input_occupation = 'Graders and sorters for agricultural products'
        plot_title_occupation = 'Graders and Sorters for Agricultural Products'
        occupation_code = '45-2041'
    elif occupation == 'reinforcingIron':
        GPT_input_occupation = 'Reinforcing iron and rebar working'
        plot_title_occupation = 'Reinforcing Iron and Rebar Workers'
        occupation_code = '47-2171'
    elif occupation == 'insuranceAppraisers':
        GPT_input_occupation = 'Insurance appraisers for auto damage'
        plot_title_occupation = 'Insurance Appraisers, Auto Damage'
        occupation_code = '13-1032'
    elif occupation == 'floorSanders':
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
    elif occupation == 'shampooers':
        GPT_input_occupation = 'shampooers'
        plot_title_occupation = 'Shampooers'
        occupation_code = '39-5093'
    
    occupation_folder = f'{data_path}/daily_tasks_occupations_analysis/{occupation}'
    return GPT_input_occupation, plot_title_occupation, occupation_code, occupation_folder



