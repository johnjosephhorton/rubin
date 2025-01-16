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
    indiffCurve_folder = f'{data_path}/daily_tasks_occupations_analysis/{occupation}/indiffCurves'

    # create folder in directory if it doesn't exist
    if not os.path.exists(occupation_folder):
        os.makedirs(occupation_folder)
    if not os.path.exists(indiffCurve_folder):
        os.makedirs(indiffCurve_folder)

    return GPT_input_occupation, plot_title_occupation, occupation_code, occupation_folder




