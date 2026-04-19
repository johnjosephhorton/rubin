# ===========================================================================
# Notebook / script -> writeup/ output dependency map
# (documentation only; not build rules -- use git log for provenance)
#
# Auto-generated 2026-04-19 from analysis/old/*.ipynb, *.py, *.R
# Paths with {placeholder} segments are constructed dynamically at runtime.
# ===========================================================================

# analysis/GPT_task_sequences_overlap.ipynb
#   -> writeup/plots/GPT_task_sequences_overlap_analysis
#
# analysis/old/bls_sector_shares.ipynb
#   -> writeup/plots
#
# analysis/combined_grid.ipynb
#   -> writeup/plots/combined_grid_{my_str}.png
#   -> writeup/plots/job_design
#   -> writeup/plots/job_design.png
#   -> writeup/plots/job_design/job_design_{index}.png
#   -> writeup/plots/job_design/job_design_{index}_{my_str}.png
#   -> writeup/plots/tent_pole.png
#
# analysis/old/cooccurring_tasks_analysis.py
#   -> writeup/plots
#
# analysis/old/create_horizon_effects.py
#   -> writeup/plots/horizon_effects.pdf
#
# analysis/old/create_optimal_paths.py
#   -> writeup/plots/optimal_paths.pdf
#
# analysis/old/create_two_task_plot.py
#   -> writeup/plots/two_tasks.pdf
#
# analysis/old/example5_costCurves.py
#   -> writeup/plots
#
# analysis/example5_plot.ipynb
#   -> writeup/plots
#
# analysis/old/heterogeneity_dist_plot.ipynb
#   -> writeup/lyx/phi_distribution_{g//2 + 1}.png
#
# analysis/old/job_design.ipynb
#   -> writeup/plots/job_design/job_design_{num_jobs}tasks_{example}.png
#
# analysis/old/job_design_visualization.py
#   -> writeup/plots/combined_grid_{my_str}.png
#   -> writeup/plots/job_design
#   -> writeup/plots/job_design.png
#   -> writeup/plots/job_design/job_design_{index}.png
#   -> writeup/plots/job_design/job_design_{index}_{my_str}.png
#   -> writeup/plots/tent_poll.png
#
# analysis/onet_GPT_aiExposure.ipynb
#   -> writeup/plots/GPT_aiExposure
#
# analysis/old/onet_adHoc_stats.ipynb
#   -> writeup/plots
#
# analysis/old/onet_antrhopicIndex.ipynb
#   -> writeup/plots/anthropic_AI_index
#
# analysis/onet_antrhopicIndex_execTypeVaryingDWA.ipynb
#   -> writeup/plots/execTypeVaryingDWA{path_suffix}
#
# analysis/old/onet_antrhopicIndex_execTypeVaryingDWA_additionalAnalyses.ipynb
#   -> writeup/plots/execTypeVaryingDWA_{folder_path_suffix}
#
# analysis/onet_antrhopicIndex_execTypeVaryingDWA_robustness.ipynb
#   -> writeup/plots/execTypeVaryingDWA{path_suffix}_robustness
#
# analysis/old/onet_antrhopicIndex_industryWeighted.ipynb
#   -> writeup/plots/anthropic_AI_index/BLS_ONET_matchedEmpShares
#
# analysis/old/onet_antrhopicIndex_industryWeighted_handoffMeasure.ipynb
#   -> writeup/plots/anthropic_AI_index/BLS_ONET_matchedEmpShares
#
# analysis/old/onet_antrhopicIndex_wageAndTime.ipynb
#   -> writeup/plots/anthropic_AI_index/wageAndTime_measures
#
# analysis/onet_chainLength.ipynb
#   -> writeup/plots/aiChain_length_count
#
# analysis/onet_chainLength_robustness.ipynb
#   -> writeup/plots/aiChain_length_count_robustness
#
# analysis/onet_cleaning.ipynb
#   -> writeup/plots
#
# analysis/old/onet_contiguityMeasure.ipynb
#   -> writeup/plots/contiguityMeasure
#
# analysis/old/onet_dwa_level_exposure.ipynb
#   -> writeup/plots/taskcounts_aiexposure
#
# analysis/onet_eloundou_anthropic_GPTsequence_mergeReshuffle.ipynb
#   -> writeup/plots/ONET_Eloundou_Anthropic_GPT
#   -> writeup/plots/ONET_Eloundou_Anthropic_GPT_frequent
#
# analysis/onet_fragmentationIndex.ipynb
#   -> writeup/plots/fragmentationIndex
#   -> writeup/plots/fragmentationIndex_frequent
#
# analysis/onet_fragmentationIndex_robustness.ipynb
#   -> writeup/plots/fragmentationIndex_robustness
#
# analysis/old/onet_my_naiveAIClassification.ipynb
#   -> writeup/plots/my_naiveAIClassification
#
# analysis/old/onet_numTechnologies.ipynb
#   -> writeup/plots/anthropic_AI_index/BLS_ONET_matchedEmpShares
#
# analysis/old/onet_occSequence_vs_antrhopicIndex.ipynb
#   -> writeup/plots/taskSequence_vs_anthropicIndex
#
# analysis/old/onet_occWage_vs_taskCount.ipynb
#   -> writeup/plots/occWage_vs_taskCount
#
# analysis/old/onet_similarTasksDWA.ipynb
#   -> writeup/plots
#
# analysis/old/onet_taskCosineSimilarity.ipynb
#   -> writeup/plots/ONET_Eloundou_Anthropic_GPT
#   -> writeup/plots/ONET_Eloundou_Anthropic_GPT_frequent
#
# analysis/old/onet_taskPerOccupation_measure.ipynb
#   -> writeup/plots
#
# analysis/onet_taskSequence_vs_antrhopicIndex.ipynb
#   -> writeup/plots/taskSequence_vs_anthropicIndex
#
# analysis/old/onet_task_adjacency.py
#   -> writeup/plots
#
# analysis/onet_task_and_occ_sequence.ipynb
#   -> writeup/plots
#
# analysis/old/onet_task_and_occ_sequence_detailed.ipynb
#   -> writeup/plots
#
# analysis/old/onet_task_and_occ_sequence_frequentTasksOnly.ipynb
#   -> writeup/plots
#
# analysis/onet_task_and_occ_sequence_robustness.ipynb
#   -> writeup/plots
#
# analysis/old/onet_task_durations.ipynb
#   -> writeup/plots
#
# analysis/old/table_simulation_results.R
#   -> writeup/plots/
#   -> writeup/tables/simulation_results.tex
#

ts := $(shell /bin/date "+%Y-%m-%d---%H-%M-%S")
project_name = rubin

tables = 
plots = 


parameters.tex:
	touch parameters.tex

../backups/$(project_name)_$(ts).tar.gz:
	tar --exclude='../backups' --exclude='../.git' -zcvf ../backups/$(project_name)_$(ts).tar.gz ../

backup: ../backups/$(project_name)_$(ts).tar.gz
	echo "Making backup"

backup_s3: ../backups/$(project_name)_$(ts).tar.gz
	aws s3 cp ../backups/$(project_name)_$(ts).tar.gz s3://john_horton_backups/

snapshot: $(project_name).pdf
	@echo Timestamp is $(ts)
	cp $(project_name).pdf ../snapshots/$(project_name)_$(ts).pdf

website: $(project_name).pdf
	printf 'cd papers\nput $<\n' \ | sftp -oPort=2222 johnjhorton@104.198.96.220

purge:
	cd plots && find . -maxdepth 1 -type f -print0 | xargs -0 -I {} mv {} unused && touch PLACEHOLDER.txt
	cd tables && find . -maxdepth 1 -type f -print0 | xargs -0 -I {} mv {} unused && touch PLACEHOLDER.txt
	cd ../computed_objects && find . -maxdepth 1 -type f -print0 | xargs -0 -I {} mv {} unused && touch PLACEHOLDER.txt
	cd parameters && find . -maxdepth 1 -type f -print0 | xargs -0 -I {} mv {} unused && touch PLACEHOLDER.txt

plots += plots/two_tasks.pdf
plots/two_tasks.pdf: ../analysis/old/create_two_task_plot.py ../analysis/old/CostVisualization.py
	 cd ../analysis/old && python create_two_task_plot.py

plots += plots/optimal_paths.pdf
plots/optimal_paths.pdf: ../analysis/old/create_optimal_paths.py ../analysis/old/CostVisualization.py
	 cd ../analysis/old && python create_optimal_paths.py


plots += plots/horizon_effects.pdf
plots/horizon_effects.pdf: ../analysis/old/create_horizon_effects.py ../analysis/old/CostVisualization.py
	 cd ../analysis/old && python create_horizon_effects.py

plots += plots/tent_poll.png
plots/tent_poll.png: ../analysis/old/job_design_visualization.py
	cd ../analysis/old && python job_design_visualization.py

plots += plots/job_design.png
plots/job_design.png: ../analysis/old/job_design_visualization.py
	cd ../analysis/old && python job_design_visualization.py

plots += plots/job_design_visualization.png
plots/job_design_visualization.png: ../analysis/old/job_design_visualization.py
	cd ../analysis/old && python job_design_visualization.py

## New Agentic AI stuff 

plots += plots/indiv_task_cooccurrence_score_histogram.png
plots += plots/indiv_occupation_cooccurrence_score_histogram.png
plots += plots/occupation_similarity_score_histogram.png
plots += plots/task_pair_counts_histogram.png

plots += plots/highly_cooccurring_tasks_task_repetition_count.png
plots += plots/highly_cooccurring_tasks_occupation_repetition_count.png
plots += plots/occupation_group_common_tasks_heatmap.png


plots/indiv_task_cooccurrence_score_histogram.png: ../analysis/old/onet_task_adjacency.py ../data/onet_occupations_yearly.csv
	cd ../analysis/old && python onet_task_adjacency.py

plots/indiv_occupation_cooccurrence_score_histogram.png: ../analysis/old/onet_task_adjacency.py ../data/onet_occupations_yearly.csv
	cd ../analysis/old && python onet_task_adjacency.py

plots/occupation_similarity_score_histogram.png: ../analysis/old/onet_task_adjacency.py ../data/onet_occupations_yearly.csv
	cd ../analysis/old && python onet_task_adjacency.py

plots/task_pair_counts_histogram.png: ../analysis/old/onet_task_adjacency.py ../data/onet_occupations_yearly.csv
	cd ../analysis/old && python onet_task_adjacency.py


plots/highly_cooccurring_tasks_task_repetition_count.png: ../analysis/old/cooccurring_tasks_analysis.py ../data/onet_occupations_yearly.csv ../data/SOC_Structure.csv 
	cd ../analysis/old && python cooccurring_tasks_analysis.py

plots/highly_cooccurring_tasks_occupation_repetition_count.png: ../analysis/old/cooccurring_tasks_analysis.py ../data/onet_occupations_yearly.csv ../data/SOC_Structure.csv 
	cd ../analysis/old && python cooccurring_tasks_analysis.py

plots/occupation_group_common_tasks_heatmap.png: ../analysis/old/cooccurring_tasks_analysis.py ../data/onet_occupations_yearly.csv ../data/SOC_Structure.csv 
	cd ../analysis/old && python cooccurring_tasks_analysis.py


$(project_name).pdf : $(project_name).tex $(project_name).bib ./parameters.tex $(LEVEL_PLOTS) $(LINE_PLOTS) $(WRAPPERS) $(COMBINED_TABLES) $(plots) $(tables)
	pdflatex -shell-escape -interaction=batchmode $(project_name)
	pdflatex -shell-escape -interaction=batchmode $(project_name)
	bibtex $(project_name) 
	pdflatex -shell-escape -interaction=batchmode $(project_name)
	pdflatex -shell-escape -interaction=batchmode $(project_name)
	gs -q -dNOPAUSE -dBATCH -dPDFSETTINGS=/prepress -sDEVICE=pdfwrite -sOutputFile=tmp.pdf $(project_name).pdf
	mv tmp.pdf $(project_name).pdf


toread: $(project_name).pdf
	## scp $(project_name).pdf bandit-local.local:/home/john/Dropbox/toread/$(project_name)_$(ts).pdf || 
	##cp $(project_name).pdf ~/Dropbox/toread/$(project_name)_$(ts).pdf 
	cp $(project_name).pdf /Users/johnhorton/Library/CloudStorage/Dropbox/toread/$(project_name)_$(ts).pdf 
