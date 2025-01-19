ts := $(shell /bin/date "+%Y-%m-%d---%H-%M-%S")
project_name = rubin

tables = 
plots = 


# ../computed_objects/simulation_results.csv: ../analysis/min_cost.py
# 	cd ../analysis && python min_cost.py

# tables += tables/simulation_results.tex
# tables/simulation_results.tex: ../analysis/table_simulation_results.R ../computed_objects/simulation_results.csv
# 	cd ../analysis && Rscript table_simulation_results.R

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
plots/two_tasks.pdf: ../analysis/create_two_task_plot.py ../analysis/CostVisualization.py
	 cd ../analysis && python create_two_task_plot.py

plots += plots/optimal_paths.pdf
plots/optimal_paths.pdf: ../analysis/create_optimal_paths.py ../analysis/CostVisualization.py
	 cd ../analysis && python create_optimal_paths.py


plots += plots/horizon_effects.pdf
plots/horizon_effects.pdf: ../analysis/create_horizon_effects.py ../analysis/CostVisualization.py
	 cd ../analysis && python create_horizon_effects.py

plots += plots/job_design.png
plots/job_design.png: ../analysis/visualize_job.py
	cd ../analysis && python visualize_job.py

plots += plots/combined_grid.png
plots/combined_grid.png: ../analysis/combined_grid.py
	cd ../analysis && python combined_grid.py

## New Agentic AI stuff 

plots += plots/indiv_occupation_cooccurrence_score_histogram.png
plots += plots/indiv_task_cooccurrence_score_histogram.png
plots += plots/task_pair_counts_histogram.png
plots += plots/task_pair_weightedScore_histogram.png

plots/indiv_occupation_cooccurrence_score_histogram.png: ../analysis/onet_task_adjacency.py ../data/onet_occupations_yearly.csv
	cd ../analysis && python onet_task_adjacency.py

plots/indiv_task_cooccurrence_score_histogram.png: ../analysis/onet_task_adjacency.py ../data/onet_occupations_yearly.csv
	cd ../analysis && python onet_task_adjacency.py

plots/task_pair_counts_histogram.png: ../analysis/onet_task_adjacency.py ../data/onet_occupations_yearly.csv
	cd ../analysis && python onet_task_adjacency.py

plots/task_pair_weightedScore_histogram.png: ../analysis/onet_task_adjacency.py ../data/onet_occupations_yearly.csv
	cd ../analysis && python onet_task_adjacency.py

# The placebo analysis plots

plots += plots/placebo/indiv_occupation_cooccurrence_score_histogram.png
plots += plots/placebo/indiv_task_cooccurrence_score_histogram.png
plots += plots/placebo/task_pair_counts_histogram.png
plots += plots/placebo/task_pair_weightedScore_histogram.png

plots/placebo/indiv_occupation_cooccurrence_score_histogram.png: ../analysis/onet_task_adjacency_placebo.py ../data/onet_occupations_yearly.csv
	cd ../analysis && python onet_task_adjacency_placebo.py

plots/placebo/indiv_task_cooccurrence_score_histogram.png: ../analysis/onet_task_adjacency_placebo.py ../data/onet_occupations_yearly.csv
	cd ../analysis && python onet_task_adjacency_placebo.py

plots/placebo/task_pair_counts_histogram.png: ../analysis/onet_task_adjacency_placebo.py ../data/onet_occupations_yearly.csv
	cd ../analysis && python onet_task_adjacency_placebo.py

plots/placebo/task_pair_weightedScore_histogram.png: ../analysis/onet_task_adjacency_placebo.py ../data/onet_occupations_yearly.csv
	cd ../analysis && python onet_task_adjacency_placebo.py


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
	cp $(project_name).pdf ~/Dropbox/toread/$(project_name)_$(ts).pdf 
