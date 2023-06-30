ts := $(shell /bin/date "+%Y-%m-%d---%H-%M-%S")
project_name = rubin

tables = 
plots = 


../computed_objects/simulation_results.csv: ../analysis/min_cost.py
	cd ../analysis && python min_cost.py

tables += tables/simulation_results.tex
tables/simulation_results.tex: ../analysis/table_simulation_results.R ../computed_objects/simulation_results.csv
	cd ../analysis && Rscript table_simulation_results.R

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


$(project_name).pdf : $(project_name).tex $(project_name).bib ./parameters.tex $(LEVEL_PLOTS) $(LINE_PLOTS) $(WRAPPERS) $(COMBINED_TABLES) $(plots) $(tables)
	pdflatex -shell-escape -interaction=batchmode $(project_name)
	pdflatex -shell-escape -interaction=batchmode $(project_name)
	bibtex $(project_name) 
	pdflatex -shell-escape -interaction=batchmode $(project_name)
	pdflatex -shell-escape -interaction=batchmode $(project_name)
	gs -q -dNOPAUSE -dBATCH -dPDFSETTINGS=/prepress -sDEVICE=pdfwrite -sOutputFile=tmp.pdf $(project_name).pdf
	mv tmp.pdf $(project_name).pdf


toread: $(project_name).pdf
	scp $(project_name).pdf bandit-local.local:/home/john/Dropbox/toread/$(project_name)_$(ts).pdf || cp $(project_name).pdf ~/Dropbox/toread/$(project_name)_$(ts).pdf 
