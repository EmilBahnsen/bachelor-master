.PHONY: all
all: n_MAE_HL1.pdf n_MAE_HL2.pdf n_MAE_HL3.pdf

n_MAE_HL3.pdf: n_MAE_HL3.txt
	python n_MAE_plot.py $< $@ "13-N-N-N-1 network after 500k itterations"
n_MAE_HL3.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc5/z-score/*-*-*_ba0.5k/*{00.28,20.58,00.13})
	python n_MAE_save.py '~/carbon_nn/logs/nn_logs/features_few/Rc5/z-score/*-*-*_ba0.5k/*{00.28,20.58,00.13}' $@ 0

n_MAE_HL2.pdf: n_MAE_HL2.txt
	python n_MAE_plot.py $< $@ "13-N-N-1 network after 500k itterations"
n_MAE_HL2.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc5/z-score/??-??_ba0.5k/*{00.33,20.48})
	python n_MAE_save.py '~/carbon_nn/logs/nn_logs/features_few/Rc5/z-score/??-??_ba0.5k/*{00.33,20.48}' $@ 0

n_MAE_HL1.pdf: n_MAE_HL1.txt
	python n_MAE_plot.py $< $@ "13-N-1 network after 500k itterations"
n_MAE_HL1.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc2.73/z-score/*_ba0.5k/*)
	python n_MAE_save.py '~/carbon_nn/logs/nn_logs/features_few/Rc2.73/z-score/*_ba0.5k/*' $@ 0

.PHONY: clean
clean:
	$(RM) *.txt *.pdf
