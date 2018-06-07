.PHONY: all
all: n_MAE_HL1.pdf n_MAE_HL2.pdf n_MAE_HL3.pdf n_MAE_HL3_1.pdf n_MAE_HL3_2.pdf n_MAE_HL3_3.pdf

n_MAE_HL3_1.pdf: n_MAE_HL3_1.txt
	python n_MAE_plot.py $< $@ "13-N-29-29-1 network after 500k itterations"
n_MAE_HL3_1.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc5/z-score/*-29-29_ba0.5k/*)
	python n_MAE_save.py '~/carbon_nn/logs/nn_logs/features_few/Rc5/z-score/*-29-29_ba0.5k/*' $@ 500000

n_MAE_HL3_2.pdf: n_MAE_HL3_2.txt
	python n_MAE_plot.py $< $@ "13-29-N-29-1 network after 500k itterations"
n_MAE_HL3_2.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc5/z-score/29-*-29_ba0.5k/*)
	python n_MAE_save_HL3_2.py '~/carbon_nn/logs/nn_logs/features_few/Rc5/z-score/29-*-29_ba0.5k/*' $@ 500000

n_MAE_HL3_3.pdf: n_MAE_HL3_3.txt
	python n_MAE_plot.py $< $@ "13-29-29-N-1 network after 500k itterations"
n_MAE_HL3_3.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc5/z-score/29-29-*_ba0.5k/*)
	python n_MAE_save_HL3_3.py '~/carbon_nn/logs/nn_logs/features_few/Rc5/z-score/29-29-*_ba0.5k/*' $@ 500000

n_MAE_HL3.pdf: n_MAE_HL3.txt
	python n_MAE_plot.py $< $@ "13-N-N-N-1 network after 500k itterations"
n_MAE_HL3.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc5/z-score/*-*-*_ba0.5k/*{00.28,20.58,00.13,2018-06-01_01.06,2018-05-31_12.55})
	python n_MAE_save.py '~/carbon_nn/logs/nn_logs/features_few/Rc5/z-score/*-*-*_ba0.5k/*{00.28,20.58,00.13,2018-06-01_01.06}' $@ 500000

n_MAE_HL2.pdf: n_MAE_HL2.txt
	python n_MAE_plot.py $< $@ "13-N-N-1 network after 500k itterations"
n_MAE_HL2.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc5/z-score/??-??_ba0.5k/*{00.33,20.48})
	python n_MAE_save.py '~/carbon_nn/logs/nn_logs/features_few/Rc5/z-score/??-??_ba0.5k/*{00.33,20.48}' $@ 0

n_MAE_HL1.pdf: n_MAE_HL1.txt
	python n_MAE_plot.py $< $@ "13-N-1 network after 500k itterations"
n_MAE_HL1.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc2.73/z-score/*_ba0.5k/*)
	python n_MAE_save.py '~/carbon_nn/logs/nn_logs/features_few/Rc2.73/z-score/*_ba0.5k/*' $@ 0

n_MAE_HL3_double_relax.pdf: n_MAE_HL3_double_relax.txt
	python n_MAE_plot.py $< $@ "13-N-N-N-1 network after 500k itterations"
n_MAE_HL3_double_relax.txt: $(wildcard ~/carbon_nn/logs/mixed_log/non_relaxed_double0.8_relaxed0.2/features_few/Rc5/z-score/*-*-*_ba0.5k/*{2018-06-05_17.14})
	python n_MAE_save.py '~/carbon_nn/logs/mixed_log/non_relaxed_double0.8_relaxed0.2/features_few/Rc5/z-score/*-*-*_ba0.5k/*{2018-06-05_17.14,}' $@ 0

.PHONY: clean
clean:
	$(RM) *.txt *.pdf
