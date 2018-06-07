.PHONY: all
all: Rc_MAE_50.pdf Rc_MAE_100_all_data.pdf Rc_MAE_100.pdf Rc_MAE_150.pdf Rc_MAE_100-100.pdf Rc_MAE_100-100-100.pdf Rc_MAE_29-29-29.pdf

Rc_MAE_29-29-29.pdf: Rc_MAE_29-29-29.txt
	python Rc_MAE_plot.py $< $@ "13-29-29-29-1 network after 200k itterations"
Rc_MAE_29-29-29.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc*/z-score/29-29-29_ba0.5k/*2018-05-30_20.05)
	python Rc_MAE_save.py '~/carbon_nn/logs/nn_logs/features_few/Rc*/z-score/29-29-29_ba0.5k/*2018-05-30_20.05' $@ 200000

Rc_MAE_100-100-100.pdf: Rc_MAE_100-100-100.txt
	python Rc_MAE_plot.py $< $@ "13-100-100-100-1 network after 150k itterations"
Rc_MAE_100-100-100.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc*/z-score/100-100-100_ba1k*/*{2018-05-28_18.25,2018-05-28_02.01})
	python Rc_MAE_save.py '~/carbon_nn/logs/nn_logs/features_few/Rc*/z-score/100-100-100_ba1k*/*{2018-05-28_18.25,2018-05-28_02.01}' $@ 150000

Rc_MAE_100-100.pdf: Rc_MAE_100-100.txt
	python Rc_MAE_plot.py $< $@ "13-100-100-1 network after 500k itterations"
Rc_MAE_100-100.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc*/z-score/100-100_ba1k*/*{2018-05-25_11.43,2018-05-28_18.19,2018-05-28_02.00})
	python Rc_MAE_save.py "~/carbon_nn/logs/nn_logs/features_few/Rc*/z-score/100-100_ba1k*/*{2018-05-25_11.43,2018-05-28_18.19,2018-05-28_02.00} | sed 's/ /\n/g' | sed '/Rc1\//d'" $@ 500000

Rc_MAE_150.pdf: Rc_MAE_150.txt
	python Rc_MAE_plot.py $< $@ "13-150-1 network after 150k itterations"
Rc_MAE_150.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc*/z-score/150_ba1k/*)
	python Rc_MAE_save.py '~/carbon_nn/logs/nn_logs/features_few/Rc*/z-score/150_ba1k/*' $@ 0

Rc_MAE_100.pdf: Rc_MAE_100.txt
	python Rc_MAE_plot_100_ba1k.py $< $@ "13-100-1 network after 500k itterations"
Rc_MAE_100.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc*/z-score/100_ba1k/*)
	python Rc_MAE_save.py "~/carbon_nn/logs/nn_logs/features_few/Rc*/z-score/100_ba1k/* | sed -e 's/ /\n/g' | sed -e '/2018-05-26_11.29/d' | sed '/Rc0.5/d' | sed '/Rc0.6/d' | sed '/Rc0.7/d' | sed '/Rc0.8/d' | sed '/Rc0.9/d' | sed '/Rc1\//d' | sed '/Rc1.1/d' | sed '/Rc1.2/d' | sed '/Rc1.3/d' | sed '/Rc1.4/d'" $@ 500000

Rc_MAE_100_all_data.pdf: Rc_MAE_100_all_data.txt Rc_MAE_plot_100.py
	python Rc_MAE_plot_100.py $< $@ "13-100-1 network after 50k itterations"
Rc_MAE_100_all_data.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc*/z-score/100/*)
	python Rc_MAE_save.py "~/carbon_nn/logs/nn_logs/features_few/Rc*/z-score/100/* | sed -e 's/ /\n/g' | sed -e '/2018-05-26_11.29/d'" $@ 50000

Rc_MAE_50.pdf: Rc_MAE_50.txt
	python Rc_MAE_plot.py $< $@ "13-50-1 network after 150k itterations"
Rc_MAE_50.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc*/z-score/50_ba1k/*)
	python Rc_MAE_save.py '~/carbon_nn/logs/nn_logs/features_few/Rc*/z-score/50_ba1k/*' $@ 0

.PHONY: clean
clean:
	$(RM) *.txt *.pdf
