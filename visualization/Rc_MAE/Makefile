.PHONY: all
all: Rc_MAE_50.pdf Rc_MAE_100.pdf Rc_MAE_150.pdf Rc_MAE_100-100.pdf Rc_MAE_100-100-100.pdf

Rc_MAE_100-100-100.pdf: Rc_MAE_100-100-100.txt
	python Rc_MAE_plot.py $< $@ "13-100-100-100-1 network after 150k itterations"
Rc_MAE_100-100-100.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc*/z-score/100-100-100_ba1k*/*)
	python Rc_MAE_save.py '~/carbon_nn/logs/nn_logs/features_few/Rc*/z-score/100-100-100_ba1k*/*' $@ 0

Rc_MAE_100-100.pdf: Rc_MAE_100-100.txt
	python Rc_MAE_plot.py $< $@ "13-100-100-1 network after 150k itterations"
Rc_MAE_100-100.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc*/z-score/100-100_ba1k*/*)
	python Rc_MAE_save.py '~/carbon_nn/logs/nn_logs/features_few/Rc*/z-score/100-100_ba1k*/*' $@ 0

Rc_MAE_150.pdf: Rc_MAE_150.txt
	python Rc_MAE_plot.py $< $@ "13-150-1 network after 150k itterations"
Rc_MAE_150.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc*/z-score/150_ba1k/*)
	python Rc_MAE_save.py '~/carbon_nn/logs/nn_logs/features_few/Rc*/z-score/150_ba1k/*' $@ 0

Rc_MAE_100.pdf: Rc_MAE_100.txt Rc_MAE_plot_100.py
	python Rc_MAE_plot_100.py $< $@ "13-100-1 network after 50k itterations"
Rc_MAE_100.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc*/z-score/100/*)
	python Rc_MAE_save.py "~/carbon_nn/logs/nn_logs/features_few/Rc*/z-score/100/* | sed -e 's/ /\n/g' | sed -e '/2018-05-26_11.29/d'" $@ 0

Rc_MAE_50.pdf: Rc_MAE_50.txt
	python Rc_MAE_plot.py $< $@ "13-50-1 network after 150k itterations"
Rc_MAE_50.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc*/z-score/50_ba1k/*)
	python Rc_MAE_save.py '~/carbon_nn/logs/nn_logs/features_few/Rc*/z-score/50_ba1k/*' $@ 0

.PHONY: clean
clean:
	$(RM) *.txt *.pdf
