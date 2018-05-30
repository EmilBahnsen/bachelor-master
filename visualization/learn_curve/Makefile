.PHONY: all
all: learn_curve-29-29-29.pdf learn_curve-68-68.pdf learn_curve-50.pdf

learn_curve-50.pdf: learn_curve-50.txt
	python learn_curve_plot.py $< $@ "Learning curve for 13-50-1"
learn_curve-50.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc2.73/z-score/50_ba5pct/struc*/*)
	python learn_curve_save.py '~/carbon_nn/logs/nn_logs/features_few/Rc2.73/z-score/50_ba5pct/struc*/*' $@ 0

learn_curve-68-68.pdf: learn_curve-68-68.txt
	python learn_curve_plot.py $< $@ "Learning curve for 13-68-68-1"
learn_curve-68-68.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc5/z-score/68-68_ba5pct/struc*/*)
	python learn_curve_save.py '~/carbon_nn/logs/nn_logs/features_few/Rc5/z-score/68-68_ba5pct/struc*/*' $@ 0

learn_curve-29-29-29.pdf: learn_curve-29-29-29.txt
	python learn_curve_plot.py $< $@ "Learning curve for 13-29-29-29-1"
learn_curve-29-29-29.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc5/z-score/29-29-29_ba5pct/struc*/*)
	python learn_curve_save.py '~/carbon_nn/logs/nn_logs/features_few/Rc5/z-score/29-29-29_ba5pct/struc*/*' $@ 0

.PHONY: clean
clean:
	$(RM) *.txt *.pdf
