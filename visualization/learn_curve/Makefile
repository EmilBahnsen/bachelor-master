.PHONY: all
all: learn_curve-29-29-29.pdf learn_curve-68-68.pdf learn_curve-50.pdf learn_curve-29-29-29_dual_point_and_relaxed.pdf learn_curve-29-29-29_single_point_and_relaxed.pdf

learn_curve-50.pdf: learn_curve-50.txt
	python learn_curve_plot.py $< $@ "Learning curve for 13-50-1"
learn_curve-50.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc2.73/z-score/50_ba5pct/struc*/*)
	python learn_curve_save.py '~/carbon_nn/logs/nn_logs/features_few/Rc2.73/z-score/50_ba5pct/struc*/*' $@ 300000

learn_curve-68-68.pdf: learn_curve-68-68.txt
	python learn_curve_plot.py $< $@ "Learning curve for 13-68-68-1"
learn_curve-68-68.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc5/z-score/68-68_ba5pct/struc*/*)
	python learn_curve_save.py '~/carbon_nn/logs/nn_logs/features_few/Rc5/z-score/68-68_ba5pct/struc*/*' $@ 300000

learn_curve-29-29-29.pdf: learn_curve-29-29-29.txt
	python learn_curve_plot.py $< $@ "Learning curve for 13-29-29-29-1"
learn_curve-29-29-29.txt: $(wildcard ~/carbon_nn/logs/nn_logs/features_few/Rc5/z-score/29-29-29_ba5pct/struc*/*)
	python learn_curve_save.py '~/carbon_nn/logs/nn_logs/features_few/Rc5/z-score/29-29-29_ba5pct/struc*/*' $@ 300000

learn_curve-29-29-29_dual_point_and_relaxed.pdf: learn_curve-29-29-29_dual_point_and_relaxed.txt
	python learn_curve_plot.py $< $@ "Learning curve for 13-29-29-29-1 dual point"
learn_curve-29-29-29_dual_point_and_relaxed.txt: $(wildcard ~/carbon_nn/logs/mixed_log/non_relaxed_double0.8_relaxed0.2/features_few/Rc5/z-score/29-29-29_ba5pct/struc*/*)
	python learn_curve_save.py '~/carbon_nn/logs/mixed_log/non_relaxed_double0.8_relaxed0.2/features_few/Rc5/z-score/29-29-29_ba5pct/struc*/*' $@ 500000

learn_curve-29-29-29_single_point_and_relaxed.pdf: learn_curve-29-29-29_single_point_and_relaxed.txt
	python learn_curve_plot.py $< $@ "Learning curve for 13-29-29-29-1 single point"
learn_curve-29-29-29_single_point_and_relaxed.txt: $(wildcard ~/carbon_nn/logs/mixed_log/non_relaxed_single0.8_relaxed0.2/features_few/Rc5/z-score/29-29-29_ba5pct/struc*/*)
	python learn_curve_save.py '~/carbon_nn/logs/mixed_log/non_relaxed_single0.8_relaxed0.2/features_few/Rc5/z-score/29-29-29_ba5pct/struc*/*' $@ 500000

.PHONY: clean
clean:
	$(RM) *.txt *.pdf
