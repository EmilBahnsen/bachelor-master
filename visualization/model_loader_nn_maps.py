import sys
import glob
import os
import ast
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from model_loader import *

def draw_nn_map_figure(model_dir):
    ml = ModelLoader(model_dir)
    _,step,AME      = ml.get_scalar('loss/mean_absolute_error_1')
    _,step,AME_test = ml.get_scalar('loss/mean_absolute_error_test_1')

    plt.suptitle(model_dir)
    ax = plt.subplot(121)
    plt.title("$w_{in}$")
    w_in = ml.get_values_of_variable_by_name('fc_in/weights:0')
    w_max = np.amax(np.absolute(w_in))
    #sns.heatmap(w_in, linewidth=0.5, cbar_kws={"orientation": "horizontal"})
    im = plt.imshow(w_in, cmap='seismic', interpolation='nearest', vmax=w_max, vmin=-w_max)
    cbar = fig.colorbar(ax=ax, mappable=im, orientation='horizontal')

    ax = plt.subplot(164)
    plt.title("$b_{in}$")
    b_in = ml.get_values_of_variable_by_name('fc_in/biases:0')
    b_in = np.expand_dims(b_in, axis=1)
    #sns.heatmap(b_in, linewidth=0.5, xticklabels=False, cbar_kws={"orientation": "horizontal"})
    im = plt.imshow(b_in, cmap='seismic', interpolation='nearest')
    cbar = fig.colorbar(ax=ax, mappable=im, orientation='horizontal')

    ax = plt.subplot(165)
    plt.title("$w_{out}$")
    w_out = ml.get_values_of_variable_by_name('fc_out/weights:0')
    #sns.heatmap(w_out, linewidth=0.5, xticklabels=False, cbar_kws={"orientation": "horizontal"})
    im = plt.imshow(w_out, cmap='seismic', interpolation='nearest')
    cbar = fig.colorbar(ax=ax, mappable=im, orientation='horizontal')
    
    ax = plt.subplot(166)
    plt.title("$b_{out}$")
    b_out = ml.get_values_of_variable_by_name('fc_out/biases:0')
    b_out = np.expand_dims(b_out, axis=1)
    #sns.heatmap(b_out, linewidth=0.5, xticklabels=False, yticklabels=False, annot=True, fmt="f", cbar=False)
    im = plt.imshow(b_out, cmap='seismic', interpolation='nearest')
    cbar = fig.colorbar(ax=ax, mappable=im, orientation='horizontal')

from bash_file_tools import *

# usage: python model_loader_nn_maps.py '../logs/nn_logs/features_few/Rc*/z-score/*/*' model_nn_maps/
model_dirs_bash_path = sys.argv[1]
out_dir = sys.argv[2]
model_dirs = list_bash_files(model_dirs_bash_path)

fig = plt.figure(1, figsize=(16, 8.5))

for i,model_dir in enumerate(model_dirs):
    fig.clf()
    draw_nn_map_figure(model_dir)
    Rc = model_dir
    fig_save_path = os.path.join(out_dir, "nn_map_" + str(i) + ".pdf")
    fig.savefig(fig_save_path)
    print(fig_save_path, "saved.")


