# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 21:00:43 2021

@author: Gabe Carvalho
"""

import json, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def normalize(img):
    img_min = np.amin(img)
    img_max = np.amax(img)
    return (img - img_min) / img_max

def main():
    parser = argparse.ArgumentParser(description='DeepMerge Data Mosaic Maker',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--settings', type=str, default='mosaic_settings.json', help='path to settings json', nargs=1)
    args = parser.parse_args()

    with open(args.settings) as f:
        params = json.load(f)
    
    data_path = params['data_path']
    dataset = params['dataset']
    grid_width = params['grid_width']
    grid_height = params['grid_height']
    start = params['start_index']
    log_scale = params['log_scale']
    out_path = params['out_path']
    
    if dataset == 0:
        X = np.load(data_path + 'SimReal_SOURCE_X_Illustris0.npy')
        Y = np.load(data_path + 'SimReal_SOURCE_y_Illustris0.npy')
        vmin = 0.02
    elif dataset == 1:
        X = np.load(data_path + 'SimReal_TARGET_x_postmergers_SDSS.npy')
        Y = np.load(data_path + 'SimReal_TARGET_y_postmergers_SDSS.npy')
        vmin = 0.03
    elif dataset == 2:
        X = np.load(data_path + 'SimSim_SOURCE_X_Illustris2_pristine.npy')
        Y = np.load(data_path + 'SimSim_SOURCE_y_Illustris2_pristine.npy')
        vmin = 0.000002
    elif dataset == 3:
        X = np.load(data_path + 'SimSim_TARGET_X_Illustris2_noisy.npy')
        Y = np.load(data_path + 'SimSim_TARGET_y_Illustris2_noisy.npy')
        vmin = 0.001
    else:
        return
    
    f, axarr = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height), dpi=300)
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    for i in range(grid_height):
        for j in range(grid_width):
            img = normalize(X[start+i*grid_width+j, 0, :, :])
            label = Y[start+i*grid_width+j]
            if grid_height > 1:
                if log_scale == True:
                    axarr[i, j].imshow(img, label=label, interpolation='antialiased', norm=LogNorm(vmin=vmin, clip=True), cmap='inferno')
                else:
                    axarr[i, j].imshow(img, label=label, interpolation='antialiased', cmap='inferno')
                axarr[i, j].axis('off')
            else:
                if log_scale == True:
                    axarr[j].imshow(img, label=label, interpolation='antialiased', norm=LogNorm(vmin=vmin, clip=True), cmap='inferno')
                else:
                    axarr[j].imshow(img, label=label, interpolation='antialiased', cmap='inferno')
                axarr[j].axis('off')
    
    plt.savefig(out_path, bbox_inches='tight',pad_inches = 0)
    
if __name__ == '__main__':
    main()
