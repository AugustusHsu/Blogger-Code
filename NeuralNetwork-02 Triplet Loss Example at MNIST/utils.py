# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:08:31 2020

@author: jimhs
"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
from args import parser

opts = parser()
def generate_and_save_images(epoch, method, triplet_net, test_x, test_y, pca):
    # Initial Log File
    fig_path = os.path.join(opts.PLOT_PATH, method)
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
        
    trip_vec = triplet_net(test_x, training=False)
    pca_vec = pca.transform(trip_vec)
    
    df = pd.DataFrame(data=pca_vec, columns=['pca-one','pca-two'])
    df['label'] = test_y
    
    # fig, ax = plt.subplots(figsize=(16,10))
    # ax.set_xlim([-0.8, 0.8])
    # ax.set_ylim([-0.8, 0.8])
    # colors = ['#6A6AFF', '#2894FF', '#00FFFF', '#1AFD9C', '#28FF28',
    #           '#EA0000', '#FF359A', '#003E3E', '#9F35FF', '#F9F900']
    # for i in range(test_x.shape[0]):
    #     imagebox = AnnotationBbox(
    #         OffsetImage(np.squeeze(test_x[i]), zoom=0.6, cmap=plt.cm.gray_r),
    #         xy=pca_vec[i], 
    #         bboxprops =dict(edgecolor=colors[test_y[i]]))
    #     ax.add_artist(imagebox)
    
    fig, ax = plt.subplots(figsize=(16,10))
    sns.scatterplot(x='pca-one', y='pca-two',
                    hue='label',
                    palette=sns.color_palette("hls", 10),
                    data=df,
                    legend="full",
                    alpha=0.8)
    plt.legend(loc='upper left')
    fig.suptitle('{}'.format(epoch), fontsize=70)
    # plt.show()
    path = os.path.join(fig_path, '{:03d}.png'.format(epoch))
    plt.savefig(path)
    plt.close('all')

def plot_line(df, col_name, method):
    # Initial Log File
    fig_path = os.path.join(opts.PLOT_LINE_PATH)
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
        
    fig, ax = plt.subplots(figsize=(16,10))
    sns.lineplot(data=df[col_name])
    
    fig.suptitle(col_name, fontsize=50)
    path = os.path.join(fig_path, method + '_' + col_name + '.png')
    plt.savefig(path)
    plt.close('all')

def generate_and_save_3d_images(triplet_net, test_x, test_y, view, method):
    # Initial Log File
    fig_path = os.path.join(opts.PLOT_3D_PATH, method)
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
        
    pca = PCA(n_components=3)
    pca_vec = pca.fit_transform(triplet_net(test_x, training=False))
    
    df = pd.DataFrame(data=pca_vec, columns=['pca-one','pca-two','pca-three'])
    df['label'] = test_y
    
    colors = ['#6A6AFF', '#2894FF', '#00FFFF', '#1AFD9C', '#28FF28',
              '#EA0000', '#FF359A', '#003E3E', '#9F35FF', '#F9F900']
    colors = np.array(colors)
    
    # plot
    fig = plt.figure(figsize=(32,20))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['pca-one'], 
               df['pca-two'], 
               df['pca-three'], 
               c=colors[df['label'].to_numpy()], 
               s=30,
               edgecolors='face')
    ax.view_init(elev=30,azim=view)
    # plt.show()
    path = os.path.join(fig_path, '{:03d}.png'.format(view))
    plt.savefig(path)
    plt.close('all')

















