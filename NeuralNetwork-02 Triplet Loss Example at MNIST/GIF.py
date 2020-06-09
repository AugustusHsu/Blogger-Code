# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:31:16 2020

@author: jimhs
"""

import os
import imageio
from args import parser

opts = parser()

def plot2dgif(method):
    images = []
    img_path = os.path.join(opts.PLOT_PATH, method)
    for epoch in range(opts.epochs+1):
        img = imageio.imread(os.path.join(img_path, '{:03d}.png'.format(epoch)))
        images.append(img)
    
    anim_file = method + '_2d.gif'
    path = os.path.join(opts.GIF_PATH, anim_file)
    imageio.mimsave(path, images)

def plot3dgif(method):
    images = []
    img_path = os.path.join(opts.PLOT_3D_PATH, method)
    for view in range(0, 360, 3):
        img = imageio.imread(os.path.join(img_path, '{:03d}.png'.format(view)))
        images.append(img)
    
    anim_file = method + '_3d.gif'
    path = os.path.join(opts.GIF_PATH, anim_file)
    imageio.mimsave(path, images)

