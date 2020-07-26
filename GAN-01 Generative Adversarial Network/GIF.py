# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:31:16 2020

@author: jimhs
"""

import os
import imageio
from args import parser

opts = parser()

images = []
img_path = os.path.join(opts.PLOT_PATH)
for epoch in range(opts.epochs):
    img = imageio.imread(os.path.join(img_path, '{:03d}.png'.format(epoch+1)))
    images.append(img)

anim_file = 'gan.gif'
imageio.mimsave(anim_file, images)

