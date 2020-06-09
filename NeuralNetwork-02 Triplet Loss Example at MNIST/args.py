# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:20:14 2020

@author: jimhs
"""

class parser(object):
    def __init__(self):
        # Random Seed
        self.seed = 1234

        # PATH
        self.LOG_PATH = 'logs/'
        self.MODEL_PATH = 'models/'
        self.PLOT_PATH = 'output/'
        self.PLOT_3D_PATH = '3d/'
        self.PLOT_LINE_PATH = 'line/'
        self.GIF_PATH = 'gif/'
        '''
        -----------------------Setting-----------------------
        '''
        self.epochs = 180
        self.BUFFER_SIZE = 60000
        self.BATCH_SIZE = 512
        self.margin = 0.4
        self.lr = 0.0000034
        self.beta = 0.5
        
