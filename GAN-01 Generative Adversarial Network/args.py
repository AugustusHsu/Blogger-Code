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
        '''
        -----------------------Setting-----------------------
        '''
        self.epochs = 120
        self.noise_dim = 100
        self.BUFFER_SIZE = 60000
        self.BATCH_SIZE = 256
        self.lr = 2e-3
        self.beta_1 = 0.5
        self.num_examples_to_generate = 16
        
