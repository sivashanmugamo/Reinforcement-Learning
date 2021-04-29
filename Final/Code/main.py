'''
Author: @sivashanmugamo
'''

# Importing required libraries
import random
from datetime import datetime
from collections import deque, namedtuple

import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', 50)

import tensorflow as tf

import torch
from torch import mode, nn, optim, autograd
from torch.nn import functional as F

# Checking GPU availability
DEVICE= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Defining path of the dataset
PATH= '/Users/shiva/Documents/GitHub/Reinforcement-Learning/Final/Code/data/S&P 500 - Jan 05 to Mar 21.csv'

# Class to prepare data & act as environment
class Data:
    def __init__(self, path: str, header: any):
        self.file_path= path
        self.read(path= self.file_path, header= header)

    def read(self, path: str, header: any):
        '''
        Reads raw CSV data from given path

        Args:
            path: str - Path to the data (.csv file)
            header: None/int - To mention the presence of header
        '''

        self.data= pd.read_csv(path, header= header)
        self.data= self.prepare(self.data)

    def prepare(self, data) -> pd.DataFrame:
        '''
        Prepares data to a NN suitable format

        Arg:
            data: pd.DataFrame - Dataframe of raw data
        '''

        # Renaming columns
        data.rename(columns= {
            0: 'date', 
            1: 'open_price', 
            2: 'high_price', 
            3: 'low_price', 
            4: 'close_price', 
            5: 'adj_close_price', 
            6: 'share_volume'
        }, inplace= True)

        # Converting dates (in str format) to datetime format
        data['date']= pd.to_datetime(data['date'], format= '%d-%b-%y')

        # Seperating day, month, & year from date 
        # (easier for computation in network)
        data['day']= data['date'].dt.day
        data['month']= data['date'].dt.month
        data['year']= data['date'].dt.year

        # Delete date column
        del data['date']

        # Prepare all values in column for int/float formatting
        data= data.replace(',', '', regex= True)

        # Converts respective columns to float or int as required
        data['open_price']= pd.to_numeric(data['open_price'])
        data['high_price']= pd.to_numeric(data['high_price'])
        data['low_price']= pd.to_numeric(data['low_price'])
        data['close_price']= pd.to_numeric(data['close_price'])
        data['adj_close_price']= pd.to_numeric(data['adj_close_price'])
        data['share_volume']= pd.to_numeric(data['share_volume'])

        return data

    def get_state(self):
        pass

    # ------------------------------------------
    # Getter & setter methods
    # ------------------------------------------

    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, value):
        if isinstance(value, str) == False:
            raise TypeError('Invalid path')
        self._file_path= value

# Format to store experience
EXPERIENCE= namedtuple('experience', ['state', 'action', 'reward', 'next_state', 'done'])

# Replay buffer for the agent
class Memory:
    def __init__(self, volume: int= 1e4) -> None:
        self.volume= volume
        self.memory= deque(
            maxlen= self.volume
            )

    def memorize(self, experience: tuple) -> None:
        '''
        Stores experience to memory

        Arg:
            experience: tuple - Tuple of state, action, reward, next state, & done
        '''
        self.memory.append(EXPERIENCE(experience))

    def sample(self, batch_size: int= 64) -> list:
        '''
        Samples a batch of experience from memory
        '''
        return random.sample(self.memory, k= batch_size)

    def __len__(self) -> int:
        '''
        Returns the filled memory size
        '''
        return len(self.memory)

    # ------------------------------------------
    # Getter & setter methods
    # ------------------------------------------

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, value):
        if isinstance(value, int) == False:
            raise TypeError('int expected, instead received {}'.format(type(value)))
        if value < 0:
            raise ValueError('Memory capacity should be positive, instead received {}'.format(value))
        if value > int(1e8):
            raise ValueError('Maximum memory size is {}'.format(int(1e8)))
        self._volume= value

class ACNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, n_hidden: int= 32, n_layers: int= 1, conv_mode: bool= False) -> None:
        super(ACNetwork, self).__init__()

        #  Observation space size
        self.input_dim= state_size
        self.n_hidden= n_hidden
        self.n_layers= n_layers

        # Action space size
        self.output_dim= action_size

        # Create hidden layers
        # self.hidden_layers= list()
        # for _ in range(self.n_hidden):
        #     self.hidden_layers.append(nn.Linear(self.n_hidden, self.n_hidden))
        #     self.hidden_layers(nn.ReLU())

        # Actor model
        # Predicts policy distribution
        self.actor= nn.Sequential(
            nn.Linear(self.input_dim, self.n_hidden), 
            nn.BatchNorm1d(), 
            nn.ReLU(), 
            nn.Linear(self.n_hidden, self.output_dim),
            nn.Softmax(dim= 1)
            )

        # Critic model
        # Predicts value
        self.critic= nn.Sequential(
            nn.Linear(self.input_dim, self.n_hidden), 
            nn.BatchNorm1d(), 
            nn.ReLU(), 
            nn.Linear(self.n_hidden, 1)
            )

    def forward(self, state):
        '''
        '''
        pass

class Agent:
    def __init__(self, env, mode: str= 'ac', volume: int= 1e4) -> None:
        '''
        '''

        self.env= env
        self.mode= mode

        # Initializing agent's memory
        self.volume= volume
        self.memory= Memory(
            volume= self.volume
            )

    def train(self, n_episodes: int= 1000):
        '''
        '''

        pass

def run(mode: str):
    if mode == 'train':
        pass
    elif mode == 'test':
        pass
    else:
        raise Exception('Invalid mode. Mode should either be \'train\' or \'test\'')

if __name__ == '__main__':
    run(
        mode= 'train'
    )