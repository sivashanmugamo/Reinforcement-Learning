'''
Creates a simple grid environment to the provided specifications (grid_x, grid_y values)

Author: @sivashanmugamo
'''

# Importing required libraries
import gym
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

from gym import spaces

class SimpleGrid(gym.Env):
    metadata= {'render.modes': []}

    def __init__(self, agent_position: list, goal_position: list, agent_value: any= 1, goal_value: any= 10, reward_set: dict= {}, grid_x: int= 3, grid_y: int= 3,stochasticity: bool= False, max_timesteps: int= 10) -> None:
        '''
        Initializes the properties of the environment
        Note: Only 4 actions are allowed in this environment - Left, Up, Right, & Down.

        Input:
            agent_position: list - Position of the Agent in the environment
            goal_postion: list - Position of the Goal in the environment
            agent_value: any - Value of the agent in the environment (Only for visualization | Won't add up for reward)
            goal_value: any - Value of the goal position
            reward_set: dict - Dictionary of reward's position (key) and the corresponsing reward (value)
            grid_x: int - Width of the grid environment
            grid_y: int - Height of the grid environment
            stochasticity: bool - If True, the environment is stochastic (uncertain), if false, deterministic (certain)
            max_timesteps: int - Maximum timesteps allowed per episode
        '''

        super().__init__()

        self.grid_x= grid_x
        self.grid_y= grid_y

        self.observation_space= spaces.Discrete(grid_x * grid_y)
        self.action_space= spaces.Discrete(4)

        self.init_agent_pos= agent_position.copy()
        self.agent_value= agent_value

        self.reward_set= reward_set

        self.init_goal_pos= goal_position.copy()
        self.goal_value= goal_value

        self.stochasticity= stochasticity
        self.max_timesteps= max_timesteps

        self.generate_state_lookup()
        self.reset()

    def generate_state_lookup(self) -> None:
        '''
        Generates a lookup table for the state (value) and its corresponding position (key)
        '''

        self.state_lookup= dict()

        state= 0
        for y in range(self.grid_y):
            for x in range(self.grid_x):
                self.state_lookup[(y, x)]= state
                state += 1

    def reset(self) -> int:
        '''
        Resets the environment to its default setup

        Returns:
            int: Agent's default state in the environment
        '''

        self.timestep= 0
        self.state= np.zeros((self.grid_y, self.grid_x))

        self.agent_pos= self.init_agent_pos
        self.state[tuple(self.agent_pos)]= self.agent_value

        if len(self.reward_set) > 0:
            for reward_pos, reward_val in self.reward_set.items():
                self.state[reward_pos]= reward_val

        self.goal_pos= self.init_goal_pos
        self.state[tuple(self.goal_pos)]= self.goal_value

        return self.state_lookup[tuple(self.agent_pos)]

    def save_environment(self, path) -> None:
        '''
        Saves environment parameters to a .pkl file

        Input:
            path: str - Path & filename 
        '''

        obj_to_save= dict()

        obj_to_save['grid_x']= self.grid_x
        obj_to_save['grid_y']= self.grid_y
        obj_to_save['observation_space']= self.observation_space
        obj_to_save['action_space']= self.action_space
        obj_to_save['init_agent_position']= self.init_agent_pos
        obj_to_save['agent_position']= self.agent_pos
        obj_to_save['agent_value']= self.agent_value
        obj_to_save['init_goal_position']= self.init_goal_pos
        obj_to_save['goal_position']= self.goal_pos
        obj_to_save['goal_value']= self.goal_value
        obj_to_save['reward_set']= self.reward_set
        obj_to_save['state']= self.state
        obj_to_save['stochasticity']= self.stochasticity
        obj_to_save['max_timesteps']= self.max_timesteps
        obj_to_save['timestep']= self.timestep

        pickle.dump(
            obj= obj_to_save, 
            file= open(path, 'wb')
        )

    def load_environment(self, path) -> None:
        '''
        Loads environment from a saved .pkl file

        Input:
            path: str - Path of the file
        '''

        with open(path, 'rb') as f:
            obj_to_load= pickle.load(f)
        
        self.grid_x= obj_to_load['grid_x']
        self.grid_y= obj_to_load['grid_y']
        self.observation_space= obj_to_load['observation_space']
        self.action_space= obj_to_load['action_space']
        self.init_agent_pos= obj_to_load['init_agent_position']
        self.agent_pos= obj_to_load['agent_position']
        self.agent_value= obj_to_load['agent_value']
        self.init_goal_pos= obj_to_load['init_goal_position']
        self.goal_pos= obj_to_load['goal_position']
        self.goal_value= obj_to_load['goal_value']
        self.reward_set= obj_to_load['reward_set']
        self.state= obj_to_load['state']
        self.stochasticity= obj_to_load['stochasticity']
        self.max_timesteps= obj_to_load['max_timesteps']
        self.timestep= obj_to_load['timestep']

        self.generate_state_lookup()

    def step(self, action) -> tuple:
        '''
        Performs the following,
            1. Moves the agent as per the chosen (given/random) action
            2. Sets the rewards per state in the environment
            3. Calcualtes the rewards as per action of the agent
        
        Input:
            action: int - Action to be performed from the action set

        Returns:
            state: int - Current state of the agent in the environment after action is performed (observation)
            reward: int - State reward
            done: bool - Denotes if the timestep is complete (or) goal is reached
            info: dict
        '''

        # Initializes the states
        self.state= np.zeros((self.grid_y, self.grid_x))
        temp_pos= self.agent_pos.copy()

        # Sets the uncertainity factor of the environment
        if self.stochasticity:
            action= action if random.uniform(0, 1) < 0.75 else random.choice([i for i in range(self.action_space.n)])

        # Moves the agent in the environment by 1 step
        if action == 0: # Down
            self.agent_pos[0] += 1
        if action == 1: # Up
            self.agent_pos[0] -= 1
        if action == 2: # Right
            self.agent_pos[1] += 1
        if action == 3: # Left
            self.agent_pos[1] -= 1

        # Keeps the agent within the confines of the environment
        if (self.agent_pos[0] < 0) or (self.agent_pos[1] > self.grid_y):
            self.agent_pos[0]= temp_pos[0]
        if (self.agent_pos[1] < 0) or (self.agent_pos[1] > self.grid_x):
            self.agent_pos[1]= temp_pos[1]

        self.state[tuple(self.agent_pos)]= self.agent_value

        # Sets the reward for the states
        if len(self.reward_set) > 0:
            for reward_pos, reward_val in self.reward_set.items():
                self.state[reward_pos]= reward_val
        
        self.goal_pos= self.init_goal_pos
        self.state[tuple(self.goal_pos)]= self.goal_value

        # Calculates the reward for the state & its corresponding action
        reward= 0
        if len(self.reward_set) > 0:
            for reward_pos, reward_val in self.reward_set.items():
                if (tuple(self.agent_pos) == reward_pos):
                    reward += reward_val

        if (self.agent_pos == self.goal_pos):
            reward+= self.goal_value

        # Time step increment
        self.timestep += 1

        done= True if ((self.timestep >= self.max_timesteps) or (self.agent_pos == self.goal_pos)) else False
        info= {}

        return (self.state_lookup[tuple(self.agent_pos)], reward, done, info)

    def render(self) -> None:
        '''
        Renders the current state of the environment
        '''
        
        plt.imsave('./environment_render.png', self.state)

    @property
    # Getter method for init_agent_pos
    def init_agent_pos(self):
        return self._init_agent_pos

    @init_agent_pos.setter
    # Setter method for init_agent_pos
    def init_agent_pos(self, value):
        # Restricts the value & type of init_agent_pos
        if isinstance(value, list) == False:
            raise TypeError('List expected for Agent Position, but received {}.'.format(type(value)))
        if (value[0] > self.grid_y-1) or (value[1] > self.grid_x-1):
            raise ValueError('Agent position should be between (0, 0) and ({}, {}).'.format(self.grid_x, self.grid_y))
        self._init_agent_pos= value

    @property
    # Getter method for init_goal_pos
    def init_goal_pos(self):
        return self._init_goal_pos

    @init_goal_pos.setter
    # Setter method for init_goal_pos
    def init_goal_pos(self, value):
        # Restricts the value & type of init_goal_pos
        if isinstance(value, list) == False:
            raise TypeError('List expected for Goal Position, but received {}.'.format(type(value)))
        if (value[0] > self.grid_y-1) or (value[1] > self.grid_x-1):
            raise ValueError('Goal position should be between (0, 0) and ({}, {}).'.format(self.grid_x, self.grid_y))
        self._init_goal_pos= value

    @property
    # Getter method for reward_set
    def reward_set(self):
        return self._reward_set

    @reward_set.setter
    # Setter method for reward_set
    def reward_set(self, value):
        # Restricts the value & type of reward_set
        if isinstance(value, dict) == False:
            raise TypeError('Dictionary expected for Reward Set, but received {}.'.format(type(value)))
        for key, val in value.items():
            if (key[0] > self.grid_y-1) or (key[1] > self.grid_x-1):
                raise ValueError('Reward position should be between (0, 0) and ({}, {}).'.format(self.grid_x, self.grid_y))
            if isinstance(val, (int, float, complex)) == False:
                raise ValueError('Reward should be numerical, but received {}.'.format(type(val)))
        self._reward_set= value

    @property
    # Getter method for max_timesteps
    def max_timesteps(self):
        return self._max_timesteps

    @max_timesteps.setter
    # Setter method for max_timesteps
    def max_timesteps(self, value):
        # Restricts the value & type of max_timesteps
        if isinstance(value, int) == False:
            raise TypeError('Int expected for maximum timesteps, but received {}.'.format(type(value)))
        if (value < 0):
            raise ValueError('Maximum timestep should be positive')