import logging
import math
import copy
from pprint import pprint
from typing import Tuple, Dict, List, Text, Callable
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from news.envs.news import News
from news.utils.config import Config

import time, random
import networkx as nx
import platform


class InfeasibleActionError(ValueError):
    """An infeasible action were passed to the env."""

    def __init__(self, action, mask):
        """Initialize an infeasible action error."""
        super().__init__(self, action, mask)
        self.action = action
        self.mask = mask

    def __str__(self):
        return 'Infeasible action ({}) when the mask is ({})'.format(
            self.action, self.mask)


def reward_info_function(news: News, stage) -> Tuple[float, Dict]:
    final_i_rate = news.get_final_i_rate()
    total_i_rate = news.get_total_i_rate()
    full_final_i_rate = news.get_full_final_i_rate()
    full_total_i_rate = news.get_full_total_i_rate()
    reward = news.get_reward() * 10

    return reward, {'reward': reward, 'fir': final_i_rate, 'tir': total_i_rate,
                    'ffir': full_final_i_rate, 'ftir': full_total_i_rate}


class NewsEnv:

    FAILURE_REWARD = -4.0
    INTERMEDIATE_REWARD = -4.0

    def __init__(self,
                 cfg: Config,
                 is_eval: bool = False,
                 reward_info_fn=reward_info_function):

        self.cfg = cfg
        self._is_eval = is_eval
        self._frozen = False
        self._action_history = []
        self._news = None  # Initialize as None
        self._copy_news = None
        self._reward_info_fn = partial(reward_info_fn)
        self._done = False
        self._tf_objects = None  # Storage for TF objects
        
        # Initialize news object lazily
        self.initialize_news()

    def __getstate__(self):
        """Custom state retrieval for pickling."""
        state = self.__dict__.copy()
        # Remove unpicklable objects
        if '_tf_objects' in state:
            del state['_tf_objects']
        # Handle news objects specially
        state['_news'] = None
        state['_copy_news'] = None
        return state

    def __setstate__(self, state):
        """Custom state restoration for unpickling."""
        self.__dict__.update(state)
        self._tf_objects = None
        # Reinitialize news objects
        self.initialize_news()

    def initialize_news(self):
        """Initialize news objects lazily."""
        if self._news is None:
            self._news = self.load_graph(self.cfg)
            self._copy_news = copy.deepcopy(self._news)

    def initialize_tf(self):
        """Initialize TensorFlow objects lazily."""
        if self._tf_objects is None:
            # Initialize any TensorFlow objects here if needed
            pass

    def load_graph(self, cfg):
        data_source = cfg.data_source
        spread_param = cfg.env_param
        spread_param['seed'] = cfg.seed
        
        n = News(data_source, spread_param)
        return n

    def _set_cached_reward_info(self):
        """Set the cached reward."""
        if not self._frozen:
            self._cached_life_circle_reward = -1.0
            self._cached_greeness_reward = -1.0
            self._cached_concept_reward = -1.0
            self._cached_life_circle_info = dict()
            self._cached_concept_info = dict()
            self._cached_land_use_reward = -1.0
            self._cached_land_use_gdf = self.snapshot_land_use()

    def get_reward_info(self) -> Tuple[float, Dict]:
        return self._reward_info_fn(self._news, self._stage)

    def eval(self):
        self._is_eval = True

    def train(self):
        self._is_eval = False

    def get_info(self):
        return self._news.get_env_info_dict()

    def get_numerical_feature_size(self):
        return self._news.get_numerical_dim()

    def get_node_dim(self):
        return self._news.get_node_dim()
    
    def get_edge_dim(self):
        return self._news.get_edge_dim()
    
    def get_max_node_num(self):
        return self._news.get_max_node_num()
    
    def get_max_edge_num(self):
        return self._news.get_max_edge_num()
    
    def get_stage(self):
        if self._stage == 'build':
            return [1,0]
        elif self._stage == 'done':
            return [0,1]

    def _get_obs(self) -> List:
        numerical, node_feature, edge_feature, edge_index, node_mask = self._news.get_obs()
        stage = self.get_stage()
        return [numerical, node_feature, edge_feature, edge_index, node_mask, stage]

    def action(self, action):
        self._news.cut_edge_from_action(int(action))

    def get_action_num(self):
        return self._news.get_cut_num()
    
    def get_total_action(self):
        return self._news.get_total_cut_num()

    def snapshot_land_use(self):
        return self._news.snapshot()
       
    def save_step_data(self):
        return

    def failure_step(self, logging_str, logger):
        """Logging and reset after a failure step."""
        logger.info('{}: {}'.format(logging_str, self._action_history))
        info = {
            'road_network': -1.0,
            'life_circle': -1.0,
            'greeness': -1.0,
        }
        return self._get_obs(), self.FAILURE_REWARD, True, info

    def step(self, action, logger: logging.Logger) -> Tuple[List, float, bool, Dict]:
        if self._done:
            raise RuntimeError('Action taken after episode is done.')

        if self._stage == 'build':
            if self.get_action_num() >= self.get_total_action():
                self.transition_stage()
            else:
                self.action(action)
                self._action_history.append(int(action))

        if self._news.get_done():
            self.transition_stage()

        reward, info = self.get_reward_info()
        if self._stage == 'done':
            self.save_step_data()

        return self._get_obs(), reward, self._done, info

    def reset(self, eval=False, agent_dict=None):
        self._news.reset(eval, agent_dict=agent_dict)
        self._action_history = []
        self._set_stage()
        self._done = False
        return self._get_obs()
    
    def get_env_info_dict(self):
        return self._news.get_env_info_dict()

    def _set_stage(self):
        self._stage = 'build'

    def transition_stage(self):
        if self._stage == 'build':
            self._stage = 'done'
            self._done = True
        else:
            raise RuntimeError('Error stage!')
        
    def plot_and_save(self, save_fig: bool = False, path: Text = None, show=False) -> None:
        """Plot and save the graph visualization."""
        self._news.plot()
        if save_fig:
            assert path is not None
            plt.savefig(path + '.svg', format='svg', transparent=True)
            data = plt.gca().get_lines()
            y_data = []
            for d in data:
                x, y = d.get_data()
                y_data.append(y)

            with open(path + '.txt', 'w') as f:
                for i in range(len(y_data)):
                    f.write('[')
                    for y_idx in range(len(y_data[i])):
                        if y_idx < len(y_data[i]) - 1:
                            f.write(str(y_data[i][y_idx]) + ',')
                        else:
                            f.write(str(y_data[i][y_idx]))
                    f.write(']\n')

        if show:
            plt.show()

        plt.cla()
        plt.close('all')

    def visualize(self, save_fig: bool = False, path: Text = None, show=False, final=None) -> None:
        """Visualize the network."""
        self.plot_and_save(save_fig, path, show)