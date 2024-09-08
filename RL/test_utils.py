import gym
import abides_gym
import numpy as np 
#np.random.seed(0)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from abides_core import abides
from abides_core.utils import parse_logs_df, ns_date, str_to_ns, fmt_ts
from abides_markets.orders import Side
from abides_markets.configs.flash_crash import build_config
from abides_markets.configs.agent_params import ExchangeConfig, NoiseAgentConfig, ValueAgentConfig, MarketMakerAgentConfig, MomentumAgentConfig, GBMOracleConfig, InstitutionalTraderAgentConfig
from abides_markets.oracles.mean_reverting_oracle import MeanRevertingOracle
import os
import json
import yaml
import shutil
import pickle
from ray.tune import Analysis
import ray.rllib.agents.dqn as dqn
import ray.rllib.agents.sac as sac
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.pg as pg
import ray.rllib.agents.ddpg as ddpg

def load_experiment_data(file_path):
    # Loads test yaml file
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

class policyRL:
    """
    policy learned during the training
    get the best policy from training {name_xp}
    Use this policy to compute action
    """
    def __init__(self, config_extra, env_config, path):
        self.name = 'rl'
        ##### LOADING POLICY 
        # cell specific to ray to 
        #this part is to get a path
        # https://github.com/ray-project/ray/issues/4569#issuecomment-480453957

        if 'DQN' in path:
            config = dqn.DEFAULT_CONFIG.copy()
            config.update(config_extra)
            self.trainer = dqn.DQNTrainer(config=config, env="markets-daily_investor-v0")
        elif 'SAC' in path:
            config = sac.DEFAULT_CONFIG.copy()
            config.update(config_extra)
            self.trainer = sac.SACTrainer(config=config, env="markets-daily_investor-v0")
        elif 'PPO' in path:
            config = ppo.DEFAULT_CONFIG.copy()
            config.update(config_extra)
            self.trainer = ppo.PPOTrainer(config=config, env="markets-daily_investor-v0")
        elif 'PG' in path:
            config = pg.DEFAULT_CONFIG.copy()
            config.update(config_extra)
            config['env_config'].update(env_config)
            self.trainer = pg.PGTrainer(config=config, env="markets-daily_investor-v0")
        elif 'DDPG' in path:
            config = ddpg.DEFAULT_CONFIG.copy()
            config.update(config_extra)
            self.trainer = ddpg.DDPGTrainer(config=config, env="markets-daily_investor-v0")
        else:
            raise ValueError(f"Unsupported agent type")
        
        data_folder = path
        analysis = Analysis(data_folder)
        trial_dataframes = analysis.trial_dataframes
        trials = list(trial_dataframes.keys())
        #best_trial_path = analysis.get_best_logdir(metric='episode_reward_mean', mode='max')
        self.best_trial_path = path
        #can replace by string here - any checkpoint of your choice 
        best_checkpoint = analysis.get_best_checkpoint(trial = self.best_trial_path, mode='max',) # metric='episode_reward_mean')
        
        #load policy from checkpoint
        self.trainer.restore(best_checkpoint)
        
    def get_action(self, state):
        return self.trainer.compute_action(state, explore=False)    
    
    def get_directory(self):
        return '.'+self.best_trial_path[8:]
    
class policyPassive:
    # Policy that always holds
    def __init__(self, continuous):
        self.name = 'passive'
        self.continuous = continuous
        
    def get_action(self, state):
        
        if not self.continuous:
            return 1
        else:
            return np.array([0])
        
class policyAggressive:
    # Policy that always buys
    def __init__(self):
        self.name = 'aggressive'
        
    def get_action(self, state):
        return 0
    
class policyTimid:
    # Policy that alway sells
    def __init__(self):
        self.name = 'timid'
        
    def get_action(self, state):
        return 2

class policyRandom:
    # Policy that randomly buys or sells
    def __init__(self):
        self.name = 'random'
        
    def get_action(self, state):
        return np.random.choice([0,2])
    
def test_ray_model(policy, seed, env):
    # Function to test policy in gym environment

    env.seed(seed)
    state = env.reset()
    done = False
    total_reward = 0

    test_data = {
    'last_transaction': [], 'best_bid': [], 'best_ask': [], 'spread': [], 'bids': [], 'asks': [], 
    'cash': [], 'current_time': [], 'holdings': [], 'orderbook': [], 'order_status': [], 'mkt_open': [], 
    'mkt_close': [], 'last_bid': [], 'last_ask': [], 'wide_spread': [], 'ask_spread': [], 
    'bid_spread': [], 'marked_to_market': [], 'total_volume': [], 'action': [], 'reward': [], 
    'total_reward': [], 'bid_volume': [], 'ask_volume': []
    }

    buy_count = 0
    hold_count = 0
    sell_count = 0

    cnt = 0
    while not done:
        action = policy.get_action(state)
        print(action)
        state, reward, done, info = env.step(action)
        total_reward += reward
        if not cnt % 100:
            print('iteration:', cnt)
        cnt += 1

        test_data['last_transaction'].append(info['last_transaction'])
        test_data['best_bid'].append(info['best_bid'])
        test_data['best_ask'].append(info['best_ask'])
        test_data['spread'].append(info['spread'])
        test_data['bids'].append(info['bids'])
        test_data['asks'].append(info['asks'])
        test_data['cash'].append(info['cash'])
        test_data['current_time'].append(info['current_time'])
        test_data['holdings'].append(info['holdings'])
        test_data['orderbook'].append(info['orderbook'])
        test_data['order_status'].append(info['order_status'])
        test_data['mkt_open'] = info['mkt_open']
        test_data['mkt_close'] = info['mkt_close']
        test_data['last_bid'].append(info['last_bid'])
        test_data['last_ask'].append(info['last_ask'])
        test_data['wide_spread'].append(info['wide_spread'])
        test_data['ask_spread'].append(info['ask_spread'])
        test_data['bid_spread'].append(info['bid_spread'])
        test_data['marked_to_market'].append(info['marked_to_market'])
        test_data['total_volume'].append(info['total_volume'])
        test_data['bid_volume'].append(info['bid_volume'])
        test_data['ask_volume'].append(info['ask_volume'])
        test_data['action'].append(action)
        test_data['reward'].append(reward)
        test_data['total_reward'].append(total_reward)
        test_data['stop_hold_values'] = info['stop_hold_values']
        test_data['stop_short_values'] = info['stop_short_values']
        test_data['first_interval'] = info['first_interval']
        test_data['stabalise2_values'] = info['stabalise2_values']

    return test_data

def plot_market(test_data, save_dir, passive, continuous=False, marker_size=10):
    # Plots market data with actions

    start = test_data['mkt_open']+test_data['first_interval']
    end = test_data['mkt_close']

    num_ticks = len(test_data['last_transaction'])
    time_mesh = np.linspace(start, end, num_ticks)

    fig, ax = plt.subplots(figsize=(9,8))

    if not continuous:
        colors = {0: 'r', 2: 'b'}  # 0 for Buy (red), 2 for Sell (blue)
        
        # Creating the color map for scatter plot
        color_map = [colors.get(int(action), None) for action in test_data['action']]

        buy_patch = mpatches.Patch(color='r', label='Buy')
        sell_patch = mpatches.Patch(color='b', label='Sell')

        ax.plot(time_mesh, test_data['last_transaction'], label='Last Transaction Price')

        updated_values = test_data['action']
        if 'stabalise2_values' in test_data and test_data['stabalise2_values'] is not None:
            print(test_data['stabalise2_values'][::-1])
            for i in range(len(test_data['stabalise2_values'])):
                if test_data['stabalise2_values'][i] == 0 or test_data['stabalise2_values'][i] == 2:
                    updated_values[i] = test_data['stabalise2_values'][i]

        # Scatter only for actions 0 (Buy) and 2 (Sell)
        for idx, (time, action, price) in enumerate(zip(time_mesh, updated_values, test_data['last_transaction'])):
            if action in colors:
                ax.scatter(time, price, color=colors[action], s=marker_size)

        ax.legend(handles=[buy_patch, sell_patch], loc='best')

    else:
        def get_color(action):
            action_value = action[0]

            if action_value < 0:
                return (0, 0, 1)  # Blue
            elif action_value > 0:
                return (1, 0, 0)  # Red
            else:
                return (0, 0, 0, 0)  # Transparent for neutral action

        buy_patch = mpatches.Patch(color='red', label='Buy')
        sell_patch = mpatches.Patch(color='blue', label='Sell')

        ax.plot(time_mesh, test_data['last_transaction'], label='Last Transaction Price')

        # Scatter plot with colors based on action values
        for idx, (time, action, price) in enumerate(zip(time_mesh, test_data['action'], test_data['last_transaction'])):
            ax.scatter(time, price, color=get_color(action), s=marker_size)

        ax.legend(handles=[buy_patch, sell_patch], loc='best')

    if 'stop_hold_values' in test_data and test_data['stop_hold_values'] is not None:
        for idx, (time,stop_value) in enumerate(zip(time_mesh,test_data['stop_hold_values'])):
            if stop_value == 1:
                ax.scatter(time, test_data['last_transaction'][idx], color='g', marker='x', s=150)

    if 'stop_short_values' in test_data and test_data['stop_short_values'] is not None:
        for idx, (time,stop_value) in enumerate(zip(time_mesh,test_data['stop_short_values'])):
            if stop_value == 1:
                ax.scatter(time, test_data['last_transaction'][idx], color='black', marker='x', s=150)

    if not passive:
        image_path = os.path.join(save_dir, 'market_plot.png')
    else:
        image_path = os.path.join(save_dir, 'market_plot_passive.png')

    ax.set_ylabel("Price")
    ax.set_xlabel("Time")
    ax.set_title('Transaction Prices')

    ax.set_xlim(start, end)

    num_ticks2 = min(10,len(test_data['last_transaction']))
    time_mesh2 = np.linspace(start, end, num_ticks2)
    ax.set_xticks(time_mesh2)
    ax.set_xticklabels([fmt_ts(time).split(" ")[1] for time in time_mesh2], rotation=60)

    plt.savefig(image_path)
    plt.show()

    plt.close()

def plot_total_reward(test_data, save_dir):
    # Plots total reward throughout simulation

    start = test_data['mkt_open'] + test_data['first_interval']
    end = test_data['mkt_close']

    num_ticks = len(test_data['total_reward'])
    time_mesh = np.linspace(start, end, num_ticks)

    fig, ax = plt.subplots(figsize=(9, 8))

    ax.plot(time_mesh, test_data['total_reward'])
    ax.set_title('Total Reward')
    ax.set_xlabel('Time')
    ax.set_ylabel('Reward')
    ax.axhline(y=0, color='r', linestyle='--')

    ax.set_xlim(start, end)

    num_ticks2 = min(10, len(test_data['total_reward']))
    time_mesh2 = np.linspace(start, end, num_ticks2)
    ax.set_xticks(time_mesh2)
    ax.set_xticklabels([fmt_ts(time).split(" ")[1] for time in time_mesh2], rotation=60)

    image_path = os.path.join(save_dir, 'total_reward_plot.png')
    plt.savefig(image_path)
    plt.show()

    plt.close()

def plot_reward(test_data, save_dir):
    # Plots reward throughout simulation

    start = test_data['mkt_open'] + test_data['first_interval']
    end = test_data['mkt_close']

    num_ticks = len(test_data['reward'])
    time_mesh = np.linspace(start, end, num_ticks)

    fig, ax = plt.subplots(figsize=(9, 8))

    ax.scatter(time_mesh, test_data['reward'])
    ax.set_title('Reward')
    ax.set_xlabel('Time')
    ax.set_ylabel('Reward')
    ax.axhline(y=0, color='r', linestyle='--')

    ax.set_xlim(start, end)

    num_ticks2 = min(10, len(test_data['reward']))
    time_mesh2 = np.linspace(start, end, num_ticks2)
    ax.set_xticks(time_mesh2)
    ax.set_xticklabels([fmt_ts(time).split(" ")[1] for time in time_mesh2], rotation=60)

    image_path = os.path.join(save_dir, 'reward_plot.png')
    plt.savefig(image_path)
    plt.show()

    plt.close()

def plot_marked_to_market(test_data, save_dir):
    # Plots value of portfolio throughout simulation

    start = test_data['mkt_open'] + test_data['first_interval']
    end = test_data['mkt_close']

    num_ticks = len(test_data['marked_to_market'])
    time_mesh = np.linspace(start, end, num_ticks)

    fig, ax = plt.subplots(figsize=(9, 8))

    ax.plot(time_mesh, test_data['marked_to_market'])
    ax.set_title('Marked to Market')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')

    ax.set_xlim(start, end)

    num_ticks2 = min(10, len(test_data['marked_to_market']))
    time_mesh2 = np.linspace(start, end, num_ticks2)
    ax.set_xticks(time_mesh2)
    ax.set_xticklabels([fmt_ts(time).split(" ")[1] for time in time_mesh2], rotation=60)

    image_path = os.path.join(save_dir, 'marked_to_market_plot.png')
    plt.savefig(image_path)
    plt.show()

    plt.close()
    
def calculate_volatility(price_series):
    # Calculates volatility
    
    price_series = np.asarray(price_series)
    returns = np.diff(price_series) / price_series[:-1]
    std_dev = np.std(returns)

    return std_dev

def calculate_liquidity(ask_prices, bid_prices):
    # Calculates liquidity

    ask_prices = np.asarray(ask_prices)
    bid_prices = np.asarray(bid_prices)
    spreads = ask_prices - bid_prices
    average_spread = np.mean(spreads)
    
    return average_spread

def calculate_metrics(data):
    # Calculates asset price indicators
    
    data = np.asarray(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
    minimum = np.min(data)
    maximum = np.max(data)
    
    return mean, std_dev, minimum, maximum

def market_indicators(test_data):
    # Calculates market indicators

    indicators = {}
    
    volatility = calculate_volatility(test_data['last_transaction'])
    avg_spread = np.mean(test_data['spread'])
    mean_price, std_price, min_price, max_price = calculate_metrics(test_data['last_transaction'])
    volume = np.sum(test_data['total_volume'])

    indicators['volatility'] = volatility
    indicators['avg_spread'] = avg_spread
    indicators['asset_price'] = {'mean_price': mean_price, 'std_price': std_price, 
                                'min_price': min_price, 'max_price': max_price }
    indicators['volume'] = volume
    indicators['total_reward'] = test_data['total_reward'][-1]
    indicators['marked_to_market'] = test_data['marked_to_market']
    indicators['bid_volume'] = test_data['bid_volume']
    indicators['ask_volume'] = test_data['ask_volume']
    indicators['last_transaction'] = test_data['last_transaction']
    indicators['spread'] = test_data['spread']

    return indicators

def convert_to_serializable(obj):
    # Allows forwriting of different objets into json

    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Add other conversions if necessary
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def save_dict_to_json(file_path, dictionary):
    # Saves dictionary to json
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(dictionary, file, default=convert_to_serializable)

def save_dict_to_pkl(file_path, dictionary):
    # Saves dictionary to pkl
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as file:
        pickle.dump(dictionary, file)
    
def get_analytics(test_data, save_dir, passive, continuous, path, seed, env):
    # Runs all plotting and test data analysis

    shutil.copy(os.path.join(path, 'params.json'), os.path.join(save_dir, 'params.json'))
    save_dict_to_pkl(os.path.join(save_dir, 'test_data.pkl'), test_data)
    plot_market(test_data, save_dir, passive=False, continuous=continuous)
    plot_total_reward(test_data, save_dir)
    plot_reward(test_data, save_dir)
    plot_marked_to_market(test_data, save_dir)

    indicators = market_indicators(test_data)
    save_dict_to_json(os.path.join(save_dir, 'indicators.json'), indicators)

    if passive:

        test_data_passive = test_ray_model(policyPassive(continuous=continuous), seed=seed, env=env)
        plot_market(test_data_passive, save_dir, passive=True, continuous=continuous)
        save_dict_to_pkl(os.path.join(save_dir, 'test_data_passive.pkl'), test_data_passive)
        
        indicators_passive = market_indicators(test_data_passive)
        save_dict_to_json(os.path.join(save_dir, 'indicators_passive.json'), indicators_passive)

        return indicators, indicators_passive

    return indicators