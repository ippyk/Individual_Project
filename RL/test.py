import ray
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
#from train import changes_fc, config_extra, env_config, env

def load_experiment_data(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

class policyRL:
    """
    policy learned during the training
    get the best policy from training {name_xp}
    Use this policy to compute action
    """
    def __init__(self, config_extra, path):
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
        print('best trial path', self.best_trial_path)
        #can replace by string here - any checkpoint of your choice 
        best_checkpoint = analysis.get_best_checkpoint(trial = self.best_trial_path, mode='max',) # metric='episode_reward_mean')
        print('best checkpoint', best_checkpoint)
        
        #load policy from checkpoint
        self.trainer.restore(best_checkpoint)
        
    def get_action(self, state):
        return self.trainer.compute_action(state, explore=False)    
    
    def get_directory(self):
        return '.'+self.best_trial_path[8:]
    
class policyPassive:
    def __init__(self):
        self.name = 'passive'
        
    def get_action(self, state):
        return 1
        
class policyAggressive:
    def __init__(self):
        self.name = 'aggressive'
        
    def get_action(self, state):
        return 0
    
class policyTimid:
    def __init__(self):
        self.name = 'timid'
        
    def get_action(self, state):
        return 2

def test_ray_model(policy, seed, env):

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


    cnt = 0
    while not done:
        action = policy.get_action(state)
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
        test_data['mkt_open'].append(info['mkt_open'])
        test_data['mkt_close'].append(info['mkt_close'])
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

    return test_data

def plot_market_discrete(test_data, save_dir, passive, marker_size = 3):

    colors = {0: 'r', 2: 'b'}  # 0 for Buy (red), 2 for Sell (blue)
    color_map = [colors.get(int(action), None) for action in test_data['action']]
    
    buy_patch = mpatches.Patch(color='r', label='Buy')
    sell_patch = mpatches.Patch(color='b', label='Sell')

    plt.legend(handles=[buy_patch, sell_patch])
    plt.plot(range(len(test_data['last_transaction'])), test_data['last_transaction'], label='Last Transaction Price')

    # Scatter only for actions 0 (Buy) and 2 (Sell)
    for idx, (action, price) in enumerate(zip(test_data['action'], test_data['last_transaction'])):
        if action in colors:
            plt.scatter(idx, price, color=colors[action], s=marker_size)

    plt.xlabel('Time')
    plt.ylabel('Asset Price')
    plt.title('Best Transaction Prices with Action-Based Colors')
    #plt.show()
    if not passive:
        image_path = os.path.join(save_dir, 'market_plot.png')
    else:
        image_path = os.path.join(save_dir, 'market_plot_passive.png')
    plt.savefig(image_path)
    plt.close()

def plot_total_reward(test_data, save_dir):

    plt.plot(test_data['total_reward'])
    plt.title('Total Reward')
    plt.xlabel('Time')
    plt.ylabel('Reward')
    plt.axhline(y=0, color='r', linestyle='--')
    image_path = os.path.join(save_dir, 'total_reward_plot.png')
    plt.savefig(image_path)
    plt.close()

def plot_reward(test_data, save_dir):

    plt.plot(test_data['reward'])
    plt.title('Reward')
    plt.xlabel('Time')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.ylabel('Reward')
    image_path = os.path.join(save_dir, 'reward_plot.png')
    plt.savefig(image_path)
    plt.close()

def plot_marked_to_market(test_data, save_dir):

    plt.plot(test_data['marked_to_market'])
    plt.title('Marked to Market')
    plt.xlabel('Time')
    plt.ylabel('Value')
    image_path = os.path.join(save_dir, 'marked_to_market_plot.png')
    plt.savefig(image_path)
    plt.close()
    
def calculate_volatility(price_series):
    
    price_series = np.asarray(price_series)
    returns = np.diff(price_series) / price_series[:-1]
    std_dev = np.std(returns)

    return std_dev

def calculate_liquidity(ask_prices, bid_prices):

    ask_prices = np.asarray(ask_prices)
    bid_prices = np.asarray(bid_prices)
    spreads = ask_prices - bid_prices
    average_spread = np.mean(spreads)
    
    return average_spread

def calculate_metrics(data):
    
    data = np.asarray(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
    minimum = np.min(data)
    maximum = np.max(data)
    
    return mean, std_dev, minimum, maximum

def market_indicators(test_data):

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

    return indicators

def convert_to_serializable(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Add other conversions if necessary
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def save_dict_to_json(file_path, dictionary):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(dictionary, file, default=convert_to_serializable)

def save_dict_to_pkl(file_path, dictionary):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as file:
        pickle.dump(dictionary, file)
    
def get_analytics(test_data, save_dir, passive, path):

    shutil.copy(os.path.join(path, 'params.json'), os.path.join(save_dir, 'params.json'))
    save_dict_to_pkl(os.path.join(save_dir, 'test_data.pkl'), test_data)
    plot_market_discrete(test_data, save_dir, passive=False)
    plot_total_reward(test_data, save_dir)
    plot_reward(test_data, save_dir)
    plot_marked_to_market(test_data, save_dir)

    indicators = market_indicators(test_data)
    save_dict_to_json(os.path.join(save_dir, 'indicators.json'), indicators)

    if passive:

        test_data_passive = test_ray_model(policyPassive(), seed=seed, env=env)
        plot_market_discrete(test_data_passive, save_dir, passive=True)
        save_dict_to_pkl(os.path.join(save_dir, 'test_data_passive.pkl'), test_data_passive)

        indicators_passive = market_indicators(test_data_passive)
        save_dict_to_json(os.path.join(save_dir, 'indicators_passive.json'), indicators_passive)

        return indicators, indicators_passive

    return indicators

if __name__ == '__main__':

    test_config = load_experiment_data('test.yml')

    if 'single_run_settings' in test_config:
        path = test_config['single_run_settings']['path']
        seed = test_config['single_run_settings']['seed']
        passive = test_config['single_run_settings']['passive']
        experiment_mode = False
    else:
        path = test_config['experiment']['path']
        seed_number = test_config['experiment']['seed_number']
        passive = test_config['experiment']['passive']
        experiment_name = test_config['experiment']['experiment_name']
        experiment_mode = True
        experiment_dict = {}
        experiment_dict['test'] = {}

        if passive:
            experiment_dict['passive'] = {}

    with open(os.path.join(path,'params.json'), 'r') as file:
        experiment_data = json.load(file)

    env_config = experiment_data['env_config']
    config_extra = experiment_data.copy()
    del config_extra['callbacks']
    del config_extra['env']
    del config_extra['env_config']

    if 'env_config' in test_config:
        env_config.update(test_config['env_config'])

    if ('order_fixed_size' in env_config):
        env_config['order_fixed_size'] = int(float(env_config['order_fixed_size']))

    if 'done_ratio' in env_config:
        env_config['done_ratio'] = float(env_config['done_ratio'])

    if ('background_config_extra_kvargs' in env_config) and (env_config['background_config'] == "flash_crash"):
    #if (env_config['background_config'] == "rmsc07") and (env_config['background_config'] == "flash_crash"):

        oracle_config = GBMOracleConfig(mu=1e-9, sigma=0.0135)
        mm_config = MarketMakerAgentConfig(price_skew_param=4, wake_up_freq='1s', subscribe=False, subscribe_freq='1s', subscribe_num_levels=10)
        # mm_config = MarketMakerAgentConfig(price_skew_param=4, wake_up_freq=1e9 * (5) , subscribe=False, subscribe_freq='1s', subscribe_num_levels=10)
        value_agent_config = ValueAgentConfig(kappa_limit=0.3, kappa_mkt=0.1, mean_wakeup_gap=1e8)
        # value_agent_config = ValueAgentConfig(kappa_limit=0.3, kappa_mkt=0.1, mean_wakeup_gap=1e8 * (5))
        momentum_agent_config = MomentumAgentConfig(beta_limit=50, beta_mkt=20, wake_up_freq='1s', subscribe=False)
        # momentum_agent_config = MomentumAgentConfig(beta_limit=50, beta_mkt=20, wake_up_freq=1e9 * (5), subscribe=False)
        exchange_config = ExchangeConfig(log_orders=True)
        institutional_config = InstitutionalTraderAgentConfig(inventory=1e13, sell_frequency="00:00:02", sell_volume_factor=1000)

        changes_fc = {
        "num_noise_agents": 15,
        "num_value_agents": 10,
        "num_mm_agents": 19,
        "num_long_momentum_agents": 5,
        "num_short_momentum_agents": 5,
        "oracle_params": oracle_config,
        "mm_agent_params": mm_config,
        "value_agent_params": value_agent_config,
        "momentum_agent_params": momentum_agent_config,
        "institutional_params": institutional_config
        }

        env_config['background_config_extra_kvargs'] = changes_fc

    env = gym.make("markets-daily_investor-v0", debug_mode=True, **env_config)
    policy = policyRL(config_extra=config_extra, path=path)

    best_trial_path = policy.get_directory()

    if not experiment_mode:

        save_dir = os.path.join('./experiment_history', best_trial_path)
        save_dir += '_seed='+str(seed)
        if 'extra_name' in test_config['single_run_settings']:
                save_dir += '_' + test_config['single_run_settings']['extra_name']

        os.makedirs(save_dir, exist_ok=True)
        policy = policyTimid()
        test_data = test_ray_model(policy, seed=seed, env=env)
        get_analytics(test_data, save_dir, passive, path)

    else:

        seeds = [i for i in range(seed_number)]

        for seed in seeds:

            save_dir = os.path.join('./experiment_history', experiment_name, best_trial_path)
            save_dir += '_seed='+str(seed)
            os.makedirs(save_dir, exist_ok=True)

            test_data = test_ray_model(policy, seed=seed, env=env)

            if passive:
                indicators, indicators_passive = get_analytics(test_data, save_dir, passive, path)
                experiment_dict['passive'][seed] = indicators_passive

            else:
                indicators = get_analytics(test_data, save_dir, passive, path)

            experiment_dict['test'][seed] = indicators

            save_dict_to_json(os.path.join('./experiment_history', experiment_name, 'analysis.json'), experiment_dict)
            


    




