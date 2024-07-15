import ray
from ray import tune
import gym
import abides_gym
from ray.tune.registry import register_env
from abides_gym.envs.markets_daily_investor_environment_v0 import SubGymMarketsDailyInvestorEnv_v0
from ray.rllib.agents.callbacks import DefaultCallbacks
import numpy as np 
#np.random.seed(0)
import pandas as pd
from matplotlib import pyplot as plt
import sys
from abides_core import abides
from abides_core.utils import parse_logs_df, ns_date, str_to_ns, fmt_ts
from abides_markets.orders import Side
from abides_markets.configs.flash_crash import build_config
from abides_markets.configs.agent_params import ExchangeConfig, NoiseAgentConfig, ValueAgentConfig, MarketMakerAgentConfig, MomentumAgentConfig, GBMOracleConfig, InstitutionalTraderAgentConfig
from abides_markets.oracles.mean_reverting_oracle import MeanRevertingOracle
import os
import argparse
import yaml

def load_experiment_data(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

class MyCustomCallbacks(DefaultCallbacks):
    def on_train_result(self, *, trainer, result: dict, **kwargs):
        print(f"Iteration: {result['training_iteration']}, "
              f"Episode Reward Mean: {result['episode_reward_mean']:.2f}, "
              f"Total Timesteps: {result['timesteps_total']}")
        
register_env(
        "markets-daily_investor-v0",
        lambda config: SubGymMarketsDailyInvestorEnv_v0(**config),
    )

def train_ray_model(agent_type, iterations, verbose, config_extra, env_config):

    ray.shutdown()
    ray.init()

    local_dir = "./models"
    name_xp = "ray_trading"+agent_type #change to your convenience

    basics = {"env": "markets-daily_investor-v0", "env_config": env_config, "callbacks": MyCustomCallbacks}
    config_extra.update(basics)

    print('training with agent type ', agent_type)
    tune.run(
        agent_type,
        name=name_xp,
        local_dir=local_dir,
        stop={"training_iteration": iterations},  
        checkpoint_at_end=True,
        checkpoint_freq=1,
        verbose=verbose,
        config=config_extra,
    )

    return

if __name__ == '__main__':

    train_config = load_experiment_data('train.yml')
    config_extra = train_config['config_extra']
    env_config = train_config['env_config']
    agent_type = train_config['run_settings']['agent_type']
    iterations = train_config['run_settings']['iterations']

    if ('order_fixed_size' in env_config):
        env_config['order_fixed_size'] = int(float(env_config['order_fixed_size']))

    if 'done_ratio' in env_config:
        env_config['done_ratio'] = float(env_config['done_ratio'])

    if ('background_config_extra_kvargs' in env_config) and (env_config['background_config_extra_kvargs'] == "changes_fc"):

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
        "momentum_agent_params": momentum_agent_config
        }

        env_config['background_config_extra_kvargs'] = changes_fc

    train_ray_model(agent_type=agent_type, iterations=iterations, verbose=0, config_extra=config_extra, env_config=env_config)