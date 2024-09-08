from test_utils import *

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

    if ('continuous_mode' in env_config) and (env_config['continuous_mode']):
        continuous = True
    else:
        continuous = False

    if ('order_fixed_size' in env_config):
        env_config['order_fixed_size'] = int(float(env_config['order_fixed_size']))

    if 'done_ratio' in env_config:
        env_config['done_ratio'] = float(env_config['done_ratio'])

    if ('background_config_extra_kvargs' in env_config) and (env_config['background_config'] == "flash_crash"):
    #if (env_config['background_config'] == "rmsc07") and (env_config['background_config'] == "flash_crash"):

        oracle_config = GBMOracleConfig(mu=1e-9, sigma=0.0135)
        mm_config = MarketMakerAgentConfig(price_skew_param=4, skew_beta=1e-4, pov=0.025, wake_up_freq='1s', subscribe=False, subscribe_freq='1s', subscribe_num_levels=10)
        # mm_config = MarketMakerAgentConfig(price_skew_param=4, wake_up_freq=1e9 * (5) , subscribe=False, subscribe_freq='1s', subscribe_num_levels=10)
        value_agent_config = ValueAgentConfig(kappa_limit=0.3, kappa_mkt=0.1, mean_wakeup_gap=1e9)
        # value_agent_config = ValueAgentConfig(kappa_limit=0.3, kappa_mkt=0.1, mean_wakeup_gap=1e8 * (5))
        momentum_agent_config = MomentumAgentConfig(beta_limit=50, beta_mkt=20, wake_up_freq='1s', subscribe=False)
        # momentum_agent_config = MomentumAgentConfig(beta_limit=50, beta_mkt=20, wake_up_freq=1e9 * (5), subscribe=False)
        exchange_config = ExchangeConfig(log_orders=True)
        institutional_config = InstitutionalTraderAgentConfig(inventory=1e6, sell_frequency='1s', sell_volume_factor=1000, trigger_time="09:30:02")

        changes_fc = {
        "num_noise_agents": 50,
        "num_value_agents": 10,
        "num_mm_agents": 19,
        "num_long_momentum_agents": 5,
        "num_short_momentum_agents": 5,
        "oracle_params": oracle_config,
        "mm_agent_params": mm_config,
        "value_agent_params": value_agent_config,
        "momentum_agent_params": momentum_agent_config,
        "institutional_params": institutional_config,
        "exchange_params": exchange_config,
        }

        env_config['background_config_extra_kvargs'] = changes_fc

    env = gym.make("markets-daily_investor-v0", debug_mode=True, **env_config)

    policy = policyRL(config_extra=config_extra, env_config=env_config, path=path)
    best_trial_path = policy.get_directory()
    # policy = policyPassive(continuous=False)
    # policy = policyTimid()
    # policy = policyRandom()
    # best_trial_path = '.'+path[8:]

    if not experiment_mode:

        save_dir = os.path.join('./experiment_history', best_trial_path)
        save_dir += '_seed='+str(seed)
        if 'extra_name' in test_config['single_run_settings']:
                save_dir += '_' + test_config['single_run_settings']['extra_name']

        os.makedirs(save_dir, exist_ok=True)
        test_data = test_ray_model(policy, seed=seed, env=env)
        get_analytics(test_data, save_dir, passive, continuous, path, seed, env)

    else:

        seeds = [i for i in range(seed_number)]

        for seed in seeds:

            save_dir = os.path.join('./experiment_history', experiment_name, best_trial_path)
            save_dir += '_seed='+str(seed)
            os.makedirs(save_dir, exist_ok=True)

            test_data = test_ray_model(policy, seed=seed, env=env)

            if passive:
                indicators, indicators_passive = get_analytics(test_data, save_dir, passive, continuous, path, seed, env)
                experiment_dict['passive'][seed] = indicators_passive

            else:
                indicators = get_analytics(test_data, save_dir, passive, continuous, path, seed, env)

            experiment_dict['test'][seed] = indicators

            save_dict_to_json(os.path.join('./experiment_history', experiment_name, 'analysis.json'), experiment_dict)
            


    




