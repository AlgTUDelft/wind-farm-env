import argparse
import random
import sys
import os
import yaml
from gym.wrappers import TimeLimit

from agent import SACAgent, NaiveAgent, FlorisAgent
from agent.deep import TD3Agent
from wind_farm_gym import WindFarmEnv
from wind_farm_gym.wind_process import MVOUWindProcess


def run(config, seed):
    eval_steps = config.get('eval_steps', 1000)
    training_steps = config.get('train_steps', 10000)
    n_eval_env = config.get('n_eval_env', 1)
    random.seed(seed)
    data_generation_train_seed = random.randint(0, 2 ** 32 - 1)
    data_generation_eval_seeds = [random.randint(0, 2 ** 32 - 1) for _ in range(n_eval_env)]
    eval_seed = random.randint(0, 2 ** 32 - 1)
    train_seed = random.randint(0, 2 ** 32 - 1)

    time_delta = config.get('time_delta', 1)

    # Create the project results directory if needed
    directory = os.path.join(config.get('directory', 'data'),
                             config.get('name', 'WindFarm'),
                             f'seed_{seed}')
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Load the wind data if it already exists, otherwise generate and save it
    print("Retrieving the wind data...")
    if config.get('wind_process') is not None:
        data_path = os.path.join(directory, 'wind_data')
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        wind_config = config.get('wind_process', {})
        if wind_config.get('type', None) == 'mvou':
            # Read the existing data
            eval_processes = [
                MVOUWindProcess.switch_to_csv(
                    os.path.join(data_path, f'evaluation_data_{i}.csv'),
                    time_steps=eval_steps,
                    time_delta=time_delta,
                    properties=wind_config.get('properties', None),
                    seed=data_generation_eval_seeds[i]
                )
                for i in range(n_eval_env)
            ]
            train_process = MVOUWindProcess.switch_to_csv(
                os.path.join(data_path, 'training_data.csv'),
                time_steps=training_steps,
                time_delta=time_delta,
                properties=wind_config.get('properties', None),
                seed=data_generation_train_seed
            )
        else:
            raise NotImplementedError
    else:
        eval_process, train_process = None, None

    # Create environments
    print("Setting  up the environments...")
    env_config = config.get('environment', {})
    env_config['time_delta'] = time_delta
    train_env = TimeLimit(WindFarmEnv(wind_process=train_process, **env_config), training_steps)
    eval_envs = [
        TimeLimit(WindFarmEnv(wind_process=eval_process, **env_config), eval_steps)
        for eval_process in eval_processes
    ]

    # Make a list of agents, check if their  data already exists, and  skip them if it does
    print("Making agent list...")
    agents = config.get('agents', [])
    results_directory = os.path.join(directory, 'results')

    for agent_description in agents:
        name = agent_description.get('name', agent_description.get('type', None))
        assert name is not None, 'agents must have a name and/or a type'
        agent_directory = os.path.join(results_directory, name)
        if os.path.exists(agent_directory):
            agent_description['exists'] = True
    agents = [agent for agent in agents if not agent.get('exists', False)]

    # Run the agents
    for agent_description in agents:
        # create a directory for  the agent
        name = agent_description.get('name', agent_description.get('type', None))
        agent_directory = os.path.join(results_directory, name)
        tb_log = agent_directory if config.get('log', True) else None

        run_config = {
            'log_directory': os.path.join(agent_directory),
            'total_steps': training_steps,
            'render': config.get('render', False),
            'rescale_rewards': config.get('rescale_rewards', True),
            'reward_range': config.get('reward_range'),
            'log': config.get('log', True),
            'log_every': config.get('log_every', 100),
            'eval_envs': eval_envs,
            'eval_steps': eval_steps,
            'eval_every': config.get('eval_every', 100),
        }

        print(f'Creating agent {name}...')
        parameters = agent_description.get('parameters', {})
        eval_once = False
        eval_only = False
        if agent_description['type'] == 'naive':
            agent = NaiveAgent(name, train_env)
            eval_once = True
            eval_only = True
        elif agent_description['type'] == 'floris':
            agent = FlorisAgent(name, train_env, **parameters)
            eval_once = True
            eval_only = True
        elif agent_description['type'] == 'sac':
            agent = SACAgent(name, train_env, **parameters)
        elif agent_description['type'] == 'td3':
            agent = TD3Agent(name, train_env, **parameters)
        else:
            agent = None

        if agent is not None:
            agent.run(eval_once=eval_once, eval_only=eval_only, **run_config)
            # agent.save()
            agent.close()

    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wind Farm Experiment')
    # Common arguments
    parser.add_argument('--config', type=str, default='configs/action_representations_baselines.yml',
                        help='the configuration file for the experiment')

    args = parser.parse_args()

    # Read the configuration file
    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit()

    # Set the seeds
    seeds = config.get('seed', random.randint(0, 2 ** 32 - 1))
    seeds = list(seeds)
    for seed in seeds:
        run(config, seed)
