# wind-farm-env
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains the source code and data for the experiments presented in
_Deep Reinforcement Learning for Active Wake Control_.

## How to Use This Repository

### Replicating the Experiments

Because each experiment has many parameters, we use configuration files to  describe each experiment. The configuration
files for the experiments presented in the paper can be found in `./code/configs`: `action_representations_*.yml` for
the action representation experiment (Section 4.1), and `noisy_*.yml` for the noisy observation experiment
(Section 4.2). You can also create new configuration files to run your own experiments.

Make sure that you are using Python 3.8 or newer. You also need to install required packages by running
`python3 -m pip install -r requirements.txt` from the project's folder.

To replicate the experiments in the paper, run `python3 code/main.py --config CONFIG_FILE`, where `CONFIG_FILE` is a
path to a configuration file. The output data is written to a directory specified in the configuration, in Tensorboard
format. You can use `./code/tensorboard_to_csv.py --path PATH` to convert the output from `PATH` to a `.csv`-file.

### Building from the Source Code

If you are interested in using the wind farm environment with your own reinforcement learning agents, you can build and
install a Python package that will make the environment available on your machine.

First, make sure that you have `build` installed by running `python3 -m pip install --upgrade build`.

Next, build the package by running `python3 -m build` from the project's directory. This will create `./dist` with 
a `wind_farm_gym-VERSION-py3-none-any.whl` file in it, where `VERSION` is the current release version.

Finally, install the package by running `python3 -m pip install wind_farm_gym-VERSION-py3-none-any.whl`, or
`python3 -m pip install wind_farm_gym-VERSION-py3-none-any.whl --force-reinstall` if you want to reinstall an existing
installation. This should install the package and its dependencies.

To test that the package is available and working, you can run the following script:
```python
from wind_farm_gym import WindFarmEnv

# Initialize the environment with 3 turbines positioned 750 meters apart in a line
env = WindFarmEnv(turbine_layout=([0, 750, 1500], [0, 0, 0]))

obs = env.reset()
for _ in range(1000):                # Repeat for 1000 steps
    a = env.action_space.sample()    # Choose an action randomly
    obs, reward, _, _ = env.step(a)  # Perform the action
    env.render()                     # Render the environment; remove this line to speed up the process
env.close()
```

## Contents

- `./code` contains the source code:
    - `agent` for implementations of RL agents;
    - `config` for configuration files used in the experiments;
    - `wind_farm_gym` for the environment.
- `./data` contains the datasets used in the paper:
    - `wind_data` contains the measurements from Hollandse Kust Noord (site B) used to estimate the transition model in
Section 3.4.
    - the other subdirectories contain the results of the experiments:
        - `action_representations_tunnel` for the experiment presented in Section 4.1.
        - `noisy_1` ... `noisy_7` for the experiment presented in Section 4.2; each folder corresponds to a different
level of noise, 1% ... 7%.

## Citation

Please, cite the paper if you use it:

```
@inproceedings{Neustroev2022,
  title     = {Deep Reinforcement Learning for Active Wake Control},
  author    = {Neustroev, Grigory and Andringa, Sytze P.E. and Verzijlbergh, Remco A. and de~Weerdt, Mathijs M.},
  booktitle = {International Conference on Autonomous Agents and Multi-Agent Systems},
  year      = {2022},
  address   = {Online},
  publisher = {IFAAMAS},
  month     = {May},
  numpages  = {10}
}
```
