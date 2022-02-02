# wind-farm-env
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains the source code and data for the experiments presented in
_Deep Reinforcement Learning for Active Wake Control_.

## How to replicate the experiments

Because each experiment has many parameters, we use configuration files to  describe each experiment. The configuration files for the experiments presented in the paper can be found in `./code/configs`: `action_representations_*.yml` for the action representation experiment (Section 4.1), and `noisy_*.yml` for the noisy observation experiment (Section 4.2). You can also create new configuration files to run your own experiments.

To replicate the experiments in the paper, run `python3 code/main.py --config CONFIG_FILE`, where `CONFIG_FILE` is a path to a configuration file.
The output data is written to a directory specified in the configuration, in Tensorboard format. You can use `./code/tensorboard_to_csv.py --path PATH` to convert the output from `PATH` to a `.csv`-file.

## Contents

- `./code` contains the source code:
    - `agent` for implementations of RL agents;
    - `config` for configuration files used in the experiments;
    - `wind_farm_gym` for the environment.
- `./data` contains the datasets used in the paper:
    - `wind_data` contains the measurements from Hollandse Kust Noord (site B) used to estimate the transition model in Section 3.4.
    - the other subdirectories contain the results of the experiments:
        - `action_representations_tunnel` for the experiment presented in Section 4.1.
        - `noisy_1` .. `noisy_7` for the experiment presented in Section 4.2; each folder corresponds to a different level of noise, 1% .. 7%.

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
