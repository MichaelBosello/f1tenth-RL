# Enabling deep reinforcement learning autonomous driving by 3D-LiDAR point clouds [[Paper](https://doi.org/10.1109/CCNC49033.2022.9700730)]

**This branch is dedicated to a different paper compared to the one on the main.** Please refer to the documentation on the main branch for in-depth usage.

The main change in this branch is the implementation of a _car_env_ interface for the [CARLA Simulator](https://carla.org). The interface is available in the directory [f1tenth-rl/car/carla](f1tenth-rl/car/carla)

## Paper

If you use this repo **branch**, please cite our paper [[DOI](https://doi.org/10.1117/12.2644369)].

```
@inproceedings{10.1117/12.2644369,
author = {Yuhan Chen and Rita Tse and Michael Bosello and Davide Aguiari and Su-Kit Tang and Giovanni Pau},
title = {{Enabling deep reinforcement learning autonomous driving by 3D-LiDAR point clouds}},
volume = {12342},
booktitle = {Fourteenth International Conference on Digital Image Processing (ICDIP 2022)},
editor = {Xudong Jiang and Wenbing Tao and Deze Zeng and Yi Xie},
organization = {International Society for Optics and Photonics},
publisher = {SPIE},
pages = {1234218},
keywords = {Deep reinforcement learning, autonomous driving, multi-agent simulation, 3D-LiDAR, point clouds},
year = {2022},
doi = {10.1117/12.2644369},
URL = {https://doi.org/10.1117/12.2644369}
}
```

## Introduction
_Abstract_ &mdash; Autonomous driving holds the promise of revolutionizing our lives and society. Robot drivers will run errands such as commuting, parking cars, or taking kids to school. It is expected that, by the mid-century, humans will drive only for their pleasure. Autonomous vehicles will increase the efficiency and safety of the transportation system by reducing accidents and increasing the overall system capacity. Current autonomous driving systems are based on supervised learning that relies on massive, labeled data. It takes a lot of time, resources, and manpower to produce such data sets. While this approach is achieving remarkable results, the required effort to produce data becomes a limiting factor for general driving scenarios. This research explores Reinforcement Learning to advance autonomous driving models without labeled data. Reinforcement Learning is a learning paradigm that uses the concept of rewards to autonomously discover, through trial & error, how to solve a task. This work uses the LiDAR sensor as a case study to explore the effectiveness of Reinforcement Learning in interpreting complex data. LiDARs provide a dynamic high time-space definition map of the environment and it could be one of the key sensors for autonomous driving.


## Installation

1) Install [CARLA Simulator](https://carla.readthedocs.io/en/latest/start_quickstart/)

2) Install the dependencies

    `$ sudo apt-get install python3-yaml`
    `$ pip3 install tensorflow`

3) Clone this branch

    `$ git clone https://github.com/MichaelBosello/f1tenth-RL.git -b carla-sim`

## Run

### Simulator
Launch the CARLA Simulator:
+ Go to the CARLA installation directory (e.g. _/opt/carla-simulator/bin/_)

`$ ./CarlaUE4.sh`

Run the RL algorithm:

+ Go to the f1tenth-rl directory

`$ python3 rl_car_driver.py --simulator`

## Experimenting with parameters
You can change several parameters when you run the program as command-line arguments. Use *-h* to see the argument help. 
You can check the list of arguments and change their default value in *rl_car_driver.py*

You can change the model size and architecture by changing the function *__build_q_net* in *dqn.py*.

You can change the behavior of the actions in *carla_car_control.py*. You can change the actions available to the agent by updating the *action_set*

Keep in mind that to use a trained model, you must have the same network size, the same input (number and dimension of frame), and the same number of actions

If you want to modify the car velocity, modify the constants in *carla_car_control.py*

### Load a model
You can use the --model argument to load a trained model, e.g.:

`python3 rl_car_driver.py --model=./models`
