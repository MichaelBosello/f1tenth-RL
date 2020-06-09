# f1tenth-RL
# (Deep Reinforcement Learning Autonomous Driving Using Lidar in the Physical World)

Implementation of *DQN* for autonomous racing using *lidar* data

It is designed to running on [f1tenth cars](https://f1tenth.org/)

*ROS* is used to control the car motor, servo and acquire the lidar data

*It can be used on both the real f1tenth car and on its simulator*

The DQN implementation provides several techniques to improve performances like double DQN, replay buffer, state history, prioritized sampling. It has various parameters (see below) that one can modify to fit the specific environment

Model saving, replay buffer serialization, and tensorboard logging are provided 

## Introduction
In our experiment, we want to test *DQN* training directly in the *real world* through realistic 1/10 scale car prototypes capable of performing training in real-time. This allows us to explore the use of RL for autonomous driving in the physical world in a cheap and safe way. In this setting, the driver agent faces all the problems of a not simulated environment, including sensors noise and actuators’ unpredictability. We start with the implementation of DQN on the car, and then we try various alterations to improve performance like reward function engineering and hyper-parameters tuning. In the end, the agent successfully learned a control policy, based on lidar data, to drive in two circuits.

## Installation

1) Install [ROS Melodic (desktop-full)](http://wiki.ros.org/melodic/Installation/Ubuntu)

2) Install the dependencies

    `$ sudo apt-get install python3-pip python3-yaml`

    `$ sudo pip3 install rospkg catkin_pkg`

    `$ sudo apt-get install ros-melodic-ackermann-msgs``

3) Setup the car *or* the simulator:
    + Real 1/10 scale car

        Follow the four tutorials (Building the car, system configuration, installing firmware, driving the car) at https://f1tenth.org/build.html to build and setup the car

        You need to add to the back of the car one or two IR sensors that are used to safely go backwards when an episode ends (because the hokuyo lidar covers only 270 degrees). Configure your pinout in the file *car/sensors.py*. The Orbitty Carrier has its own method to use gpio (i.e. bash commands). Check the numbers associated to the pins [here](http://connecttech.com/resource-center/kdb342-using-gpio-connect-tech-jetson-tx1-carriers/) and [here](http://connecttech.com/pdf/CTIM-ASG003_Manual.pdf). If you use the developer kit board, you have to implement the methods using Jetson.GPIO. If you use another board, find out how to use gpio and implement the methods

    + Simulator

        `sudo apt-get install ros-melodic-map-server`

        `sudo apt-get install ros-melodic-joy`

        `$​ ​mkdir simulator`

        `$​ ​cd simulator`

        `$​ ​mkdir src`

        `$​ ​cd src`

        `$ git clone https://github.com/f1tenth/f1tenth_labs.git`

        `$​ ​cd ../`

        `$ catkin_make`

4) Install tensorflow 2.1.x

    + If you are on a *PC* (i.e. simulator)

        `$ pip3 install tensorflow`

    + In the real car, you need to install tensorflow for Jetson with Cuda (installed via JetPack)

        follow the [tutorial](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html) for your specific Jetson (verify that it will install at least v2.1.x, otherwise execute the upgrade command)


5) Clone this repo

    `$ git clone https://github.com/MichaelBosello/f1tenth-RL.git`

## Run

### Real car

> To let the control to the algorithm: 
> + you need to hold down the corresponding joystick button (the button depends on your joystick and on the configuration in joy_teleop.yaml, you can test your joystick [here](https://html5gamepad.com/))
> + Alternatively, you should change the **priority** of the control topics: edit the configuration of **low_level_mux.yaml** in f110_ws/f1tenth_system/racecar/racecar/config. Set priority of topic input/teleop to 1, input/safety to 2 and input/navigation to 3.

Launch the f1tenth system:
+ Go to the working directory of the f1tenth system (*/f110_ws*)

`$ source devel/setup.bash`

`$ roslaunch racecar teleop.launch`

Run the RL algorithm:

+ Go to the f1tenth-rl directory

 `$ python3 rl_car_driver.py`

### Simulator
Launch the f1tenth simulator:
+ Go to the working directory of the simulator (*/simulator*)

`$ source devel/setup.bash`

`$ roslaunch f110_simulator simulator.launch`

Run the RL algorithm:

+ Go to the f1tenth-rl directory

`$ python3 rl_car_driver.py --simulator`

#### Simulator options:
+ The guide of the simulator is in the readme *src/f1tenth_labs/f110_simulator/README/md*

+ You may want to change the simulator options, check out *src/f1tenth_labs/f110_simulator/params.yaml*

+ If you want to change the circuit, you must edit *src/f1tenth_labs/f110_simulator/launch/simulator.launch*

    Search for `<arg name="map" default="$(find f110_simulator)/maps/levine.yaml"/>`
    Change *levine* (the default map) with one map present in the folder *src/f1tenth_labs/f110_simulator/maps*

## Experimenting with parameters
You can change several parameters when you run the program as command-line arguments. Use *-h* to see the argument help. 
You can check the list of arguments and change their default value in *rl_car_driver.py*

You can change the model size and architecture by changing the function *__build_q_net* in *dqn.py*. We provide two basic networks: a fully connected and a CNN

You can change the behavior of the actions in *car_env.py*. You can change the actions available to the agent by updating the *action_set*

Keep in mind that to use a trained model, you must have the same network size, the same input (number and dimension of frame), and the same number of actions

For safety reasons, the car doesn't run at max speed. If you want the car to go faster, modify the constants in *car/car_control.py*

## Trained models
We will provide pre-trained models in the release section

## Source code structure
The package *car* provides the interfaces to the car sensors (*sensors.py*) and actuators (*car_control.py*). It contains also a module that ensure the car will not (strongly) hit obstacles (*safety_control.py*)

*car_env.py* is the Reinforcement Learning environment.

*rl_car_driver.py* contains the training cycle.
*dqn.py* includes the NN and the DQN algorithm.

*state.py* creates states by preprocesing data and stacking them to form a history. It also provides the compression to save RAM.

*replay.py* manage the samples and the replay buffer.

## Use on alternative cars/simulators

One can still use the dqn algorithm in alternative driving environments. You only need to implement your interfaces to the car sensors and actuators. To do so, implement your version of the files in the directory */car*. You are also free to not use ROS at all.