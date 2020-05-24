# f1tenth-RL

Implementation of DQN for autonomous racing using lidar data

It is based on the [f1tenth project](https://f1tenth.org/)

ROS is used to control the car motor, servo and acquire the lidar data

*It can be used on both the real f1tenth car and on its simulator*

## Installation

1) Install [ROS Melodic (desktop-full)](http://wiki.ros.org/melodic/Installation/Ubuntu)

2) Setup the car *or* the simulator:
    + Real 1/10 scale car

        Follow the four tutorials (Building the car, system configuration, installing firmware, driving the car) at https://f1tenth.org/build.html to build and setup the car.



    + Simulator

        `$​ ​mkdir simulator`

        `$​ ​cd simulator`

        `$ git clone https://github.com/f1tenth/f1tenth_labs.git`

        `$ catkin_make`

3) Install the dependencies to run ROS with python3

    `$ sudo apt-get install python3-pip python3-yaml`

    `$ sudo pip3 install rospkg catkin_pkg`

    `$ sudo apt-get install ros-melodic-ackermann-msgs`

4) Clone this repo

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

 `$ python3 [todo]`

### Simulator
Launch the f1tenth simulator:
+ Go to the working directory of the simulator (*/simulator*)

`$ source devel/setup.bash`

`$ roslaunch f110_simulator simulator.launch`

Run the RL algorithm:

+ Go to the f1tenth-rl directory

`$ python3 [todo] --simulator`