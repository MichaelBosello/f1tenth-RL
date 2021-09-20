# Experiments Settings (chronological order)
Experiments results and trained models are in the release section

## F1 racetracks experiment (run-f1-tracks.zip)
Simulation, PC with i7, maps: f1_aut, f1_esp, f1_gbr, f1_mco

Use the repo as is

---
**Testing**
f1tenth-rl/rl_car_driver.py
```
gpu-time = 0.0001
train-epoch-steps = 0
max-step-limit = 10000000
save-model-freq = 1000000
```


## Sim2real experiment 2: transfer learning (run-unibo-roof.zip)
Training: simulation, PC with i7, unibo-roof map

Testing: real car with Jetson NX

f1tenth-rl/rl_car_driver.py
```
epsilon-decay = 0.99993
gpu-time = 0.018
add-velocity = False
```

f1tenth-rl/car/safety_control.py
```
USE_TTC_SIM = True
TTC_THRESHOLD_SIM = 1.21
ONLY_EXTERNAL_BARRIER = True
```

f1tenth-rl/car/car_control.py
```
MAX_SPEED_REDUCTION_SIM = 4.5
STEERING_SPEED_REDUCTION_SIM = 4.5
BACKWARD_SPEED_REDUCTION_SIM = 4.5
```

f1tenth-rl/car_env.py
```
USE_VELOCITY_AS_REWARD = False
ADD_LIDAR_DISTANCE_REWARD = False
...
self.action_set = [0, 1, 2]
```

---
**Testing**
f1tenth-rl/rl_car_driver.py
```
gpu-time = 0.001
train-epoch-steps = 0
save-model-freq = 1000000
max-step-limit = 10000000
reduce-lidar-data = 36
cut-lidar-data = 5 
```

## NNs comparison experiment 
Simulation, PC with i7, hairpin-track map

f1tenth-rl/rl_car_driver.py
```
add-velocity = False
```

f1tenth-rl/car/car_control.py
```
MAX_SPEED_REDUCTION_SIM = 3
STEERING_SPEED_REDUCTION_SIM = 3
BACKWARD_SPEED_REDUCTION_SIM = 3
BACKWARD_SECONDS_SIM = 1.8
```
f1tenth-rl/car/safety_control.py
```
EUCLIDEAN_THRESHOLD_SIM = 0.48
USE_TTC_SIM = False
```

f1tenth-rl/car_env.py
```
USE_VELOCITY_AS_REWARD = False
ADD_LIDAR_DISTANCE_REWARD = False
...
self.action_set = [0, 1, 2]
...

self.control.forward()
            reward = 0.2
...
self.control.right()
            reward = 0.05
...
self.control.left()
            reward = 0.05
```

### CNN1D (run-hairpin-track-simulator-cnn1d.zip)

f1tenth-rl/rl_car_driver.py
```
epsilon-decay = 0.9999342
observation-steps = 400
gpu-time = 0.014
reduce-lidar-data = 30
cut-lidar-data = 8 
```

### Dense (run-hairpin-track-simulator-dense.zip)
same as CNN1D, plus:

f1tenth-rl/dqn.py
```
    def __build_q_net(self):
        ...
            return self.__build_dense()
```

### CNN2D (run-hairpin-track-simulator-cnn2d.zip)
same as CNN1D, plus:

f1tenth-rl/rl_car_driver.py
```
epsilon-decay = 0.99992
slowdown-cycle = False
lidar-to-image = True
```

## Sim2real experiment 1: training on the physical car (run-real-car.zip)

Jetson TX2 - slow speed (1/8 of max speed)

f1tenth-rl/car/car_control.py
```
MAX_SPEED_REDUCTION = 8
STEERING_SPEED_REDUCTION = 8
BACKWARD_SPEED_REDUCTION = 8
BACKWARD_SECONDS = 1.8
```

f1tenth-rl/car/safety_control.py
```
USE_TTC = True
TTC_THRESHOLD_REAL_CAR = 0.62
EUCLIDEAN_THRESHOLD_REAL_CAR = 0.08
```

f1tenth-rl/rl_car_driver.py
```
epsilon-decay = 0.9998
observation-steps = 500
repeat-action = 5
gpu-time = 0.03
reduce-lidar-data = 36
cut-lidar-data = 5
max-distance-norm = 5
save-model-freq = 5000
add-velocity = False
```

f1tenth-rl/car_env.py
```
USE_VELOCITY_AS_REWARD = True
ADD_LIDAR_DISTANCE_REWARD = True
LIDAR_DISTANCE_WEIGHT = 0.1
VELOCITY_NORMALIZATION = 0.55
...
self.action_set = [0, 1, 2]
```
---
Evaluation only - high speed (1/4 of max speed)

Same as above, plus:

f1tenth-rl/rl_car_driver.py
```
gpu-time = 0.001
train-epoch-steps = 0
save-model-freq = 1000000
max-step-limit = 10000000
```
f1tenth-rl/car/car_control.py
```
MAX_SPEED_REDUCTION = 4
STEERING_SPEED_REDUCTION = 4
BACKWARD_SECONDS = 2
```