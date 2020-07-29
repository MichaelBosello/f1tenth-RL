# Experiments Settings
Experiments results and trained models are in the release section
## PC with i7 (simulator)
Use the repo as is
## Real car (Jetson TX2) - slow speed (1/8 of max speed)
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
```
f1tenth-rl/car_env.py
```
USE_VELOCITY_AS_REWARD = True
ADD_LIDAR_DISTANCE_REWARD = True
```
## CNN2D on PC (simulator)
f1tenth-rl/rl_car_driver.py
```
epsilon-decay = 0.99992
slowdown-cycle = False
lidar-to-image = True
```

# Evaluation only
## Real car - high speed (1/4 of max speed)
Same as above, plus:

f1tenth-rl/rl_car_driver.py
```
gpu-time = 0.001
train-epoch-steps = 0
save-model-freq = 1000000
```
f1tenth-rl/car_control.py
```
MAX_SPEED_REDUCTION = 4
STEERING_SPEED_REDUCTION = 4
BACKWARD_SECONDS = 2
```