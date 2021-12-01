import random
import time
import carla


SPEED = 0.5
STEERING_ANGLE = 1

class Drive():
    def __init__(self, vehicle, world, sensors):
        self.vehicle = vehicle
        self.world = world
        self.sensors = sensors

    def forward(self):
        self.send_drive_command(SPEED, 0)
    def backward(self):
        self.send_drive_command(SPEED, 0, reverse=True)
    def stop(self):
        self.send_drive_command(0, 0, brake=1)
    def right(self):
        self.send_drive_command(SPEED, STEERING_ANGLE)
    def left(self):
        self.send_drive_command(SPEED, -STEERING_ANGLE)

    def send_drive_command(self, speed, steering_angle, brake=0, reverse=False):
        self.vehicle.apply_control(carla.VehicleControl(speed, steering_angle, brake=brake, reverse=reverse))

    def reset_position(self):
        self.stop()
        time.sleep(0.1)
        transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle.set_transform(transform)
        time.sleep(1)