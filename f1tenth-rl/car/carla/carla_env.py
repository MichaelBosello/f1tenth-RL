import glob
import os
import sys
import time
import random
from threading import Thread

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

try:
    from car.carla.carla_car_control import Drive
    from car.carla.carla_sensors import Sensors
    from car.carla.carla_safety_control import SafetyControl
except:
    from carla_car_control import Drive
    from carla_sensors import Sensors
    from carla_safety_control import SafetyControl

DELTA = 0.05
NO_RENDERING = False
SHOW_LIDAR = False
SLEEP_RT = True


class CarlaEnv:

    def __init__(self, main_sensor):
        self.exit_flag = False
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        self.client.load_world('Town03')
        self.world = self.client.get_world()
        self.original_settings = self.world.get_settings()
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = DELTA
        self.settings.synchronous_mode = True
        self.settings.no_rendering_mode = NO_RENDERING
        self.world.apply_settings(self.settings)

        blueprint_library = self.world.get_blueprint_library()
        model_3 = blueprint_library.filter('model3')[0]
        transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(model_3, transform)

        imu_bp = self.world.get_blueprint_library().find('sensor.other.imu')
        imu_location = carla.Location(-6,0,3)
        imu_rotation = carla.Rotation(0,0,0)
        imu_transform = carla.Transform(imu_location,imu_rotation)
        self.imu = self.world.spawn_actor(imu_bp,imu_transform,attach_to=self.vehicle)

        self.sensors = Sensors(self.vehicle, self.world, DELTA, main_sensor)
        self.control = Drive(self.vehicle, self.world, self.sensors)
        self.safety_control = SafetyControl(self.control, self.sensors)

        process = Thread(target=self._update_server)
        process.daemon = True
        process.start()

        if SHOW_LIDAR:
            self.sensors.open_lidar_window()

        time.sleep(3)

    def _update_server(self):
        while self.exit_flag == False:
            transform = self.imu.get_transform()
            spectator_transform = carla.Transform(
            carla.Location(transform.location.x, transform.location.y, transform.location.z),
            carla.Rotation(transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll))
            self.world.get_spectator().set_transform(spectator_transform)
            self.world.tick()
            if SLEEP_RT:
                time.sleep(DELTA)

    def update_view(self):
        '''
        This function is called by the main thread to update the view of the LIDAR as
        only the main thread can call open3d functions
        '''
        if SHOW_LIDAR:
            self.sensors.update_lidar_window()

    def destroy(self):
        self.exit_flag = True
        self.vehicle.destroy()
        self.imu.destroy()
        self.sensors.destroy()
        self.world.apply_settings(self.original_settings)


PRINT_SPEED = False

if __name__ == '__main__':
    main_sensor = '3d-lidar'
    env = CarlaEnv(main_sensor)
    exit_flag = False

    def input_thread():
        global exit_flag
        while True:
            print("Write command")
            cmd = input()
            if cmd == "w":
                env.control.forward()
            if cmd == "a":
                env.control.left()
            if cmd == "d":
                env.control.right()
            if cmd == "s":
                env.control.backward()
            if cmd == " ":
                env.control.stop()
            if cmd == "r":
                env.control.reset_position()
            if cmd == "u":
                env.safety_control.unlock_brake()
            if cmd == "q":
                exit_flag = True
                break

    process = Thread(target=input_thread)
    process.daemon = True
    process.start()

    while exit_flag is False:
        if PRINT_SPEED:
            print("######################################")
            print(env.get_car_linear_velocity())
        if SHOW_LIDAR:
            env.update_view()
        else:
            time.sleep(1)

    env.destroy()
    exit()




