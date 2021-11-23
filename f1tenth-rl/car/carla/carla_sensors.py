import math
import numpy as np
import carla

try:
    import open3d as o3d
    from matplotlib import cm
    open3d_installed = True
except ImportError:
    open3d_installed = False


# Velodyne VLP-16 params in comment
LIDAR_CHANNELS = 16 #16
LIDAR_RANGE = 100 #100
LIDAR_POINTS_PER_SCAN = 40000 #300000
LIDAR_UPPER_FOW = 15 #15
LIDAR_LOWER_FOW = -15 #-15
LIDAR_HORIZONTAL_FOV = 270 #360
LIDAR_LOCATION_X = 0.5
LIDAR_LOCATION_Z = 1.9

LIDAR_2D_CHANNELS = 1
LIDAR_2D_RANGE = 50
MAX_LIDAR_RANGE = 184460000000000000
LIDAR_2D_POINTS_PER_SCAN = 1080
LIDAR_2D_UPPER_FOW = 0
LIDAR_2D_LOWER_FOW = 0
LIDAR_2D_HORIZONTAL_FOV = 270
LIDAR_2D_LOCATION_X = 2
LIDAR_2D_LOCATION_Z = 0.1

# the 'a' coefficient that measures the LIDAR instensity loss per meter in the eq: e^-a*d
# it depends on the lidar wavelenght and meteo conditions. It should affects the LIDAR performance during rain etc.
# Wavelength of Velodyne: 905 nm
# not sure how to calculate it, keeping default value
LIDAR_ATTENUATION_RATE = 0.004

# to be changed to improve realism
LIDAR_DROPOFF_RATE = 0
LIDAR_DROPOFF_INTENSITY_LIMIT = 0
LIDAR_DROPOFF_ZERO_INTENSITY = 0
LIDAR_NOISE_STD = 0

class Sensors():
    def __init__(self, vehicle, world, world_delta, main_sensor='3d-lidar'):
        self.vehicle = vehicle
        self.world = world
        self.world_delta = world_delta
        self.main_sensor = main_sensor
        self.custom_lidar_callbacks = []
        self.collision_callbacks = []
        self.lane_callbacks = []
        self._lidar_data = None

        self.vis = None
        self.lidar_rotation_frequency = 1/world_delta if world_delta>0 else 1


        lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        if main_sensor == '3d-lidar':
            self.lidar_range = LIDAR_RANGE
            self.lidar_points_per_scan = LIDAR_POINTS_PER_SCAN
            self.lidar_points_per_seconds = self.lidar_points_per_scan * self.lidar_rotation_frequency
            lidar_bp.set_attribute('channels',str(LIDAR_CHANNELS))
            lidar_bp.set_attribute('range',str(LIDAR_RANGE))
            lidar_bp.set_attribute('points_per_second',str(self.lidar_points_per_seconds))
            lidar_bp.set_attribute('rotation_frequency',str(self.lidar_rotation_frequency))
            lidar_bp.set_attribute('upper_fov',str(LIDAR_UPPER_FOW))
            lidar_bp.set_attribute('lower_fov',str(LIDAR_LOWER_FOW))
            lidar_bp.set_attribute('horizontal_fov',str(LIDAR_HORIZONTAL_FOV))
            x = LIDAR_LOCATION_X
            z = LIDAR_LOCATION_Z
        elif main_sensor == '2d-lidar':
            self.lidar_range = LIDAR_2D_RANGE
            self.lidar_points_per_scan = LIDAR_2D_POINTS_PER_SCAN
            self.lidar_points_per_seconds = self.lidar_points_per_scan * self.lidar_rotation_frequency
            lidar_bp.set_attribute('channels',str(LIDAR_2D_CHANNELS))
            # When the Lidar point is too far (out of range), the sensor will return NOTHING instead of a default value on that direction. see https://github.com/carla-simulator/carla/issues/2262
            # we will take all the possible points and exclude the out-of-range later
            lidar_bp.set_attribute('range',str(MAX_LIDAR_RANGE))
            lidar_bp.set_attribute('points_per_second',str(self.lidar_points_per_seconds))
            lidar_bp.set_attribute('rotation_frequency',str(self.lidar_rotation_frequency))
            lidar_bp.set_attribute('upper_fov',str(LIDAR_2D_UPPER_FOW))
            lidar_bp.set_attribute('lower_fov',str(LIDAR_2D_LOWER_FOW))
            lidar_bp.set_attribute('horizontal_fov',str(LIDAR_2D_HORIZONTAL_FOV))
            x = LIDAR_2D_LOCATION_X
            z = LIDAR_2D_LOCATION_Z
        lidar_bp.set_attribute('atmosphere_attenuation_rate',str(LIDAR_ATTENUATION_RATE))
        lidar_bp.set_attribute('dropoff_general_rate',str(LIDAR_DROPOFF_RATE))
        lidar_bp.set_attribute('dropoff_intensity_limit',str(LIDAR_DROPOFF_INTENSITY_LIMIT))
        lidar_bp.set_attribute('dropoff_zero_intensity',str(LIDAR_DROPOFF_ZERO_INTENSITY))
        lidar_bp.set_attribute('noise_stddev',str(LIDAR_NOISE_STD))
        lidar_location = carla.Location(x,0,z)
        lidar_rotation = carla.Rotation(0,0,0)
        lidar_transform = carla.Transform(lidar_location,lidar_rotation)
        self.lidar = self.world.spawn_actor(lidar_bp,lidar_transform,attach_to=self.vehicle)

        self.max_intensity = math.e**(-LIDAR_ATTENUATION_RATE * self.lidar_range)
        self.lidar.listen(lambda raw_data: self._lidar_callback(raw_data))

        self.collision_sensor = self.world.spawn_actor(self.world.get_blueprint_library().find('sensor.other.collision'),
                                        carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._collision_callback(event))

        self.lane_sensor = self.world.spawn_actor(self.world.get_blueprint_library().find('sensor.other.lane_invasion'),
                                        carla.Transform(), attach_to=self.vehicle)
        self.lane_sensor.listen(lambda event: self._lane_invasion_callback(event))


    def add_collision_callback(self, callback):
        self.collision_callbacks.append(callback)

    def _collision_callback(self, event):
        for callback in self.collision_callbacks:
            callback(event)

    def add_lane_invasion_callback(self, callback):
        self.lane_callbacks.append(callback)

    def _lane_invasion_callback(self, event):
        for callback in self.lane_callbacks:
            callback(event)

    def add_lidar_callback(self, callback):
        self.custom_lidar_callbacks.append(callback)

    def _lidar_callback(self, data):
        data = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        self._not_filled_data = np.reshape(data, (int(data.shape[0] / 4), 4))
        if self.main_sensor == '2d-lidar':
            self._not_filled_data = self._remove_out_of_range(self._not_filled_data, self.lidar_range, self.max_intensity)
        self._lidar_data = self._fill_scan(self._not_filled_data, self.lidar_points_per_scan, 0)

        for callback in self.custom_lidar_callbacks:
            callback(self.get_lidar_data())

    def get_lidar_data(self):
        if self.main_sensor == '3d-lidar':
            return self._lidar_data
        if self.main_sensor == '2d-lidar':
            return self.lidar_distances(self._lidar_data)

    def get_car_linear_velocity(self):
        v = self.vehicle.get_velocity()
        return math.sqrt(v.x**2 + v.y**2 + v.z**2)

    def lidar_distances(self, lidar_data):
        distances = []
        for point in lidar_data:
            if point[3] > 0:
                distances.append(-math.log(point[3])/LIDAR_ATTENUATION_RATE)
            else:
                distances.append(0)
        return distances

    def destroy(self):
        self.lidar.destroy()
        if self.vis is not None:
            self.vis.destroy_window()


    def open_lidar_window(self):
        if open3d_installed is False:
            print('Unable to import open3d or matplotlib. Please install them if you want to visualize lidar data')
        else:
            self.VIRIDIS = np.array(cm.get_cmap('plasma').colors)
            self.VID_RANGE = np.linspace(0.0, 1.0, self.VIRIDIS.shape[0])

            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name='Carla Lidar',
                width=960,
                height=540,
                left=480,
                top=270)
            self.vis.get_render_option().background_color = [0.05, 0.05, 0.05]
            self.vis.get_render_option().point_size = 1
            self.vis.get_render_option().show_coordinate_frame = True

            self.point_list = o3d.geometry.PointCloud()
            self.first_run = True

    def update_lidar_window(self):
        data = self._not_filled_data.copy()

        if open3d_installed:
            if self.vis is None:
                print('You should call open_lidar_window before')
                return
            # Isolate the intensity and compute a color for it
            intensity = data[:, -1]
            intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-LIDAR_ATTENUATION_RATE * self.lidar_range))
            int_color = np.c_[
                np.interp(intensity_col, self.VID_RANGE, self.VIRIDIS[:, 0]),
                np.interp(intensity_col, self.VID_RANGE, self.VIRIDIS[:, 1]),
                np.interp(intensity_col, self.VID_RANGE, self.VIRIDIS[:, 2])]
            # Isolate the 3D data
            points = data[:, :-1]
            # We're negating the y to correclty visualize a world that matches
            # what we see in Unreal since Open3D uses a right-handed coordinate system
            points[:, :1] = -points[:, :1]

            self.point_list.points = o3d.utility.Vector3dVector(points)
            self.point_list.colors = o3d.utility.Vector3dVector(int_color)

            if self.first_run:
                self.vis.add_geometry(self.point_list)
                self.first_run = False
            else:
                self.vis.update_geometry(self.point_list)

            self.vis.poll_events()
            self.vis.update_renderer()
        else:
            print("lidar data {}".format(data))

    def _remove_out_of_range(self, lidar_data, range, default_value):
        filtered = []
        for point in lidar_data:
            distance = -math.log(point[3])/LIDAR_ATTENUATION_RATE if point[3] > 0 else 0
            if distance <= range:
                filtered.append(point)
            else:
                filtered.append(np.array([0,0,0,default_value]))
        return np.array(filtered)

    def _fill_scan(self, lidar_data, points_per_scan, default_value):
        n_missing = points_per_scan - lidar_data.shape[0]
        if n_missing > 0:
            left_pad = int(n_missing/2)
            right_pad = n_missing - left_pad
            lidar_data = np.pad(lidar_data, ((left_pad, right_pad), (0,0)), mode='constant', constant_values=0)
            if default_value != 0:
                lidar_data[:,3] = np.where(lidar_data[:,3] > 0, lidar_data[:,3], default_value)
        return lidar_data