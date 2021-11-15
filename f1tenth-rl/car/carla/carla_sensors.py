import math
import numpy as np
import carla

try:
    import open3d as o3d
    from matplotlib import cm
    open3d_installed = True
except ImportError:
    open3d_installed = False


# based on Velodyne VLP-16
LIDAR_CHANNELS = 16
LIDAR_RANGE = 100
LIDAR_POINTS_PER_SECONDS = 300000
LIDAR_UPPER_FOW = 15
LIDAR_LOWER_FOW = -15
LIDAR_HORIZONTAL_FOV = 360

LIDAR_2D_CHANNELS = 1
LIDAR_2D_RANGE = 30
LIDAR_2D_POINTS_PER_SECONDS = 1080
LIDAR_2D_UPPER_FOW = 0
LIDAR_2D_LOWER_FOW = 0
LIDAR_2D_HORIZONTAL_FOV = 270

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
        self.custom_lidar_callbacks = []
        self.lidar_data = None

        self.vis = None
        self.lidar_range = LIDAR_RANGE
        self.lidar_rotation_frequency = 1/world_delta if world_delta>0 else 0

        lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        if main_sensor == '3d-lidar':
            lidar_bp.set_attribute('channels',str(LIDAR_CHANNELS))
            lidar_bp.set_attribute('range',str(LIDAR_RANGE))
            lidar_bp.set_attribute('points_per_second',str(LIDAR_POINTS_PER_SECONDS))
            lidar_bp.set_attribute('rotation_frequency',str(self.lidar_rotation_frequency))
            lidar_bp.set_attribute('upper_fov',str(LIDAR_UPPER_FOW))
            lidar_bp.set_attribute('lower_fov',str(LIDAR_LOWER_FOW))
            lidar_bp.set_attribute('horizontal_fov',str(LIDAR_HORIZONTAL_FOV))
        elif main_sensor == '2d-lidar':
            lidar_bp.set_attribute('channels',str(LIDAR_2D_CHANNELS))
            lidar_bp.set_attribute('range',str(LIDAR_2D_RANGE))
            lidar_bp.set_attribute('points_per_second',str(LIDAR_2D_POINTS_PER_SECONDS))
            lidar_bp.set_attribute('rotation_frequency',str(self.lidar_rotation_frequency))
            lidar_bp.set_attribute('upper_fov',str(LIDAR_2D_UPPER_FOW))
            lidar_bp.set_attribute('lower_fov',str(LIDAR_2D_LOWER_FOW))
            lidar_bp.set_attribute('horizontal_fov',str(LIDAR_2D_HORIZONTAL_FOV))
        lidar_bp.set_attribute('atmosphere_attenuation_rate',str(LIDAR_ATTENUATION_RATE))
        lidar_bp.set_attribute('dropoff_general_rate',str(LIDAR_DROPOFF_RATE))
        lidar_bp.set_attribute('dropoff_intensity_limit',str(LIDAR_DROPOFF_INTENSITY_LIMIT))
        lidar_bp.set_attribute('dropoff_zero_intensity',str(LIDAR_DROPOFF_ZERO_INTENSITY))
        lidar_bp.set_attribute('noise_stddev',str(LIDAR_NOISE_STD))
        lidar_location = carla.Location(0.5,0,1.8)
        lidar_rotation = carla.Rotation(0,0,0)
        lidar_transform = carla.Transform(lidar_location,lidar_rotation)
        self.lidar = self.world.spawn_actor(lidar_bp,lidar_transform,attach_to=self.vehicle)

        self.lidar.listen(lambda raw_data: self.lidar_callback(raw_data))

    def add_lidar_callback(self, callback):
        self.custom_lidar_callbacks.append(callback)

    def lidar_callback(self, data):
        self.lidar_data = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        for callback in self.custom_lidar_callbacks:
            callback(self.lidar_data)

    def get_lidar_data(self):
        return self.lidar_data

    def get_car_linear_velocity(self):
        v = self.vehicle.get_velocity()
        return math.sqrt(v.x**2 + v.y**2 + v.z**2)

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
        data = self.lidar_data.copy()
        data = np.reshape(data, (int(data.shape[0] / 4), 4))

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