import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import time

class Sensors():
    def __init__(self):
        self.lidar_data = None
        rospy.Subscriber("scan", LaserScan, self.lidar_callback)
        rospy.Subscriber("odom", Odometry, self.odometry_callback)

    def lidar_callback(self, lidar_data):
        self.lidar_data = lidar_data

    def odometry_callback(self, odometry):
        self.odometry = odometry

    def get_lidar_ranges(self):
        return self.lidar_data.ranges

    def get_car_linear_acelleration(self):
        return self.odometry.twist.twist.linear.x
    def get_car_angular_acelleration(self):
        return self.odometry.twist.twist.angular.z

if __name__ == '__main__':
    rospy.init_node('sensors_test')
    sensor = Sensors()
    time.sleep(1)
    while True:
        print(sensor.lidar_data)
        print(sensor.odometry)
        time.sleep(5)