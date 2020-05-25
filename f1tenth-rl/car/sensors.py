import rospy
from sensor_msgs.msg import LaserScan
import time

class Sensors():
    def __init__(self):
        self.lidar_data = None
        rospy.Subscriber("scan", LaserScan, self.lidar_callback)

    def lidar_callback(self, lidar_data):
        self.lidar_data = lidar_data

if __name__ == '__main__':
    rospy.init_node('sensors_test')
    sensor = Sensors()
    time.sleep(2)
    while True:
        print(sensor.lidar_data)
        time.sleep(10)