import rospy
from ackermann_msgs.msg import AckermannDriveStamped

import time
import argparse

try:
    from car.sensors import Sensors
    from geometry_msgs.msg import PoseStamped
except ImportError:
    from sensors import Sensors

MAX_SPEED_REDUCTION = 6
STEERING_SPEED_REDUCTION = 4
BACKWARD_SPEED_REDUCTION = 15
LIGHTLY_STEERING_REDUCTION = 2.4
BACKWARD_SECONDS = 1.5

MAX_SPEED_REDUCTION_SIM = 3
STEERING_SPEED_REDUCTION_SIM = 3
USE_RESET_INSTEAD_OF_BACKWARDS_SIM = False

class Drive():
    def __init__(self, is_simulator=False):
        self.is_simulator = is_simulator
        if not is_simulator:
            topic = "/vesc/high_level/ackermann_cmd_mux/input/nav_0"
            max_steering = 0.34
            self.max_speed_reduction = MAX_SPEED_REDUCTION
            self.steering_max_speed_reduction = STEERING_SPEED_REDUCTION
        else:
            topic = "/drive"
            max_steering = 0.4189
            self.max_speed_reduction = MAX_SPEED_REDUCTION_SIM
            self.steering_max_speed_reduction = STEERING_SPEED_REDUCTION_SIM
            self.reset_publisher = rospy.Publisher("/pose", PoseStamped, queue_size=0)
        self.max_speed = rospy.get_param("max_speed", 5)
        self.max_steering = rospy.get_param("max_steering", max_steering)
        self.drive_publisher = rospy.Publisher(topic, AckermannDriveStamped, queue_size=0)
        self.sensors = Sensors(is_simulator)
        print("max_speed: ", self.max_speed, ", max_steering: ", self.max_steering)

    def forward(self):
        self.send_drive_command(self.max_speed/self.max_speed_reduction, 0)
    
    def backward(self):
        self.send_drive_command(-self.max_speed/self.max_speed_reduction, 0)
    
    def stop(self):
        self.send_drive_command(0, 0)
    
    def right(self):
        self.send_drive_command(self.max_speed/self.steering_max_speed_reduction, -self.max_steering)

    def left(self):
        self.send_drive_command(self.max_speed/self.steering_max_speed_reduction, self.max_steering)

    def lightly_right(self):
        self.send_drive_command(self.max_speed/self.steering_max_speed_reduction, -self.max_steering/LIGHTLY_STEERING_REDUCTION)

    def lightly_left(self):
        self.send_drive_command(self.max_speed/self.steering_max_speed_reduction, self.max_steering/LIGHTLY_STEERING_REDUCTION)

    def send_drive_command(self, speed, steering_angle):
        ack_msg = AckermannDriveStamped()
        ack_msg.drive.speed = speed
        ack_msg.drive.steering_angle = steering_angle
        self.drive_publisher.publish(ack_msg)

    def backward_until_obstacle(self):
        if self.is_simulator:
            if USE_RESET_INSTEAD_OF_BACKWARDS_SIM:
                self.reset_publisher.publish(PoseStamped())
            else:
                self.send_drive_command(-self.max_speed, 0)
                time.sleep(0.2)
                self.stop()
        else:
            start = time.time()
            while not self.sensors.back_obstacle() and time.time() - start < BACKWARD_SECONDS:
                self.send_drive_command(-self.max_speed/BACKWARD_SPEED_REDUCTION, 0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", action='store_true', help="to set the use of the simulator")
    args = parser.parse_args()

    run_seconds = 0.6
    rospy.init_node('drive_test')
    drive = Drive(args.simulator)
    while True:
        print("Write command")
        cmd = input()
        start = time.time()
        if cmd == "w":
            while time.time() - start < run_seconds:
                drive.forward()
        if cmd == "s":
            while time.time() - start < run_seconds:
                drive.backward()
        if cmd == "a":
            while time.time() - start < run_seconds:
                drive.lightly_left()
        if cmd == "d":
            while time.time() - start < run_seconds:
                drive.lightly_right()
        if cmd == "aa":
            while time.time() - start < run_seconds:
                drive.left()
        if cmd == "dd":
            while time.time() - start < run_seconds:
                drive.right()
        if cmd == " ":
            while time.time() - start < run_seconds:
                drive.stop()
        if cmd == "buo":
            drive.backward_until_obstacle()
        if cmd == "q":
            exit()            

