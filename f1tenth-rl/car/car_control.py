import rospy
from ackermann_msgs.msg import AckermannDriveStamped

import time
import argparse

try:
    import Jetson.GPIO as GPIO
except ImportError:
    pass

MAX_SPEED_REDUCTION = 5
STEERING_SPEED_REDUCTION = 5
LIGHTLY_STEERING_REDUCTION = 2.4

LX_IR_SENSOR_PIN = 1
RX_IR_SENSOR_PIN = 0

class Drive():
    def __init__(self, is_simulator=False):
        self.is_simulator = is_simulator
        topic = "/drive"
        if not is_simulator:
            topic = "/vesc/high_level/ackermann_cmd_mux/input/nav_0"
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(LX_IR_SENSOR_PIN, GPIO.IN)
            GPIO.setup(RX_IR_SENSOR_PIN, GPIO.IN)
        self.max_speed = rospy.get_param("max_speed", 5)
        self.max_steering = rospy.get_param("max_steering", 0.34)
        print("max_speed: ", self.max_speed, ", max_steering: ", self.max_steering)
        self.drive_publisher = rospy.Publisher(topic, AckermannDriveStamped, queue_size=0)

    def forward(self):
        self.send_drive_command(self.max_speed/MAX_SPEED_REDUCTION, 0)
    
    def backward(self):
        self.send_drive_command(-self.max_speed/MAX_SPEED_REDUCTION, 0)
    
    def stop(self):
        self.send_drive_command(0, 0)
    
    def right(self):
        self.send_drive_command(self.max_speed/STEERING_SPEED_REDUCTION, -self.max_steering)

    def left(self):
        self.send_drive_command(self.max_speed/STEERING_SPEED_REDUCTION, self.max_steering)

    def lightly_right(self):
        self.send_drive_command(self.max_speed/STEERING_SPEED_REDUCTION, -self.max_steering/LIGHTLY_STEERING_REDUCTION)

    def lightly_left(self):
        self.send_drive_command(self.max_speed/STEERING_SPEED_REDUCTION, self.max_steering/LIGHTLY_STEERING_REDUCTION)

    def send_drive_command(self, speed, steering_angle):
        ack_msg = AckermannDriveStamped()
        ack_msg.drive.speed = speed
        ack_msg.drive.steering_angle = steering_angle
        self.drive_publisher.publish(ack_msg)

    def backward_until_obstacle(self):
        if self.is_simulator:
            self.backward()
            time.sleep(1)
            self.stop()
        else:
            while not self.GPIO.input(LX_IR_SENSOR_PIN) == self.GPIO.LOW
                    and not self.GPIO.input(RX_IR_SENSOR_PIN) == self.GPIO.LOW:
                self.backward()

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

