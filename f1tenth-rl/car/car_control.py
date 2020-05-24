import rospy
import time
from ackermann_msgs.msg import AckermannDriveStamped

MAX_SPEED_REDUCTION = 2
STEERING_SPEED_REDUCTION = 3
LIGHTLY_STEERING_REDUCTION = 2

class Drive():
  def __init__(self):
    self.max_speed = rospy.get_param("max_speed", 2)
    self.max_steering = rospy.get_param("max_steering", 0.4189)
    print("max_speed: ", self.max_speed, ", max_steering: ", self.max_steering)
    self.drive_publisher = rospy.Publisher('/drive', AckermannDriveStamped, queue_size=0)

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

if __name__ == '__main__':
    rospy.init_node('drive_test')
    drive = Drive()
    while True:
        print("Write command")
        cmd = input()
        if cmd == "w":
            drive.forward()
        if cmd == "s":
            drive.backward()
        if cmd == "a":
            drive.lightly_left()
        if cmd == "d":
            drive.lightly_right()
        if cmd == "aa":
            drive.left()
        if cmd == "dd":
            drive.right()
            
        time.sleep(1)
        drive.stop()