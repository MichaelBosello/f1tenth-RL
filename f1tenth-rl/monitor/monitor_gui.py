#!/usr/bin/env python3

from threading import Thread
import math
import socket
import pickle
import time

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, QCoreApplication, QTimer

import rospy
from sensor_msgs.msg import LaserScan
from rospy.exceptions import ROSException

from rviz import bindings as rviz
from gui_components.AnalogGaugeWidgetPyQt.analoggaugewidget import AnalogGaugeWidget
from gui_components.qt_compass import CompassWidget

UPDATE_FREQ = 1000 / 60
UDP_IP = "192.168.1.28"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

class LidarRviz(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.frame = rviz.VisualizationFrame()
        self.frame.setSplashPath( "" )
        self.frame.initialize()
        reader = rviz.YamlConfigReader()
        config = rviz.Config()
        reader.readFile( config, "monitor_config.rviz" )
        self.frame.load( config )
        self.frame.setMenuBar( None )
        self.frame.setStatusBar( None )
        self.frame.setHideButtonVisibility( False )
        self.manager = self.frame.getManager()
        layout = QVBoxLayout()
        layout.addWidget( self.frame )
        self.setLayout( layout )

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("F1tenth Monitor")

        lidarWidget = LidarRviz()

        speed_label = QLabel('Speed (km/h)')
        speed_label.setAlignment(Qt.AlignCenter)
        speed_label.setFont(QFont('Arial', 17))
        acc_label = QLabel('Acceleration (m/s^2)')
        acc_label.setAlignment(Qt.AlignCenter)
        acc_label.setFont(QFont('Arial', 14))
        action_label = QLabel('Selected Action')
        action_label.setAlignment(Qt.AlignCenter)
        action_label.setFont(QFont('Arial', 17))
        compass_label = QLabel('Orientation')
        compass_label.setAlignment(Qt.AlignCenter)
        compass_label.setFont(QFont('Arial', 14))

        speed_gauge = AnalogGaugeWidget()
        speed_gauge.set_MaxValue(30)
        speed_gauge.setFixedSize(300, 300)

        acc_gauge = AnalogGaugeWidget()
        acc_gauge.set_MaxValue(30)
        acc_gauge.set_enable_ScaleText(False)
        acc_gauge.set_gauge_color_inner_radius_factor(800)
        acc_gauge.set_enable_CenterPoint(False)
        acc_gauge.set_enable_big_scaled_grid(False)
        acc_gauge.setFixedSize(150, 150)

        action_image_label = QLabel()
        self.no_action_pixmap = QPixmap('gui_components/img/f110.png').scaled(300, 300)
        self.lx_pixmap = QPixmap('gui_components/img/f110-left.png').scaled(300, 300)
        self.rx_pixmap = QPixmap('gui_components/img/f110-right.png').scaled(300, 300)
        self.fw_pixmap = QPixmap('gui_components/img/f110-forward.png').scaled(300, 300)
        self.slow_pixmap = QPixmap('gui_components/img/f110-stop.png').scaled(300, 300)
        action_image_label.setPixmap(self.no_action_pixmap)
        action_image_label.setFixedSize(300, 300)

        auto_label = QLabel()
        auto_label.setText('Autonomous Mode: <font color="green"><b>On</b></font>')
        auto_label.setAlignment(Qt.AlignCenter)
        auto_label.setFont(QFont('Arial', 17))

        compass = CompassWidget()
        compass.setFixedSize(150, 150)

        layout = QHBoxLayout()
        layout_sx = QVBoxLayout()
        layout_dx = QVBoxLayout()
        gauge_layout = QHBoxLayout()
        compass_layout = QHBoxLayout()
        layout_sx.setContentsMargins(25, 0, 25, 25)
        layout_dx.setContentsMargins(25, 0, 25, 25)
        gauge_layout.setContentsMargins(75, 0, 75, 0)
        compass_layout.setContentsMargins(75, 0, 75, 0)

        layout_sx.addStretch()
        layout_sx.addWidget(auto_label)
        layout_sx.addSpacing(15)
        layout_sx.addStretch()
        layout_sx.addWidget(action_label)
        layout_sx.addSpacing(15)
        layout_sx.addWidget(action_image_label)
        layout_sx.addStretch()
        layout_sx.addWidget(compass_label)
        layout_sx.addSpacing(15)
        compass_layout.addWidget(compass)
        layout_sx.addLayout(compass_layout)
        layout_sx.addStretch()

        layout_dx.addStretch()
        layout_dx.addWidget(speed_label)
        layout_dx.addSpacing(15)
        layout_dx.addWidget(speed_gauge)
        layout_dx.addStretch()
        layout_dx.addWidget(acc_label)
        layout_dx.addSpacing(15)
        gauge_layout.addWidget(acc_gauge)
        layout_dx.addLayout(gauge_layout)
        layout_dx.addStretch()

        layout.addLayout( layout_sx )
        layout.addWidget(lidarWidget)
        layout.addLayout( layout_dx )
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.speed_gauge = speed_gauge
        self.acc_gauge = acc_gauge
        self.compass = compass
        self.action_image_label = action_image_label
        self.auto_label = auto_label

class MonitorDataHandler():
    def __init__(self, window):
        self.window = window
        self.speed = 0
        self.acc = 0
        self.orientation = 0
        self.action = -1
        self.autonomous_mode = True
        rospy.init_node('monitor')
        self.laser_publisher = rospy.Publisher("/scan", LaserScan, queue_size=0)
        process = Thread(target=self.data_receiver)
        process.daemon = True
        process.start()

    def data_receiver(self):
        while True:
            data_msg, addr = sock.recvfrom(32768)
            data = pickle.loads(data_msg)

            try:
                self.laser_publisher.publish(data["ldr"])
            except ROSException as e:
                if str(e) == "publish() to a closed topic":
                    raise e
                else:
                    raise e

            speed = abs(data["spd"]) * 3.6 * 1.6
            if speed > self.window.speed_gauge.get_value_max():
                print("Speed out of range: {}".format(speed))
            else:
                self.speed = speed
            if data["acc"] > self.window.acc_gauge.get_value_max():
                print("Acceleration out of range: {}".format(data["acc"]))
            else:
                self.acc = data["acc"]
            self.orientation = data["orn"] * 180 / math.pi
            
            self.action = data["act"]
            self.autonomous_mode = data["auto"]


    def update_gui(self):
        window.speed_gauge.update_value(self.speed)
        window.acc_gauge.update_value(self.acc)
        window.compass.setAngle(self.orientation)
        if self.action == -1:
            window.action_image_label.setPixmap(window.no_action_pixmap)
        elif self.action == 0:
            window.action_image_label.setPixmap(window.fw_pixmap)
        elif self.action == 1:
            window.action_image_label.setPixmap(window.rx_pixmap)
        elif self.action == 2:
            window.action_image_label.setPixmap(window.lx_pixmap)
        elif self.action == 3:
            window.action_image_label.setPixmap(window.slow_pixmap)

        if self.autonomous_mode:
            window.auto_label.setText('Autonomous Mode: <font color="green"><b>On</b></font>')
        else:
            window.auto_label.setText('Autonomous Mode: <font color="red"><b>Off</b></font>')

        QCoreApplication.processEvents()

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.resize(1920, 1080)
    window.showFullScreen()
    window.show()

    handler = MonitorDataHandler(window)
    
    timer_gui = QTimer()
    timer_gui.timeout.connect(handler.update_gui)
    timer_gui.setInterval(UPDATE_FREQ)
    timer_gui.start()

    app.exec_()