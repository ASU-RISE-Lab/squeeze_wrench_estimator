import rclpy
import serial
from rclpy.node import Node
from std_msgs.msg import String
from bno055_raspi_squeeze.msg import ImuData
from px4_msgs.msg import VehicleMagnetometer
import numpy as np
import time
from scipy.spatial.transform import Rotation
import math
from scipy.signal import butter,filtfilt
from px4_msgs.msg import Timesync
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleOdometry
from px4_msgs.msg import SensorCombined

from std_msgs.msg import Float32

class LP_Filter(Node):

    def __init__(self):
        super().__init__('LP_Filter')

        self.acc_x_prev = 0.0

        self.imu_sub = self.create_subscription(SensorCombined ,"/fmu/sensor_combined/out",self.imu_callback,10)
        self.time_sub = self.create_subscription(Timesync ,"/fmu/timesync/out",self.timestamp_callback,10)

        self.pub = self.create_publisher(Float32, 'LP_Filter', 1)

        timer_period = 50 # Hz
        # self.timer = self.create_timer(1/timer_period, self.timer_callback)

    def timestamp_callback(self,msg):
        self.timestamp = msg.timestamp

    def imu_callback(self,msg):
        self.acc_x = msg.accelerometer_m_s2[0]
        self.acc_y = msg.accelerometer_m_s2[1]
        self.acc_z = msg.accelerometer_m_s2[2]
        self.acc_norm = np.sqrt((self.acc_x**2)+(self.acc_y**2)+(self.acc_z**2))

        self.acc_x_new = (self.acc_x_prev * 0.9) + (self.acc_x * 0.1)
        self.acc_x_prev = self.acc_x_new

        msg = Float32()
        msg.data = self.acc_x_new
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    wrench = LP_Filter()

    rclpy.spin(wrench)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    LP_Filter.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()