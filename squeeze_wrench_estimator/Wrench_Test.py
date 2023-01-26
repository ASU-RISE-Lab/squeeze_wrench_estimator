import rclpy
import serial
from rclpy.node import Node
from std_msgs.msg import String
# from bno055_raspi_squeeze.msg import ImuData
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

from squeeze_custom_msgs.msg import ExternalWrenchEstimate

from std_msgs.msg import Float64

class LP_Filter(Node):

    def __init__(self):
        super().__init__('Wrench_Tester')

        self.wrench_sub = self.create_subscription(ExternalWrenchEstimate, 'External_Wrench_Estimate_1', self.wrench_callback,10)
        self.max = -100.0

    def wrench_callback(self, msg):
        if (abs(msg.tau_q) > self.max):
            self.max = abs(msg.tau_q)
            print(self.max)


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