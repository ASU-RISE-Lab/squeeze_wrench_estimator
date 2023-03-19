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
from px4_msgs.msg import ControllerOut

from squeeze_custom_msgs.msg import ExternalWrenchEstimate
from MLPF import Median_LPF


from std_msgs.msg import Float64

class LP_Filter(Node):

    def __init__(self):
        super().__init__('Wrench_Tester')

        self.controller_fx_filter = Median_LPF()
        self.controller_fy_filter = Median_LPF()
        self.controller_fz_filter = Median_LPF()
        self.controller_taux_filter = Median_LPF()
        self.controller_tauy_filter = Median_LPF()
        self.controller_tauz_filter = Median_LPF()
        self.alpha_controller_out = np.array([0.8,0.8,0.4,0.8,0.8,0.8,0.8]) # MLPF Filter time constants for Controller_Out # fx = 0.8
        self.controller_out_filter_buffer = np.zeros((6,3)) # Fx, Fy, Fz, Taux, Tauy, Tauz = [prev, current, next]

        self.filt_controller_out_pub = self.create_publisher(ControllerOut, 'New/fmu/controller_out_filtered/out', 1)
        self.controller_out_sub = self.create_subscription(ControllerOut, '/fmu/controller_out/out', self.controller_out_callback, 1)

        # self.wrench_sub = self.create_subscription(ExternalWrenchEstimate, 'External_Wrench_Estimate_1', self.wrench_callback,10)
        self.max = -100.0

    def wrench_callback(self, msg):
        if (abs(msg.tau_q) > self.max):
            self.max = abs(msg.tau_q)
            print(self.max)

    def controller_out_callback(self, msg): # Controller Out Callback

        # print(msg.f[0], msg.f[1], msg.f[2], msg.tau[0], msg.tau[1], msg.tau[2])

        filt_controller_out = ControllerOut()

        self.controller_out_estimate = np.array([msg.f[0]*0, msg.f[1]*0, msg.f[2]*32.66, msg.tau[0]*1.36, msg.tau[1]*1.36, msg.tau[2]*0.012])

        self.controller_out_filter_buffer[0][:2] = self.controller_out_filter_buffer[0][1:]
        self.controller_out_filter_buffer[0][2] = msg.f[0]

        self.controller_out_filter_buffer[1][:2] = self.controller_out_filter_buffer[1][1:]
        self.controller_out_filter_buffer[1][2] = msg.f[1]

        self.controller_out_filter_buffer[2][:2] = self.controller_out_filter_buffer[2][1:]
        self.controller_out_filter_buffer[2][2] = msg.f[2]

        self.controller_out_filter_buffer[3][:2] = self.controller_out_filter_buffer[3][1:]
        self.controller_out_filter_buffer[3][2] = msg.tau[0]

        self.controller_out_filter_buffer[4][:2] = self.controller_out_filter_buffer[4][1:]
        self.controller_out_filter_buffer[4][2] = msg.tau[1]

        self.controller_out_filter_buffer[5][:2] = self.controller_out_filter_buffer[5][1:]
        self.controller_out_filter_buffer[5][2] = msg.tau[2]

        filt_controller_out.f[0] = self.controller_fx_filter.med_lp(self.controller_out_filter_buffer[0],self.alpha_controller_out[0])
        filt_controller_out.f[1] = self.controller_fy_filter.med_lp(self.controller_out_filter_buffer[1],self.alpha_controller_out[1])
        filt_controller_out.f[2] = self.controller_fz_filter.med_lp(self.controller_out_filter_buffer[2],self.alpha_controller_out[2])
        filt_controller_out.tau[0] = self.controller_taux_filter.med_lp(self.controller_out_filter_buffer[3],self.alpha_controller_out[3])
        filt_controller_out.tau[1] = self.controller_tauy_filter.med_lp(self.controller_out_filter_buffer[4],self.alpha_controller_out[4])
        filt_controller_out.tau[2] = self.controller_tauz_filter.med_lp(self.controller_out_filter_buffer[5],self.alpha_controller_out[5])

        self.f = np.array([filt_controller_out.f[0], filt_controller_out.f[1], filt_controller_out.f[2] * 28.0])
        self.tau = np.array([filt_controller_out.tau[0], filt_controller_out.tau[1], filt_controller_out.tau[2]])

        filt_controller_out.timestamp = msg.timestamp

        self.filt_controller_out_pub.publish(filt_controller_out)


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