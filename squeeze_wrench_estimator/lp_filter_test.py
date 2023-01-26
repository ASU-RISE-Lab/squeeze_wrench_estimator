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

from std_msgs.msg import Float64

class LP_Filter(Node):

    def __init__(self):
        super().__init__('LP_Filter')

        # self.acc_x_filt_prev = 0.0

        # self.imu_sub = self.create_subscription(SensorCombined ,"/fmu/sensor_combined/out",self.imu_callback,10)
        # self.time_sub = self.create_subscription(Timesync ,"/fmu/timesync/out",self.timestamp_callback,10)

        self.vehicle_odometry_sub = self.create_subscription(VehicleOdometry, '/fmu/vehicle_odometry/out', self.vehicle_odometry_callback, 1)

        self.omega_pub = self.create_publisher(Float64, 'Omega', 1)
        self.omega_dot_pub = self.create_publisher(Float64, 'Omega_dot', 1)


        timer_period = 50 # Hz
        # self.timer = self.create_timer(1/timer_period, self.timer_callback)

        self.v = np.array([0.0,0.0,0.0])
        self.v_prev = np.array([0.0,0.0,0.0])
        self.v_dot = np.array([0.0,0.0,0.0])
        self.v_dot_prev = np.array([0.0,0.0,0.0])

        self.omega = np.array([0.0,0.0,0.0])
        self.omega_prev = np.array([0.0,0.0,0.0])
        self.omega_dot = np.array([0.0,0.0,0.0])
        self.omega_dot_prev = np.array([0.0,0.0,0.0])

        self.v_timestamp = 0.0

        self.gamma = 0.1

        print("LP Filter Node Initialized")


    def vehicle_odometry_callback(self, msg):

        self.v = np.array([msg.vx, msg.vy, msg.vz])
        omega = np.array([msg.rollspeed, msg.pitchspeed, msg.yawspeed])
        self.v_dt = (msg.timestamp - self.v_timestamp) * 10**-6

        omega = (omega * self.gamma) + (self.omega_prev * (1 - self.gamma))

        self.v_timestamp = msg.timestamp
        self.v_dot = (self.v - self.v_prev) / self.v_dt
        omega_dot = (omega - self.omega_prev) / self.v_dt


        self.v_dot = (self.v_dot * self.gamma) + (self.v_dot_prev * (1 - self.gamma))
        omega_dot = (omega_dot * self.gamma) + (self.omega_dot_prev * (1 - self.gamma))


        self.v_prev = self.v
        self.omega_prev = omega
        self.v_dot_prev = self.v_dot
        self.omega_dot_prev = omega_dot
        self.v_timestamp = msg.timestamp

        omega_pub = Float64()
        omega_dot_pub = Float64()


        omega_pub.data = float(omega[1])
        omega_dot_pub.data  = float(omega_dot[1])

        self.omega_pub.publish(omega_pub)
        self.omega_dot_pub.publish(omega_dot_pub)
        
        q = np.zeros(4)

        q[3] = msg.q[0]
        q[0] = msg.q[1]
        q[1] = msg.q[2]
        q[2] = msg.q[3]

        rot = Rotation.from_quat(q)
        rot_euler = rot.as_euler('xyz', degrees=True)  # [roll, pitch, yaw]

    # def timestamp_callback(self,msg):
    #     self.timestamp = msg.timestamp

    # def imu_callback(self,msg):
    #     self.acc_x = msg.accelerometer_m_s2[0]
    #     self.acc_y = msg.accelerometer_m_s2[1]
    #     self.acc_z = msg.accelerometer_m_s2[2]
    #     self.acc_norm = np.sqrt((self.acc_x**2)+(self.acc_y**2)+(self.acc_z**2))

    #     self.acc_x_filt_new = (self.acc_x_filt_prev * 0.9) + (self.acc_x * 0.1)
    #     self.acc_x__filt_prev = self.acc_x_filt_new

    #     msg = Float32()
    #     msg.data = self.acc_x_new
    #     self.pub.publish(msg)

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