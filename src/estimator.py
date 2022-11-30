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


class Wrench_Estimator(Node):

    def __init__(self):
        super().__init__('Wrench_Estimator')

        self.imu1 = [0.0, 0.0, 0.0, 0.0] # [theta, dtheta, dtheta/dt, ddtheta/dtt]
        self.imu2 = [0.0, 0.0, 0.0, 0.0]
        self.imu3 = [0.0, 0.0, 0.0, 0.0]
        self.imu4 = [0.0, 0.0, 0.0, 0.0]
        self.timestamp = time.time()

        self.imu1_prev = [0.0,0.0,0.0,0.0]
        self.imu2_prev = [0.0,0.0,0.0,0.0]
        self.imu3_prev = [0.0,0.0,0.0,0.0]
        self.imu4_prev = [0.0,0.0,0.0,0.0]

        self.first_call_back = 0
        self.count = 0
        self.len = 7
        self.filtimu1 = self.filtimu2 = self.filtimu3 = self.filtimu4 = np.zeros(self.len)

        self.imu_angle_sub = self.create_subscription(ImuData, 'IMU_Data', self.imu_angle_callback, 1)
        self.force_pub = self.create_publisher(ImuData, 'Force_Est', 1)

        timer_period = 50 # Hz
        # self.timer = self.create_timer(1/timer_period, self.timer_callback)

    def imu_angle_callback(self, msg):

        # msg.imu1 = msg.imu1 - 135.0
        # msg.imu2 = 45.0 - msg.imu2
        # msg.imu3 = -45 - msg.imu3
        # msg.imu4 = -135.0 - msg.imu4

        msg.imu1 = (msg.imu1 - 135.0) * 3.14/180
        msg.imu2 = (msg.imu2 - 45.0) * 3.14/180
        msg.imu3 = (-45.0 - msg.imu3) * 3.14/180
        msg.imu4 = (-135.0 - msg.imu4) * 3.14/180

        if(msg.timestamp - self.timestamp != 0):

            self.dt = msg.timestamp - self.timestamp

            # self.imu1[3] = ((msg.imu1 - self.imu1[1])/self.dt - self.imu1[2]) / self.dt

            # self.imu1[2] = (msg.imu1 - self.imu1[1])/self.dt

            # self.imu1[1] = self.dthetha_estimate(msg.imu1)

            self.imu1[1] = self.dthetha_eff(msg.imu1)
            self.imu2[1] = self.dthetha_eff(msg.imu2)
            self.imu3[1] = self.dthetha_eff(msg.imu3)
            self.imu4[1] = self.dthetha_eff(msg.imu4)

            self.imu1[2] = (self.dthetha_eff(msg.imu1) - self.imu1_prev[1] ) / self.dt
            self.imu2[2] = (self.dthetha_eff(msg.imu2) - self.imu2_prev[1] ) / self.dt
            self.imu3[2] = (self.dthetha_eff(msg.imu3) - self.imu3_prev[1] ) / self.dt
            self.imu4[2] = (self.dthetha_eff(msg.imu4) - self.imu4_prev[1] ) / self.dt

            self.imu1[3] = (self.imu1[2] - self.imu1_prev[2]) / self.dt
            self.imu2[3] = (self.imu2[2] - self.imu2_prev[2]) / self.dt
            self.imu3[3] = (self.imu3[2] - self.imu3_prev[2]) / self.dt
            self.imu4[3] = (self.imu4[2] - self.imu4_prev[2]) / self.dt

            self.imu1_prev = self.imu1
            self.imu2_prev = self.imu2
            self.imu3_prev = self.imu3
            self.imu4_prev = self.imu4

            self.timestamp = msg.timestamp

        # print(self.imu1)
        self.timer_callback()

    def dthetha_eff(self, dtheta):

        # if (dtheta != 0):
        #     # dthetha_eff =  ( 4 / (1 + 0.15 * abs(dtheta))) * (dtheta - 0.085 * (dtheta / abs(dtheta)))**3
        #     dtheta_eff = (4.2 * dtheta)**3
        # else:
        #     dtheta_eff = 0

        if (abs(dtheta) < 0.26):
            #dtheta_eff = 0
            dtheta_eff = (2 * dtheta)**3 
        else:
            dtheta_eff = dtheta

        return dtheta_eff
        # return dtheta

    def timer_callback(self):

        msg = ImuData()
        
        msg.imu1_force = ( 1.3077 * self.imu1[1] + 0.01 * self.imu1[2] + 0.0015 * self.imu1[3] ) / 0.15
        msg.imu2_force = ( 1.3077 * self.imu2[1] + 0.01 * self.imu2[2] + 0.0015 * self.imu2[3] ) / 0.15
        msg.imu3_force = ( 1.3077 * self.imu3[1] + 0.01 * self.imu3[2] + 0.0015 * self.imu3[3] ) / 0.15
        msg.imu4_force = ( 1.3077 * self.imu4[1] + 0.01 * self.imu4[2] + 0.0015 * self.imu4[3] ) / 0.15
        
        print("Arm1 Force:",msg.imu1_force,"Angle:",self.imu1[1]*180/3.14)
        print("Arm2 Force:",msg.imu2_force,"Angle:",self.imu2[1]*180/3.14)
        print("Arm3 Force:",msg.imu3_force,"Angle:",self.imu3[1]*180/3.14)
        print("Arm4 Force:",msg.imu4_force,"Angle:",self.imu4[1]*180/3.14)
        self.force_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)

    wrench = Wrench_Estimator()

    rclpy.spin(wrench)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    Wrench_Estimator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
