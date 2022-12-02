import rclpy
import serial
from rclpy.node import Node
from std_msgs.msg import String
from bno055_raspi_squeeze.msg import ImuData
from squeeze_wrench_estimator.msg import ExternalWrenchEstimate
from px4_msgs.msg import VehicleMagnetometer
import numpy as np
import time
from scipy.spatial.transform import Rotation
import math
from scipy.signal import butter,filtfilt

from px4_msgs.msg import VehicleAttitude
from scipy.spatial.transform import Rotation as R

class Wrench_Estimator(Node):

    def __init__(self):
        super().__init__('Wrench_Estimator')

        self.rotation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

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

        self.alpha = np.zeros(4)

        self.imu_angle_sub = self.create_subscription(ImuData, 'IMU_Data', self.imu_angle_callback, 1)

        self.att_sub = self.create_subscription(VehicleAttitude, '/fmu/vehicle_attitude/out', self.attitude_callback, 1)

        self.wrench_pub = self.create_publisher(ExternalWrenchEstimate, 'External_Wrench_Estimate', 1)

        timer_period = 50 # Hz
        # self.timer = self.create_timer(1/timer_period, self.timer_callback)

    def attitude_callback(self, msg):

        self.rotation_matrix = R.from_quat([msg.q[1],msg.q[2],msg.q[3],msg.q[0]]).as_matrix() # Input in x,y,z,w

        # print(msg.q[0],msg.q[1],msg.q[2],msg.q[3])
        # print(self.rotation_matrix)

    def imu_angle_callback(self, msg):

        # msg.imu1 = msg.imu1 - 135.0
        # msg.imu2 = 45.0 - msg.imu2
        # msg.imu3 = -45 - msg.imu3
        # msg.imu4 = -135.0 - msg.imu4

        self.theta = [msg.imu1, msg.imu2, msg.imu3, msg.imu4] # Actual arm angles from node

        msg.imu1 = -(msg.imu1 - 135.0) * 3.14/180
        msg.imu2 = (msg.imu2 - 45.0) * 3.14/180
        msg.imu3 = -(-45.0 - msg.imu3) * 3.14/180
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

        msg = ExternalWrenchEstimate()

        self.arm_length = 0.15
        
        self.f_hat_imu1 = ( 1.3077 * self.imu1[1] + 0.01 * self.imu1[2] + 0.0015 * self.imu1[3] ) / self.arm_length # Arm Distance = 0.15m
        self.f_hat_imu2 = ( 1.3077 * self.imu2[1] + 0.01 * self.imu2[2] + 0.0015 * self.imu2[3] ) / self.arm_length
        self.f_hat_imu3 = ( 1.3077 * self.imu3[1] + 0.01 * self.imu3[2] + 0.0015 * self.imu3[3] ) / self.arm_length
        self.f_hat_imu4 = ( 1.3077 * self.imu4[1] + 0.01 * self.imu4[2] + 0.0015 * self.imu4[3] ) / self.arm_length

        self.e1 = np.array([1,0,0])
        self.e2 = np.array([0,1,0])

        self.b1 = np.matmul(self.rotation_matrix,self.e1)
        self.b2 = np.matmul(self.rotation_matrix,self.e2)

        self.alpha[0] = (-90 + self.theta[0]) * 3.14/180
        self.alpha[1] = (self.theta[1] - 90) * 3.14/180
        self.alpha[2] = (90 + self.theta[2]) * 3.14/180
        self.alpha[3] = (90 + self.theta[3])* 3.14/180

        self.rt_matrix_alpha1 = self.rt_matrix(self.alpha[0])
        self.rt_matrix_alpha2 = self.rt_matrix(self.alpha[1])
        self.rt_matrix_alpha3 = self.rt_matrix(self.alpha[2])
        self.rt_matrix_alpha4 = self.rt_matrix(self.alpha[3])

        self.f_hat_imu1_b = np.matmul(self.rt_matrix_alpha1,np.array([0,self.f_hat_imu1,0]))
        self.f_hat_imu2_b = np.matmul(self.rt_matrix_alpha2,np.array([0,self.f_hat_imu2,0]))
        self.f_hat_imu3_b = np.matmul(self.rt_matrix_alpha3,np.array([0,self.f_hat_imu3,0]))
        self.f_hat_imu4_b = np.matmul(self.rt_matrix_alpha4,np.array([0,self.f_hat_imu4,0]))
        
        self.f_hat_imu1_b_w = np.matmul(self.rotation_matrix, self.f_hat_imu1_b)
        self.f_hat_imu2_b_w = np.matmul(self.rotation_matrix, self.f_hat_imu2_b)
        self.f_hat_imu3_b_w = np.matmul(self.rotation_matrix, self.f_hat_imu3_b)
        self.f_hat_imu4_b_w = np.matmul(self.rotation_matrix, self.f_hat_imu4_b)

        self.f_hat_b_w =  self.f_hat_imu1_b_w + self.f_hat_imu2_b_w + self.f_hat_imu3_b_w + self.f_hat_imu4_b_w

        msg.f_x = float(round(self.f_hat_b_w[0],4))
        msg.f_y = float(round(self.f_hat_b_w[1],4))
        msg.f_z = float(round(self.f_hat_b_w[2],4))

        self.wrench_pub.publish(msg)

    def rt_matrix(self, theta):

        return np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

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
