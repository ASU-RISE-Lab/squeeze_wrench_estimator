import rclpy
import serial
from rclpy.node import Node
from std_msgs.msg import String
# from bno055_raspi_squeeze.msg import ImuData
# from squeeze_wrench_estimator.msg import ExternalWrenchEstimate
from squeeze_custom_msgs.msg import ImuData
from squeeze_custom_msgs.msg import ExternalWrenchEstimate
from px4_msgs.msg import VehicleMagnetometer
import numpy as np
import time
from scipy.spatial.transform import Rotation
import math
from scipy.signal import butter,filtfilt

from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float32

from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import SensorCombined
from px4_msgs.msg import ControllerOut
from px4_msgs.msg import VehicleOdometry

from scipy.spatial.transform import Rotation as R

class Wrench_Estimator(Node):

    def __init__(self):
        super().__init__('Wrench_Estimator')

        print("Starting Wrench Estimator Node")

        self.rotation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # self.imu1 = [0.0, 0.0, 0.0] # [dtheta, dtheta/dt, ddtheta/dtt]
        # self.imu2 = [0.0, 0.0, 0.0]
        # self.imu3 = [0.0, 0.0, 0.0]
        # self.imu4 = [0.0, 0.0, 0.0]
        self.timestamp = time.time()

        self.avg_dt = 0.0

        self.imu1_prev = [0.0,0.0,0.0]
        self.imu2_prev = [0.0,0.0,0.0]
        self.imu3_prev = [0.0,0.0,0.0]
        self.imu4_prev = [0.0,0.0,0.0]

        self.f = np.array([0.0,0.0,0.0])
        self.tau = np.array([0.0,0.0,0.0])

        self.v = np.array([0.0,0.0,0.0])
        self.v_prev = np.array([0.0,0.0,0.0])
        self.v_dot = np.array([0.0,0.0,0.0])
        self.v_dot_prev = np.array([0.0,0.0,0.0])

        self.omega = np.array([0.0,0.0,0.0])
        self.omega_prev = np.array([0.0,0.0,0.0])
        self.omega_dot = np.array([0.0,0.0,0.0])
        self.omega_dot_prev = np.array([0.0,0.0,0.0])

        self.gamma = 0.05 # LP filter for v_dot and omega_dot

        self.inertia_mat = ([0.011160,0.000003,0.000058],[0.000003, 0.011260,0.000044],[0.000058,0.000044,0.018540]) # kg.m^2
        self.mass = 1.25 # kg
        self.g = 9.81 # m/s^2

        self.controller_out_prev = np.array([0.0,0.0,0.0,0.0,0.0,0.0]) # fx, fy, fz, taux, tauy, tauz
        self.alpha_controller_out = np.array([0.01,0.01,0.05,0.05,0.02,0.5])

        self.v_timestamp = 0.0

        self.e1 = np.array([1,0,0])
        self.e2 = np.array([0,1,0])
        self.e3 = np.array([0,0,1])

        self.first_call_back = 0
        self.count = 0
        self.len = 7
        self.filtimu1 = self.filtimu2 = self.filtimu3 = self.filtimu4 = np.zeros(self.len)

        self.alpha = np.zeros(4)

        # Subscribers
        self.imu_angle_sub = self.create_subscription(ImuData, 'IMU_Data_Fil', self.imu_angle_callback, 1)
        self.att_sub = self.create_subscription(VehicleAttitude, '/fmu/vehicle_attitude/out', self.attitude_callback, 1)
        self.sensor_combined_sub = self.create_subscription(SensorCombined, '/fmu/sensor_combined/out', self.sensor_combined_callback, 1)
        self.vehicle_odometry_sub = self.create_subscription(VehicleOdometry, '/fmu/vehicle_odometry/out', self.vehicle_odometry_callback, 1)
        self.controller_out_sub = self.create_subscription(ControllerOut, '/fmu/controller_out/out', self.controller_out_callback, 1)



        # Publishers
        self.wrench_pub = self.create_publisher(ExternalWrenchEstimate, 'External_Wrench_Estimate_1', 1)

        self.yaw_euler_pub = self.create_publisher(Float32, 'Yaw_Euler', 1)
        
        self.filt_controller_out_pub = self.create_publisher(ControllerOut, '/fmu/controller_out_filtered/out', 1)
        self.omega_pub = self.create_publisher(Float32MultiArray, 'Omega', 1)
        self.omega_dot_pub = self.create_publisher(Float32MultiArray, 'Omega_Dot', 1)

        self.pub_x = self.create_publisher(Float32MultiArray, 'wrench_estimator_body/term1', 1)
        self.pub_y = self.create_publisher(Float32MultiArray, 'wrench_estimator_body/term2', 1)
        self.pub_z = self.create_publisher(Float32MultiArray, 'wrench_estimator_body/term3', 1)
        self.total = self.create_publisher(Float32MultiArray, 'wrench_estimator_body/total', 1)
        self.max = -100.0

        # timer_period = 50 # Hz
        # self.timer = self.create_timer(1/timer_period, self.wrench_estimator)
        # 

    def lp_filter(self,current, previous, alpha):
        return alpha*current + (1-alpha)*previous

    def controller_out_callback(self, msg):

        # print(msg.f[0], msg.f[1], msg.f[2], msg.tau[0], msg.tau[1], msg.tau[2])

        filt_controller_out = ControllerOut()

        self.controller_out_estimate = np.array([msg.f[0]*0, msg.f[1]*0, msg.f[2]*32.66, msg.tau[0]*1.36, msg.tau[1]*1.36, msg.tau[2]*0.012])
        self.controller_out_lpf  = self.lp_filter(self.controller_out_estimate, self.controller_out_prev, self.alpha_controller_out)
        self.controller_out_prev = self.controller_out_lpf

        self.f = self.controller_out_lpf[:3]
        self.tau = self.controller_out_lpf[3:]

        filt_controller_out.f[0] = self.controller_out_lpf[0]
        filt_controller_out.f[1] = self.controller_out_lpf[1]
        filt_controller_out.f[2] = self.controller_out_lpf[2]
        filt_controller_out.tau[0] = self.controller_out_lpf[3]
        filt_controller_out.tau[1] = self.controller_out_lpf[4]
        filt_controller_out.tau[2] = self.controller_out_lpf[5]

        filt_controller_out.timestamp = msg.timestamp

        self.filt_controller_out_pub.publish(filt_controller_out)

        # print(self.f_hat_b_w,self.tau_hat_arm)
        # self.wrench_estimator_body()

    def vehicle_odometry_callback(self, msg):

        self.v = np.array([msg.vx, msg.vy, msg.vz])
        self.omega = np.array([msg.rollspeed, msg.pitchspeed, msg.yawspeed])
        self.v_dt = (msg.timestamp - self.v_timestamp) * 10**-6

        self.omega = (self.omega * 0.1) + (self.omega_prev * (1 - 0.1))

        self.v_timestamp = msg.timestamp
        self.v_dot = (self.v - self.v_prev) / self.v_dt
        self.omega_dot = (self.omega - self.omega_prev) / self.v_dt

        self.v_dot = (self.v_dot * self.gamma) + (self.v_dot_prev * (1 - self.gamma))
        self.omega_dot = (self.omega_dot * self.gamma) + (self.omega_dot_prev * (1 - self.gamma))

        self.v_prev = self.v
        self.omega_prev = self.omega
        self.v_dot_prev = self.v_dot
        self.omega_dot_prev = self.omega_dot
        self.v_timestamp = msg.timestamp

        omega_pub = Float32MultiArray()
        omega_dot_pub = Float32MultiArray()

        omega_pub.data = [float(self.omega[0]), float(self.omega[1]), float(self.omega[2])]
        omega_dot_pub.data = [float(self.omega_dot[0]), float(self.omega_dot[1]), float(self.omega_dot[2])]

        self.omega_pub.publish(omega_pub)
        self.omega_dot_pub.publish(omega_dot_pub)
        
        q = np.zeros(4)

        q[3] = msg.q[0]
        q[0] = msg.q[1]
        q[1] = msg.q[2]
        q[2] = msg.q[3]

        rot = Rotation.from_quat(q)
        rot_euler = rot.as_euler('xyz', degrees=True)  # [roll, pitch, yaw]

        yaw = Float32()
        yaw.data = rot_euler[2]
        self.yaw_euler_pub.publish(yaw)

    def sensor_combined_callback(self, msg):

        self.acc_x = msg.accelerometer_m_s2[0]
        self.acc_y = msg.accelerometer_m_s2[1]
        self.acc_z = msg.accelerometer_m_s2[2]

    def attitude_callback(self, msg):

        self.rotation_matrix = R.from_quat([msg.q[1],msg.q[2],msg.q[3],msg.q[0]]).as_matrix() # Input in x,y,z,w

        # print(msg.q[0],msg.q[1],msg.q[2],msg.q[3])
        # print(self.rotation_matrix)

    def imu_angle_callback(self, msg):

        imu1 = [0.0, 0.0, 0.0] # [dtheta, dtheta/dt, ddtheta/dtt]
        imu2 = [0.0, 0.0, 0.0]
        imu3 = [0.0, 0.0, 0.0]
        imu4 = [0.0, 0.0, 0.0]

        # self.imu_test_pub.publish(msg)

        # msg.imu1 = msg.imu1 - 135.0
        # msg.imu2 = 45.0 - msg.imu2
        # msg.imu3 = -45 - msg.imu3
        # msg.imu4 = -135.0 - msg.imu4

        self.theta = [msg.imu1, msg.imu2, msg.imu3, msg.imu4] # Actual arm angles from node

        msg.imu1 = -(msg.imu1 - 130.0) * 3.14/180
        msg.imu2 = (46.0 - msg.imu2) * 3.14/180
        msg.imu3 = (47.0 + msg.imu3) * 3.14/180
        msg.imu4 = (130.0 + msg.imu4) * 3.14/180

        # self.eff_imu1 = self.dthetha_eff(msg.imu1)
        # self.eff_imu2 = self.dthetha_eff(msg.imu2)
        # self.eff_imu3 = self.dthetha_eff(msg.imu3)
        # self.eff_imu4 = self.dthetha_eff(msg.imu4)

        if(msg.timestamp - self.timestamp != 0):

            dt = (msg.timestamp - self.timestamp) / 10**6  # TimeStamp in Microseconds
            # self.dt = 0.05

            imu1[0] = self.dthetha_eff(msg.imu1)
            imu2[0] = self.dthetha_eff(msg.imu2)
            imu3[0] = self.dthetha_eff(msg.imu3)
            imu4[0] = self.dthetha_eff(msg.imu4)

            imu1[1] = (imu1[0] - self.imu1_prev[0])/ dt
            imu2[1] = (imu2[0] - self.imu2_prev[0])/ dt
            imu3[1] = (imu3[0] - self.imu3_prev[0])/ dt
            imu4[1] = (imu4[0] - self.imu4_prev[0])/ dt

            imu1[2] = (imu1[1] - self.imu1_prev[1]) / dt
            imu2[2] = (imu2[1] - self.imu2_prev[1]) / dt
            imu3[2] = (imu3[1] - self.imu3_prev[1]) / dt
            imu4[2] = (imu4[1] - self.imu4_prev[1]) / dt
            
            self.timestamp = msg.timestamp

            self.wrench_estimator_body()
            self.wrench_estimator_arm(imu1,imu2,imu3,imu4)

            self.imu1_prev = imu1
            self.imu2_prev = imu2
            self.imu3_prev = imu3
            self.imu4_prev = imu4

        else :
            print("Timestamp Error")


    def dthetha_eff(self, dtheta): # Curve Fitting
                                   # If Deflection is less than 15 Degrees (0.26 Rads) we use (2*dtheta^3) else we use same dtheta
 
        # if (dtheta != 0):
        #     # dthetha_eff =  ( 4 / (1 + 0.15 * abs(dtheta))) * (dtheta - 0.085 * (dtheta / abs(dtheta)))**3
        #     dtheta_eff = (4.2 * dtheta)**3
        # else:
        #     dtheta_eff = 0

        if (abs(dtheta) < 0.26):
            dtheta_eff = (2 * dtheta)**3 
        else:
            dtheta_eff = dtheta

        return dtheta_eff
        # return dtheta

    def wrench_estimator_body(self):

        self.f_hat_body = (self.mass * self.v_dot) - (self.mass * self.g * self.e3) + self.f

        self.tau_hat_body = (np.matmul(self.inertia_mat , (self.omega_dot))) + (np.cross((self.omega), np.matmul(self.inertia_mat , self.omega))) - (np.transpose(self.tau))
        
        # x = (np.matmul(self.inertia_mat , (self.omega_dot)))
        # y = (np.cross((self.omega), np.matmul(self.inertia_mat , self.omega)))
        # z = (np.transpose(self.tau))

        # x_pub = Float32MultiArray()
        # y_pub = Float32MultiArray()
        # z_pub = Float32MultiArray()
        # total = Float32MultiArray()

        # x_pub.data = [float(x[0]), float(x[1]), float(x[2])]
        # y_pub.data = [float(y[0]), float(y[1]), float(y[2])]
        # z_pub.data = [float(z[0]), float(z[1]), float(z[2])]
        # total.data = [float(self.tau_hat_body[0]), float(self.tau_hat_body[1]), float(self.tau_hat_body[2])]

        # self.pub_x.publish(x_pub)
        # self.pub_y.publish(y_pub)
        # self.pub_z.publish(z_pub)
        # self.total.publish(total)

        # if (abs(self.tau_hat_body[1]) > self.max):
        #     self.max = abs(self.tau_hat_body[1])
        #     print("Total",self.tau_hat_body[1])
        #     print("Term1:",x[1],"Term2:",y[1],"Term3:",z[1])

        # print(self.tau_hat_body[1])
        # print(total_pub)

    def wrench_estimator_arm(self,imu1,imu2,imu3,imu4):

        msg = ExternalWrenchEstimate()

        msg.f_x = 0.0
        msg.f_y = 0.0
        msg.f_z = 0.0
        msg.tau_p = 0.0
        msg.tau_q = 0.0
        msg.tau_r = 0.0

        self.arm_length = 0.15
        
        self.f_hat_imu1 = ( 1.3077 * imu1[0] + 0.01 * imu1[1] + 0.0015 * imu1[2] ) / self.arm_length # Arm Distance = 0.15m
        self.f_hat_imu2 = ( 1.3077 * imu2[0] + 0.01 * imu2[1] + 0.0015 * imu2[2] ) / self.arm_length
        self.f_hat_imu3 = ( 1.3077 * imu3[0] + 0.01 * imu3[1] + 0.0015 * imu3[2] ) / self.arm_length
        self.f_hat_imu4 = ( 1.3077 * imu4[0] + 0.01 * imu4[1] + 0.0015 * imu4[2] ) / self.arm_length

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

        self.f_hat_imu1_b = np.matmul(self.rt_matrix_alpha1,np.array([self.f_hat_imu1,0,0]))
        self.f_hat_imu2_b = np.matmul(self.rt_matrix_alpha2,np.array([self.f_hat_imu2,0,0]))
        self.f_hat_imu3_b = np.matmul(self.rt_matrix_alpha3,np.array([self.f_hat_imu3,0,0]))
        self.f_hat_imu4_b = np.matmul(self.rt_matrix_alpha4,np.array([self.f_hat_imu4,0,0]))
        
        self.f_hat_imu1_b_w = np.matmul(self.rotation_matrix, self.f_hat_imu1_b)
        self.f_hat_imu2_b_w = np.matmul(self.rotation_matrix, self.f_hat_imu2_b)
        self.f_hat_imu3_b_w = np.matmul(self.rotation_matrix, self.f_hat_imu3_b)
        self.f_hat_imu4_b_w = np.matmul(self.rotation_matrix, self.f_hat_imu4_b)
        
        # print("Alpha:", self.alpha[2])
        # print(self.f_hat_imu3,"--",self.f_hat_imu3_b,"--",self.f_hat_imu3_b_w)

        self.f_hat_b_w =  self.f_hat_imu1_b_w + self.f_hat_imu2_b_w + self.f_hat_imu3_b_w + self.f_hat_imu4_b_w

        self.tau_hat_arm = (self.f_hat_imu1 + self.f_hat_imu2 + self.f_hat_imu3 + self.f_hat_imu4) * self.arm_length

        msg.f_x = float(self.f_hat_b_w[0] + self.f_hat_body[0])
        msg.f_y = float(self.f_hat_b_w[1] + self.f_hat_body[1])
        msg.f_z = float(self.f_hat_b_w[2] + self.f_hat_body[2]) + 11.772

        msg.tau_p = float(self.tau_hat_body[0])
        msg.tau_q = float(self.tau_hat_body[1])
        msg.tau_r = float(self.tau_hat_body[2] + self.tau_hat_arm)

        if (abs(msg.tau_q) > self.max):
            self.max = abs(msg.tau_q)
            print(self.max)

        msg.timestamp = self.timestamp

        # print(self.f_hat_b_w,self.tau_hat_arm)
        self.wrench_pub.publish(msg)
        # print("Wrench Estimator Running")
        # print("------------------------")
        # print(msg)

    def rt_matrix(self, theta):

        return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

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
