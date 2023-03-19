##################################################################################################
import rclpy
from rclpy.node import Node
##################################################################################################
"""
Custom Messages to obtain Arm Angles and to Publish the Estimated Wrench
Refer "https://github.com/ASU-RISE-Lab/squeeze_custom_msgs.git" for more info on the message type.
"""
from squeeze_custom_msgs.msg import ImuData
from squeeze_custom_msgs.msg import ExternalWrenchEstimate
##################################################################################################
import numpy as np
import time
##################################################################################################
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter,filtfilt
##################################################################################################
import math
##################################################################################################
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float64
##################################################################################################
from px4_msgs.msg import ControllerOut
from px4_msgs.msg import VehicleOdometry
##################################################################################################
"""
Filters Imported to be used in the Wrench Estimator
"""
from MLPF import Median_LPF
# from .MLPF import Median_LPF
from MV_Filter import Moving_Avg_Filter
##################################################################################################

class Wrench_Estimator(Node):

    def __init__(self):
        super().__init__('Wrench_Estimator')
        print("Starting Wrench Estimator Node")
########################################################################################################################################
        self.initialize_variables()
########################################################################################################################################
        self.initialize_controller_filters()
        self.initialize_odometry_filters()
        self.initialize_wrench_filters()
        self.initialize_body_wrench_filter()
########################################################################################################################################
        # Subscribers
        self.imu_angle_sub = self.create_subscription(ImuData, '/Arm_Angles_Filtered', self.imu_angle_callback, 1)
        self.vehicle_odometry_sub = self.create_subscription(VehicleOdometry, '/fmu/vehicle_odometry/out', self.vehicle_odometry_callback, 1)
        self.controller_out_sub = self.create_subscription(ControllerOut, '/fmu/controller_out/out', self.controller_out_callback, 1)
        # self.move_direction_sub = self.create_subscription(Float64, '/Move_Direction', self.move_direction_callback, 1)
        # self.contacts_sub = self.create_subscription(Float64, '/Contacts', self.contacts_callback, 1)

        # self.debugging_publishers()
        self.experiment_publishers()

########################################################################################################################################

    def initialize_variables(self): # Initialize the Common Variables used in the Node
        self.rotation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        self.timestamp = time.time()

        self.imu1_prev = [0.0,0.0,0.0]
        self.imu2_prev = [0.0,0.0,0.0]
        self.imu3_prev = [0.0,0.0,0.0]
        self.imu4_prev = [0.0,0.0,0.0]

        self.f = np.array([0.0,0.0,0.0])
        self.tau = np.array([0.0,0.0,0.0])

        self.e1 = np.array([1,0,0])
        self.e2 = np.array([0,1,0])
        self.e3 = np.array([0,0,1])
        self.alpha = np.zeros(4) # Angle to compute Rotation Matrix for arm wrench

        self.inertia_mat = ([0.011160,0.000003,0.000058],
                            [0.000003, 0.011260,0.000044],
                            [0.000058,0.000044,0.018540]) # Inertia Matrix of the Drone in Kg.m^2
        self.mass = 1.2 # 1.174  # 1.566  # Mass of the Drone in Kilograms
        self.g = 9.81       # Acceleration due to gravity in m/s^2

        self.friction_force = [0.0,0.0,0.0]
        self.contacts = 0.0
        self.net_wrench = np.zeros(6)

        self.tau_hat_body = np.zeros(3)

        self.tau_hat = np.zeros(3)
        self.tau_hat_dot = np.zeros(3)

    def node_subscribers(self): # Initialize the Subscribers (TBD)
        pass

    def node_publishers(self): # Initialize the Publishers (TBD)
        pass
    
    def experiment_publishers(self): # Initialize the Experiment Publishers
        # Publishers
        self.wrench_pub = self.create_publisher(ExternalWrenchEstimate, '/External_Wrench_Estimate', 1)
        self.yaw_euler_pub = self.create_publisher(Float32MultiArray, '/Euler_Angles', 1)

        # Debug Publishers
        self.filt_controller_out_pub = self.create_publisher(ControllerOut, '/fmu/controller_out_filtered/out', 1)
        self.velocity_unfilt_pub = self.create_publisher(Float32MultiArray, '/Velocity/UnFilt', 1)
        self.velocity_dot_unfilt_pub = self.create_publisher(Float32MultiArray, '/Velocity_Dot/UnFilt', 1)
        self.velocity_pub = self.create_publisher(Float32MultiArray, '/Velocity/Filt', 1)
        self.velocity_dot_pub = self.create_publisher(Float32MultiArray, '/Velocity_Dot/Filt', 1)
        self.omega_unfilt_pub = self.create_publisher(Float32MultiArray, '/Omega/UnFilt', 1)
        self.omega_dot_unfilt_pub = self.create_publisher(Float32MultiArray, '/Omega_Dot/UnFilt', 1)
        self.omega_pub = self.create_publisher(Float32MultiArray, '/Omega_New/Filt', 1)
        self.omega_dot_pub = self.create_publisher(Float32MultiArray, '/Omega_Dot/Filt', 1)
        self.body_wrench_pub = self.create_publisher(Float32MultiArray, 'Wrench/body_wrench', 1)
        self.arm_wrench_pub = self.create_publisher(Float32MultiArray, 'Wrench/arm_wrench', 1)
        self.friction_wrench_pub = self.create_publisher(Float32MultiArray,'Wrench/friction_wrench',1)
        self.rotation_matrix_pub = self.create_publisher(Float32MultiArray, '/Rotation_Matrix', 1)
        self.body_wrench_fre3_pub = self.create_publisher(Float32MultiArray, 'Wrench/body_wrench_fre3', 1)

    def debugging_publishers(self): # Initialize the Debugging Publishers
        # Publishers
        self.wrench_pub = self.create_publisher(ExternalWrenchEstimate, 'Debug/External_Wrench_Estimate', 1)
        self.yaw_euler_pub = self.create_publisher(Float32MultiArray, 'Debug/Euler_Angles', 1)

        # Debug Publishers
        self.filt_controller_out_pub = self.create_publisher(ControllerOut, 'Debug/fmu/controller_out_filtered/out', 1)
        self.velocity_unfilt_pub = self.create_publisher(Float32MultiArray, 'Debug/Velocity/UnFilt', 1)
        self.velocity_dot_unfilt_pub = self.create_publisher(Float32MultiArray, 'Debug/Velocity_Dot/UnFilt', 1)
        self.velocity_pub = self.create_publisher(Float32MultiArray, 'Debug/Velocity/Filt', 1)
        self.velocity_dot_pub = self.create_publisher(Float32MultiArray, 'Debug/Velocity_Dot/Filt', 1)
        self.omega_unfilt_pub = self.create_publisher(Float32MultiArray, 'Debug/Omega/UnFilt', 1)
        self.omega_dot_unfilt_pub = self.create_publisher(Float32MultiArray, 'Debug/Omega_Dot/UnFilt', 1)
        self.omega_pub = self.create_publisher(Float32MultiArray, 'Debug/Omega_New/Filt', 1)
        self.omega_dot_pub = self.create_publisher(Float32MultiArray, 'Debug/Omega_Dot/Filt', 1)
        self.body_wrench_pub = self.create_publisher(Float32MultiArray, 'Debug/Wrench/body_wrench', 1)
        self.arm_wrench_pub = self.create_publisher(Float32MultiArray, 'Debug/Wrench/arm_wrench', 1)
        self.friction_wrench_pub = self.create_publisher(Float32MultiArray,'Debug/Wrench/friction_wrench',1)
        self.rotation_matrix_pub = self.create_publisher(Float32MultiArray, 'Debug/Rotation_Matrix', 1)
        self.body_wrench_fre3_pub = self.create_publisher(Float32MultiArray, 'Debug/Wrench/body_wrench_fre3', 1)

    def initialize_controller_filters(self): # Initialize Class object & Variables for Controller Data Filters
        self.controller_fx_filter = Median_LPF()
        self.controller_fy_filter = Median_LPF()
        self.controller_fz_filter = Median_LPF()
        self.controller_taux_filter = Median_LPF()
        self.controller_tauy_filter = Median_LPF()
        self.controller_tauz_filter = Median_LPF()
        self.alpha_controller_out = np.array([0.8,0.8,0.4,0.8,0.8,0.8,0.8]) # MLPF Filter time constants for Controller_Out # fx = 0.8
        self.controller_out_filter_buffer = np.zeros((6,3)) # Fx, Fy, Fz, Taux, Tauy, Tauz = [prev, current, next]

    def initialize_body_wrench_filter(self):
        self.fre3_filter_x = Median_LPF()
        self.fre3_filter_y = Median_LPF()
        self.fre3_filter_z = Median_LPF()
        self.fre3_buffer = np.zeros((3,3))
        self.fre3_alpha = np.array([0.2,0.2,0.4])

        self.fre3_x_mv_filter = Moving_Avg_Filter(30)
        self.fre3_y_mv_filter = Moving_Avg_Filter(30)
        self.fre3_z_mv_filter = Moving_Avg_Filter(30)

        self.fre3_filter_prev = [float(0.0),float(0.0),float(0.0)]

        # self.fre3_delay_buffer = np.zeros((3,15))

    def initialize_odometry_filters(self): # Initialize Class object & Variables for Odometry Data Filters
        self.vx_filter = Median_LPF()
        self.vy_filter = Median_LPF()
        self.vz_filter = Median_LPF()
        self.ax_filter = Median_LPF()
        self.ay_filter = Median_LPF()
        self.az_filter = Median_LPF()
        self.omega_x_filter = Median_LPF()
        self.omega_y_filter = Median_LPF()
        self.omega_z_filter = Median_LPF()
        self.omega_dot_p_filter = Median_LPF()
        self.omega_dot_q_filter = Median_LPF()
        self.omega_dot_r_filter = Median_LPF()
        self.v_filter_buffer = np.zeros((3,3))
        self.v_dot_filter_buffer = np.zeros((3,3))
        self.omega_filter_buffer = np.zeros((3,3))
        self.omega_dot_filter_buffer = np.zeros((3,3))
        self.alpha_v = [0.2,0.2,0.4]        # MLPF Filter time constants for v
        self.alpha_v_dot = [0.4,0.4,1.0]   # MLPF Filter time constants for v_dot  # old vy = 0.4
        # self.alpha_omega = 0.1              # LP filter time constant for omega
        # self.alpha_omega_dot = 0.05         # LP filter time constant for omega_dot
        self.alpha_omega = [0.4,0.4,0.6]
        self.alpha_omega_dot = [0.4,0.4,0.75]

        self.vx_mv_filter = Moving_Avg_Filter(30)
        self.vy_mv_filter = Moving_Avg_Filter(30)
        self.vz_mv_filter = Moving_Avg_Filter(30)

        self.v_timestamp = 0.0
        self.v = np.array([0.0,0.0,0.0])
        self.v_prev = np.array([0.0,0.0,0.0])
        self.v_dot = np.array([0.0,0.0,0.0])
        self.v_dot_prev = np.array([0.0,0.0,0.0])
        self.omega = np.array([0.0,0.0,0.0])
        self.omega_prev = np.array([0.0,0.0,0.0])
        self.omega_dot = np.array([0.0,0.0,0.0])
        # self.omega_dot_prev = np.array([0.0,0.0,0.0])

    def initialize_wrench_filters(self): # Initialize Class object & Variables for Odometry Wrench Filters
        
        self.f_hat_body_prev = np.array([0.0,0.0,0.0])
        self.f_hat_body = np.array([0.0,0.0,0.0])

        self.wrench_fx_filter = Median_LPF()
        self.wrench_fy_filter = Median_LPF()
        self.wrench_fz_filter = Median_LPF()

        self.alpha_wrench = [0.4,0.4,0.4] # MLPF Filter time constants for Wrench Force

        self.wrench_f_buffer = np.zeros((3,3))

    def lp_filter(self,current, previous, alpha): # First Order Low Pass Filter
        return alpha*current + (1-alpha)*previous

    def controller_out_callback(self, msg): # Controller Out Callback

        # print(msg.f[0], msg.f[1], msg.f[2], msg.tau[0], msg.tau[1], msg.tau[2])

        filt_controller_out = ControllerOut()

        # self.controller_out_estimate = np.array([msg.f[0]*0, msg.f[1]*0, msg.f[2]*32.66, msg.tau[0]*1.36, msg.tau[1]*1.36, msg.tau[2]*0.012])

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

        self.f = np.array([filt_controller_out.f[0], filt_controller_out.f[1], filt_controller_out.f[2] * 32.5])
        self.tau = np.array([filt_controller_out.tau[0], filt_controller_out.tau[1], filt_controller_out.tau[2]])

        filt_controller_out.timestamp = msg.timestamp

        self.filt_controller_out_pub.publish(filt_controller_out)

    def vehicle_odometry_callback(self, msg): # Vehicle Odometry Callback
        self.v = np.array([msg.vx, msg.vy, msg.vz])
        self.omega = np.array([msg.rollspeed, msg.pitchspeed, msg.yawspeed])
        self.v_dt = (msg.timestamp - self.v_timestamp) * 10**-6 # Convert to Microseconds
#################################################################################################
        v_unfilt = Float32MultiArray()
        v_unfilt.data = [msg.vx, msg.vy, msg.vz]
        self.velocity_unfilt_pub.publish(v_unfilt)

        omega_unfilt = Float32MultiArray()
        omega_unfilt.data = [msg.rollspeed, msg.pitchspeed, msg.yawspeed]
        self.omega_unfilt_pub.publish(omega_unfilt)
#################################################################################################3
        self.v_filter_buffer[0][:2] = self.v_filter_buffer[0][1:]
        self.v_filter_buffer[0][2] = msg.vx

        self.v_filter_buffer[1][:2] = self.v_filter_buffer[1][1:]
        self.v_filter_buffer[1][2] = msg.vy

        self.v_filter_buffer[2][:2] = self.v_filter_buffer[2][1:]
        self.v_filter_buffer[2][2] = msg.vz

        self.v[0] = self.vx_filter.med_lp(self.v_filter_buffer[0],self.alpha_v[0])        # VelX Filter
        self.v[1] = self.vy_filter.med_lp(self.v_filter_buffer[1],self.alpha_v[1])        # VelY Filter
        self.v[2] = self.vz_filter.med_lp(self.v_filter_buffer[2],self.alpha_v[2])        # VelZ Filter
#################################################################################################
        # self.omega = (self.omega * self.alpha_omega) + (self.omega_prev * (1 - self.alpha_omega))     # Omega Filter

        self.omega_filter_buffer[0][:2] = self.omega_filter_buffer[0][1:]
        self.omega_filter_buffer[0][2] = msg.rollspeed

        self.omega_filter_buffer[1][:2] = self.omega_filter_buffer[1][1:]
        self.omega_filter_buffer[1][2] = msg.pitchspeed

        self.omega_filter_buffer[2][:2] = self.omega_filter_buffer[2][1:]
        self.omega_filter_buffer[2][2] = msg.yawspeed

        self.omega[0] = self.omega_x_filter.med_lp(self.omega_filter_buffer[0],self.alpha_omega[0])        # OmegaX Filter
        self.omega[1] = self.omega_y_filter.med_lp(self.omega_filter_buffer[1],self.alpha_omega[1])        # OmegaY Filter
        self.omega[2] = self.omega_z_filter.med_lp(self.omega_filter_buffer[2],self.alpha_omega[2])        # OmegaZ Filter
#################################################################################################
        self.v_dot = (self.v - self.v_prev) / self.v_dt
        self.omega_dot = (self.omega - self.omega_prev) / self.v_dt
#################################################################################################
        v_dot_unfilt = Float32MultiArray()
        v_dot_unfilt.data = [self.v_dot[0], self.v_dot[1], self.v_dot[2]]
        self.velocity_dot_unfilt_pub.publish(v_dot_unfilt)

        omega_dot_unfilt = Float32MultiArray()
        omega_dot_unfilt.data = [self.omega_dot[0], self.omega_dot[1], self.omega_dot[2]]
        self.omega_dot_unfilt_pub.publish(omega_dot_unfilt)
#################################################################################################
        #self.omega_dot = (self.omega_dot * self.alpha_omega_dot) + (self.omega_dot_prev * (1 - self.alpha_omega_dot))   # Omegadot Filter

        self.omega_dot_filter_buffer[0][:2] = self.omega_dot_filter_buffer[0][1:]
        self.omega_dot_filter_buffer[0][2] = self.omega_dot[0]

        self.omega_dot_filter_buffer[1][:2] = self.omega_dot_filter_buffer[1][1:]
        self.omega_dot_filter_buffer[1][2] = self.omega_dot[1]

        self.omega_dot_filter_buffer[2][:2] = self.omega_dot_filter_buffer[2][1:]
        self.omega_dot_filter_buffer[2][2] = self.omega_dot[2]

        self.omega_dot[0] = self.omega_dot_p_filter.med_lp(self.omega_dot_filter_buffer[0],self.alpha_omega_dot[0])    # Omega_DotP Filter
        self.omega_dot[1] = self.omega_dot_q_filter.med_lp(self.omega_dot_filter_buffer[1],self.alpha_omega_dot[1])    # Omega_DotQ Filter
        self.omega_dot[2] = self.omega_dot_r_filter.med_lp(self.omega_dot_filter_buffer[2],self.alpha_omega_dot[2])    # Omega_DotR Filter
#################################################################################################
        self.v_dot_filter_buffer[0][:2] = self.v_dot_filter_buffer[0][1:]
        self.v_dot_filter_buffer[0][2] = self.v_dot[0]

        self.v_dot_filter_buffer[1][:2] = self.v_dot_filter_buffer[1][1:]
        self.v_dot_filter_buffer[1][2] = self.v_dot[1]

        self.v_dot_filter_buffer[2][:2] = self.v_dot_filter_buffer[2][1:]
        self.v_dot_filter_buffer[2][2] = self.v_dot[2]

        self.v_dot[0] = self.ax_filter.med_lp(self.v_dot_filter_buffer[0],self.alpha_v_dot[0])    # AccX Filter
        self.v_dot[1] = self.ay_filter.med_lp(self.v_dot_filter_buffer[1],self.alpha_v_dot[1])    # AccY Filter
        self.v_dot[2] = self.az_filter.med_lp(self.v_dot_filter_buffer[2],self.alpha_v_dot[2])    # AccZ Filter

        self.v_dot[0] = self.vx_mv_filter.callback(self.v_dot[0]) # Moving Avg Window 50
        self.v_dot[1] = self.vy_mv_filter.callback(self.v_dot[1])
        self.v_dot[2] = self.vz_mv_filter.callback(self.v_dot[2])
#################################################################################################
        self.v_prev = self.v                 # To Calculate Vdot
        self.omega_prev = self.omega         # To Calculate Omegadot
        # self.omega_dot_prev = self.omega_dot # Used in Omegadot Filter
        self.v_timestamp = msg.timestamp     # To Calculate dt
#################################################################################################
        vel_pub_filt = Float32MultiArray()
        vel_dot_pub_filt = Float32MultiArray()

        omega_pub_filt = Float32MultiArray()
        omega_dot_pub_filt = Float32MultiArray()

        vel_pub_filt.data = [float(self.v[0]), float(self.v[1]), float(self.v[2])]
        vel_dot_pub_filt.data = [float(self.v_dot[0]), float(self.v_dot[1]), float(self.v_dot[2])]   

        omega_pub_filt.data = [float(self.omega[0]), float(self.omega[1]), float(self.omega[2])]
        omega_dot_pub_filt.data = [float(self.omega_dot[0]), float(self.omega_dot[1]), float(self.omega_dot[2])]

        self.velocity_pub.publish(vel_pub_filt)
        self.velocity_dot_pub.publish(vel_dot_pub_filt)

        self.omega_pub.publish(omega_pub_filt)
        self.omega_dot_pub.publish(omega_dot_pub_filt)
#################################################################################################
        q = np.zeros(4)

        q[3] = msg.q[0]
        q[0] = msg.q[1]
        q[1] = msg.q[2]
        q[2] = msg.q[3]

        rot = Rotation.from_quat(q)

        self.rotation_matrix = R.from_quat([msg.q[1],msg.q[2],msg.q[3],msg.q[0]]).as_matrix()

        # if (self.rotation_matrix[0][2] < 0.02): 
        #     self.rotation_matrix[0][2] = 0

        # if (self.rotation_matrix[1][2] < 0.02): 
        #     self.rotation_matrix[1][2] = 0

        rot_euler = rot.as_euler('xyz', degrees=True)  # [roll, pitch, yaw]

        rot_matrix_pub = Float32MultiArray()

        rot_matrix_pub.data = [float(self.rotation_matrix[0][0]), float(self.rotation_matrix[0][1]), float(self.rotation_matrix[0][2]), 
                           float(self.rotation_matrix[1][0]), float(self.rotation_matrix[1][1]), float(self.rotation_matrix[1][2]), 
                           float(self.rotation_matrix[2][0]), float(self.rotation_matrix[2][1]), float(self.rotation_matrix[2][2])]

        yaw = Float32MultiArray()

        self.rotation_matrix_pub.publish(rot_matrix_pub)
        yaw.data = [float(rot_euler[0]), float(rot_euler[1]), float(rot_euler[2])]
        self.yaw_euler_pub.publish(yaw)

        self.wrench_estimator_body()

    def imu_angle_callback(self, msg): # IMU Angles Callback and Wrench Estimator Function Calls

        imu1 = [0.0, 0.0, 0.0] # [dtheta, dtheta/dt, ddtheta/dtt]
        imu2 = [0.0, 0.0, 0.0]
        imu3 = [0.0, 0.0, 0.0]
        imu4 = [0.0, 0.0, 0.0]

        self.theta = [msg.imu1, msg.imu2, msg.imu3, msg.imu4] # Actual arm angles from node

        msg.imu1 = -(msg.imu1 - 131.4788) * 3.14/180
        msg.imu2 = (46.6722 - msg.imu2) * 3.14/180
        msg.imu3 = (47.9821 + msg.imu3) * 3.14/180
        msg.imu4 = (131.8567 + msg.imu4) * 3.14/180

        if(msg.timestamp - self.timestamp != 0):

            dt = (msg.timestamp - self.timestamp) / 10**6  # TimeStamp in Microseconds

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

            self.wrench_estimator_arm(imu1,imu2,imu3,imu4)
            # self.wrench_estimator_friction()

            self.imu1_prev = imu1
            self.imu2_prev = imu2
            self.imu3_prev = imu3
            self.imu4_prev = imu4

        else :
            print("Timestamp Error")

    def dthetha_eff(self, dtheta): # Curve Fitting for Arm Angle Deflection
                                   # If Deflection is less than 15 Degrees (0.26 Rads) we use (2*dtheta^3) else we use same dtheta
 
        # if (dtheta != 0):
        #     # dthetha_eff =  ( 4 / (1 + 0.15 * abs(dtheta))) * (dtheta - 0.085 * (dtheta / abs(dtheta)))**3
        #     dtheta_eff = (4.2 * dtheta)**3
        # else:
        #     dtheta_eff = 0

        if (abs(dtheta) < 0.12): #15 Degress = 0.26 Rads # 7 Degrees = 0.12 Rads
            dtheta_eff = (2 * dtheta)**3 
        else:
            dtheta_eff = dtheta
        
        # return dtheta
        return dtheta_eff

    def contacts_callback(self,msg):
        self.contacts = msg.data

    def move_direction_callback(self,msg):

        self.friction_force = [0.0,0.0,0.0]

        if (self.contacts != 0):
            if(msg.data == 0.0 or msg.data == 1.0):
                self.friction_direction = 'x'
            if(msg.data == 2.0 or msg.data == 3.0):
                self.friction_direction = 'y'
            
            if(self.friction_direction == 'x'):
                self.friction_force = -self.mass * self.v * 30.0
                self.friction_force = [self.friction_force[0],0.0,0.0]
            elif(self.friction_direction == 'y'):
                self.friction_force = -self.mass * self.v * 6.0
                self.friction_force = [0.0,self.friction_force[1],0.0]

    def wrench_estimator_friction(self):
        if (abs(self.net_wrench[0]) > 1.1):
                self.friction_force = -self.mass * self.v * 6.0
                self.friction_force = [0.0,self.friction_force[1],0.0]
        elif (abs(self.net_wrench[1]) > 1.1):
                self.friction_force = -self.mass * self.v * 60.0
                self.friction_force = [self.friction_force[0],0.0,0.0]
        else : 
            self.friction_force = [0.0,0.0,0.0]

    def wrench_estimator_body(self): # Body Wrench Estimator called from Vehicle Odom Callback

        fre3_filter = [0.0,0.0,0.0]
        fre3_filter_delay = [0.0,0.0,0.0]

        # fre3 = (self.f[2] * np.matmul(np.transpose(self.rotation_matrix), self.e3))
        fre3 = (self.f[2] * np.matmul((self.rotation_matrix), self.e3))

        self.fre3_buffer[0][:2] = self.fre3_buffer[0][1:]
        self.fre3_buffer[0][2] = fre3[0]

        self.fre3_buffer[1][:2] = self.fre3_buffer[1][1:]
        self.fre3_buffer[1][2] = fre3[1]

        self.fre3_buffer[2][:2] = self.fre3_buffer[2][1:]
        self.fre3_buffer[2][2] = fre3[2]

        fre3_filter[0] = self.fre3_filter_x.med_lp(self.fre3_buffer[0],self.fre3_alpha[0])
        fre3_filter[1] = self.fre3_filter_y.med_lp(self.fre3_buffer[1],self.fre3_alpha[1])
        fre3_filter[2] = self.fre3_filter_z.med_lp(self.fre3_buffer[2],self.fre3_alpha[2])
        ########################################################################################
        fre3_filter[0] = float(self.fre3_x_mv_filter.callback(fre3_filter[0]))
        fre3_filter[1] = float(self.fre3_y_mv_filter.callback(fre3_filter[1]))
        fre3_filter[2] = float(self.fre3_z_mv_filter.callback(fre3_filter[2]))
        ########################################################################################
        fre3_filter[0] = fre3_filter[0] * 0.2 + self.fre3_filter_prev[0] * 0.8
        fre3_filter[1] = fre3_filter[1] * 0.2 + self.fre3_filter_prev[1] * 0.8
        fre3_filter[2] = fre3_filter[2] * 0.2 + self.fre3_filter_prev[2] * 0.8

        self.fre3_filter_prev = fre3_filter
        ########################################################################################
        # self.fre3_delay_buffer[0][:-1] = self.fre3_delay_buffer[0][1:]
        # self.fre3_delay_buffer[0][-1] = fre3_filter[0]

        # self.fre3_delay_buffer[1][:-1] = self.fre3_delay_buffer[1][1:]
        # self.fre3_delay_buffer[1][-1] = fre3_filter[1]

        # self.fre3_delay_buffer[2][:-1] = self.fre3_delay_buffer[2][1:]
        # self.fre3_delay_buffer[2][-1] = fre3_filter[2]

        # fre3_filter_delay[0] = self.fre3_delay_buffer[0][0]
        # fre3_filter_delay[1] = self.fre3_delay_buffer[1][0]
        # fre3_filter_delay[2] = self.fre3_delay_buffer[2][0]
        ########################################################################################
        self.fre3_pub_var = Float32MultiArray()
        self.fre3_pub_var.data = [float(fre3_filter[0]), float(fre3_filter[1]), float(fre3_filter[2])]
        ########################################################################################
        self.f_hat_body = ((self.mass * self.v_dot) - (self.mass * self.g * self.e3) - fre3_filter)

        # self.tau_hat_dot = ((self.mass * self.v_dot) - (self.mass * self.g * self.e3) - fre3_filter - self.tau_hat) #* 0.02

        # self.tau_hat = self.tau_hat + (self.tau_hat_dot * 0.02)

        # self.f_hat_body = self.tau_hat

        self.tau_hat_body = (np.matmul(self.inertia_mat , (self.omega_dot))) + (np.cross((self.omega), np.matmul(self.inertia_mat , self.omega))) - (np.transpose(self.tau))
        
        self.f_hat_body = self.lp_filter(self.f_hat_body, self.f_hat_body_prev, 0.3)
        self.f_hat_body_prev = self.f_hat_body
        ########################################################################################
        self.body_wrench_fre3_pub.publish(self.fre3_pub_var)
        x = Float32MultiArray()
        x.data = [float(self.f_hat_body[0]), float(self.f_hat_body[1]), float(self.f_hat_body[2])]
        self.body_wrench_pub.publish(x)
        # self.wrench_estimator_friction()

    def wrench_estimator_arm(self,imu1,imu2,imu3,imu4): # Arm Wrench Estimator and Total Wrench Estimator called from IMU Callback

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
        
        self.f_hat_b_w =  self.f_hat_imu1_b_w + self.f_hat_imu2_b_w + self.f_hat_imu3_b_w + self.f_hat_imu4_b_w

        self.tau_hat_arm = (self.f_hat_imu1 + self.f_hat_imu2 + self.f_hat_imu3 + self.f_hat_imu4) * self.arm_length

        # msg.f_x = self.net_wrench[0] = float(self.f_hat_b_w[0] + self.f_hat_body[0] - self.friction_force[0])
        # msg.f_y = self.net_wrench[1] = float(self.f_hat_b_w[1] + self.f_hat_body[1] - self.friction_force[1])
        # msg.f_z = self.net_wrench[2] = float(self.f_hat_b_w[2] + self.f_hat_body[2] - self.friction_force[2])

        msg.f_x = float(self.f_hat_body[0])
        msg.f_y = float(self.f_hat_body[1])
        msg.f_z = float(self.f_hat_body[2])

        msg.tau_p = self.net_wrench[3] = float(self.tau_hat_body[0])
        msg.tau_q = self.net_wrench[4] = float(self.tau_hat_body[1])
        msg.tau_r = self.net_wrench[5] = float(self.tau_hat_body[2] + self.tau_hat_arm)

        self.wrench_f_buffer[0][:2] = self.wrench_f_buffer[0][1:]
        self.wrench_f_buffer[0][2] = msg.f_x

        self.wrench_f_buffer[1][:2] = self.wrench_f_buffer[1][1:]
        self.wrench_f_buffer[1][2] = msg.f_y

        # self.wrench_f_buffer[2][:2] = self.wrench_f_buffer[2][1:]
        # self.wrench_f_buffer[2][2] = msg.f_z

        msg.f_x = self.wrench_fx_filter.med_lp(self.wrench_f_buffer[0],self.alpha_wrench[0])
        msg.f_y = self.wrench_fy_filter.med_lp(self.wrench_f_buffer[1],self.alpha_wrench[1])
        # msg.f_z = self.wrench_fz_filter.med_lp(self.wrench_f_buffer[2],self.alpha_wrench[2])

        msg.timestamp = self.timestamp
###################################################################################################################
        arm_f = Float32MultiArray()
        arm_f.data = [self.f_hat_imu1,self.f_hat_imu2,self.f_hat_imu3,self.f_hat_imu4]
        self.arm_wrench_pub.publish(arm_f)

        friction_wrench = Float32MultiArray()
        friction_wrench.data = [float(self.friction_force[0]),float(self.friction_force[1]),float(self.friction_force[2])]
        self.friction_wrench_pub.publish(friction_wrench)
###################################################################################################################
        self.wrench_pub.publish(msg)
        print("Wrench Estimator Running - ", self.timestamp)
        print("------------------------")

    def rt_matrix(self, theta): # Function to compute the Rotation Matrix

        return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

def main(args=None):
    rclpy.init(args=args)

    wrench = Wrench_Estimator()

    rclpy.spin(wrench)

    Wrench_Estimator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()