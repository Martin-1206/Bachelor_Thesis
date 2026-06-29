import os
import mujoco
import mujoco_viewer
import mujoco._render
import numpy as np
import qmt
import kinematics_and_jacobian.J_pos_ypr_fk_base_link_left_toe_motor_jax as Jacobian_left
import kinematics_and_jacobian.J_pos_ypr_fk_base_link_right_toe_motor_jax as Jacobian_right
import kinematics_and_jacobian.pos_quaternion_fk_base_link_left_toe_jax as forward_kinematic_left_body
import kinematics_and_jacobian.pos_quaternion_fk_base_link_right_toe_jax as forward_kinematic_right_body

import kinematics_and_jacobian.pos_quaternion_fk_origin_left_toe_jax as forward_kinematic_left
import kinematics_and_jacobian.pos_quaternion_fk_origin_right_toe_jax as forward_kinematic_right

import jax
from jax import jit
import jax.numpy as jnp
import matplotlib.pyplot as plt
import invers_kinematic_levenberg_marquardt as inverse_kinematic
from scipy.spatial.transform import Rotation as R

from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
model = mujoco.MjModel.from_xml_path('/home/martin/Bachelorarbeit/DIGIT/digit-v3_stl/digit-v3.xml') # create the Model from Path
data = mujoco.MjData(model) # creates the data


q_hip_roll_vel_desired = 0
q_hip_yaw_vel_desired = 0
q_hip_pitch_vel_desired = 0
q_knee_vel_desired = 0
q_toe_pitch_vel_desired = 0
q_vels_desired = np.array([q_hip_roll_vel_desired, q_hip_yaw_vel_desired, q_hip_pitch_vel_desired, q_knee_vel_desired, q_toe_pitch_vel_desired])

actuator_names = [
    'left-hip-roll',
    'left-hip-yaw' ,
    'left-hip-pitch',
    'left-knee' ,
    'left-toe-A',
    'left-toe-B',
    'right-hip-roll' ,
    'right-hip-yaw' ,
    'right-hip-pitch' ,
    'right-knee' ,
    'right-toe-A' ,
    'right-toe-B' ,
    'left-shoulder-roll' ,
    'left-shoulder-pitch' ,
    'left-shoulder-yaw' ,
    'left-elbow' ,
    'right-shoulder-roll' ,
    'right-shoulder-pitch' ,
    'right-shoulder-yaw',
    'right-elbow' ,
]

# Damping per joint from hardware
damping_limit = np.array([66.849046,
                          26.112909,
                          38.05002,
                          38.05002,
                          28.553161,
                          28.553161,
                          66.849046,
                          26.112909,
                          38.05002,
                          38.05002,
                          28.553161,
                          28.553161,
                          66.849046,
                          66.849046,
                          26.112909,
                          66.849046,
                          66.849046,
                          66.849046,
                          26.112909,
                          66.849046,
                          ])

kd = 1 # 0.9 #0.6

jnt_dofadr= [model.jnt_dofadr[mujoco.mj_name2id(model,3, name)]for name in actuator_names]

def set_start_position():
    ''' Sets the angle of the joints --> results in a Start Position '''
    data.qpos[2] = 0.91 #20218
    data.qpos[7] = np.deg2rad(18)  # left-hip-roll
    data.qpos[8] = 0.0  # left-hip-yaw
    data.qpos[9] = 0.17 # left-hip-pitch
    data.qpos[10] = 0.04 
    data.qpos[11] = 0.01
    data.qpos[31] = np.deg2rad(50)# Test by Martin # Arme hängen lassen
    data.qpos[33] = np.deg2rad(60)# Test by Martin
    data.qpos[34] = -np.deg2rad(18) # right-hip-roll
    data.qpos[35] = 0.0 # right-hip-yaw
    data.qpos[36] = -0.17 # right-hip-pitch
    data.qpos[37] = -0.04
    data.qpos[38] = 0.01
    data.qpos[58] = -np.deg2rad(50) # Test by Martin
    data.qpos[60] = -np.deg2rad(60) # Test by Martin


def configuration_space_function(sensordata):
    ''' Gelenkwinkel '''
    
    base_pos = sensordata[model.sensor_adr[mujoco.mj_name2id(model, 19, "base-pos")]:3] # Sensor_adress 1st to 3rd

    base_quat = sensordata[model.sensor_adr[mujoco.mj_name2id(model, 19, "base-quat")]:model.sensor_adr[mujoco.mj_name2id(model, 19, "base-quat")] + 4] # +4 bcs the quaternion has 4 items
    pelvis_orientation = qmt.eulerAngles(base_quat) # transform the quaternion to euler angles z y x 

    joint_poses = sensordata[7:35]
    
    configuration_space_vector = np.concatenate((base_pos, pelvis_orientation, joint_poses), axis=0)

    return configuration_space_vector

    # ich probiere ob ich das vielleicht besser hinbekomme; aber sollte erstmal so funktioren; 4.7.2024  

def configuration_space_error_function(configuration_space_vector_desired, configuration_space_vector, pelvis_pitch_theta):

    ''' q-errors , configuration space errors'''
    q_left_hip_roll_error  = configuration_space_vector_desired[6]  - configuration_space_vector[6]
    q_left_hip_yaw_error   = configuration_space_vector_desired[7]  - configuration_space_vector[7]
    q_left_hip_pitch_error = configuration_space_vector_desired[8]  - configuration_space_vector[8] + pelvis_pitch_theta-np.deg2rad(0)
    q_left_knee_error      = configuration_space_vector_desired[9]  - configuration_space_vector[9]
    q_left_toe_pitch_error = configuration_space_vector_desired[14] - configuration_space_vector[14] + pelvis_pitch_theta

    q_right_hip_roll_error  = configuration_space_vector_desired[16] - configuration_space_vector[16]
    q_right_hip_yaw_error   = configuration_space_vector_desired[17] - configuration_space_vector[17]
    q_right_hip_pitch_error = configuration_space_vector_desired[18] - configuration_space_vector[18] - pelvis_pitch_theta + np.deg2rad(0)
    q_right_knee_error      = configuration_space_vector_desired[19] - configuration_space_vector[19]
    q_right_toe_pitch_error = configuration_space_vector_desired[24] - configuration_space_vector[24] - pelvis_pitch_theta


    return np.array([
        q_left_hip_roll_error, 
        q_left_hip_yaw_error, 
        q_left_hip_pitch_error, 
        q_left_knee_error, 
        q_left_toe_pitch_error, 
        q_right_hip_roll_error, 
        q_right_hip_yaw_error, 
        q_right_hip_pitch_error, 
        q_right_knee_error, 
        q_right_toe_pitch_error
    ])

def velocity_error_function(q_vels_desired, sensordata):
    ''' q-vel-errors '''    
    q_left_hip_roll_vel_error  = q_vels_desired[0]  - sensordata[model.sensor_adr[mujoco.mj_name2id(model, 19, "left-hip-roll-vel")]]
    q_left_hip_yaw_vel_error   = q_vels_desired[1]  - sensordata[model.sensor_adr[mujoco.mj_name2id(model, 19, "left-hip-yaw-vel")]]
    q_left_hip_pitch_vel_error = q_vels_desired[2]  - sensordata[model.sensor_adr[mujoco.mj_name2id(model, 19, "left-hip-pitch-vel")]]
    q_left_knee_vel_error      = q_vels_desired[3]  - sensordata[model.sensor_adr[mujoco.mj_name2id(model, 19, "left-knee-vel")]]
    q_left_toe_pitch_vel_error = q_vels_desired[4]  - sensordata[model.sensor_adr[mujoco.mj_name2id(model, 19, "left-toe-pitch-vel")]]

    q_right_hip_roll_vel_error  = q_vels_desired[0] - sensordata[model.sensor_adr[mujoco.mj_name2id(model, 19, "right-hip-roll-vel")]]
    q_right_hip_yaw_vel_error   = q_vels_desired[1] - sensordata[model.sensor_adr[mujoco.mj_name2id(model, 19, "ght-hip-yaw-vel")]]
    q_right_hip_pitch_vel_error = q_vels_desired[2] - sensordata[model.sensor_adr[mujoco.mj_name2id(model, 19, "right-hip-pitch-vel")]]
    q_right_knee_vel_error      = q_vels_desired[3] - sensordata[model.sensor_adr[mujoco.mj_name2id(model, 19, "right-knee-vel")]]
    q_right_toe_pitch_vel_error = q_vels_desired[4] - sensordata[model.sensor_adr[mujoco.mj_name2id(model, 19, "right-toe-pitch-vel")]]

    return np.array([
        q_left_hip_roll_vel_error, 
        q_left_hip_yaw_vel_error, 
        q_left_hip_pitch_vel_error, 
        q_left_knee_vel_error, 
        q_left_toe_pitch_vel_error, 
        q_right_hip_roll_vel_error, 
        q_right_hip_yaw_vel_error, 
        q_right_hip_pitch_vel_error,
        q_right_knee_vel_error,
        q_right_toe_pitch_vel_error
    ])


def PD_Controller(gains, configuration_space_error_vector, velocity_errors_vector, torques_left, torques_right):
    ''' Calculates torques for left and right leg ''' 
    #torques_left = np.zeros(5)
    #torques_right = np.zeros(5)
    torques_left[0]  = 600 * configuration_space_error_vector[0] + 60 * velocity_errors_vector[0]
    torques_left[1]  = 600 * configuration_space_error_vector[1] + 60 * velocity_errors_vector[1]
    torques_left[2]  = 600 * configuration_space_error_vector[2] + 60 * velocity_errors_vector[2]
    torques_left[3]  = 600 * configuration_space_error_vector[3] + 60 * velocity_errors_vector[3]
    torques_left[4]  = 400 * configuration_space_error_vector[4] + 40 * velocity_errors_vector[4]

    torques_right[0] = 600 * configuration_space_error_vector[5] + 60 * velocity_errors_vector[5]
    torques_right[1] = 600 * configuration_space_error_vector[6] + 60 * velocity_errors_vector[6]
    torques_right[2] = 600 * configuration_space_error_vector[7] + 60 * velocity_errors_vector[7]
    torques_right[3] = 600 * configuration_space_error_vector[8] + 60 * velocity_errors_vector[8]
    torques_right[4] = 400 * configuration_space_error_vector[9] + 40 * velocity_errors_vector[9]

    return torques_left, torques_right


def gravity_compensation(pelvis_y_world, pose_left_leg_body, pose_right_leg_body, configuration_space_vector, previous_value, kcp, kcpv):
    ''' Gravity Compensation and Center of pressure '''

    change = previous_value - pelvis_y_world
    #center_of_pressure_desired = pelvis_y_world - 1.2 * (0 - pelvis_y_world)# ohne dämpfung
    center_of_pressure_desired = pelvis_y_world - kcp * (0 - pelvis_y_world) + kcpv * change # mit dämpfung
    weight_robot = 48 * (-9.81)      
    Cr = (center_of_pressure_desired - pose_right_leg_body[1]) / (np.absolute(pose_left_leg_body[1]) + np.absolute(pose_right_leg_body[1]))
    Cr = np.clip(Cr, 0, 1)
    force_left = weight_robot * Cr 
    force_right = weight_robot - force_left
    force_right_vector = np.array([0, 0, force_right ,0 ,0 ,0])
    force_left_vector = np.array([0, 0, force_left ,0 ,0 ,0])
    Jacobian_matrix_left = Jacobian_left.J_pos_ypr_fk_base_link_left_toe_motor(configuration_space_vector)
    Jacobian_matrix_right = Jacobian_right.J_pos_ypr_fk_base_link_right_toe_motor(configuration_space_vector)
    Jacobian_matrix_left_transpose = np.transpose(Jacobian_matrix_left)
    Jacobian_matrix_right_transpose = np.transpose(Jacobian_matrix_right)
    torques_gravity_left = Jacobian_matrix_left_transpose @ force_left_vector
    torques_gravity_right = Jacobian_matrix_right_transpose @ force_right_vector
    torques_gravity_left = torques_gravity_left[:-1] # remove last element ( toe roll)
    torques_gravity_right = torques_gravity_right[:-1]

    previous_value = pelvis_y_world

    return torques_gravity_left, torques_gravity_right

def first_order_filter(tau, tau_old, alpha):
    return tau * alpha + tau_old * (1- alpha)


def actuate_motor(torques_left, torques_right):
    ''' Actuate Motors '''
    data.ctrl[0] =  torques_left[0] / 80 # left-hip-roll
    data.ctrl[1] =  torques_left[1] / 50 # left-hip-yaw
    data.ctrl[2] =  torques_left[2] / 16 # left-hip-pitch 
    data.ctrl[3] =  torques_left[3] / 16 # left-knee
    data.ctrl[4] =  -torques_left[4] / 50  / 2 # left-toe-A
    data.ctrl[5] = torques_left[4] / 50 / 2  # left-toe-B 


    data.ctrl[6]  = torques_right[0] / 80 # right-hip-roll 
    data.ctrl[7]  = torques_right[1] / 50 # right-hip-yaw 
    data.ctrl[8]  = torques_right[2] / 16 # right-hip-pitch
    data.ctrl[9]  = torques_right[3] / 16 # right-knee
    data.ctrl[10] = -torques_right[4] / 50  / 2 # right-toe-A
    data.ctrl[11] = torques_right[4] / 50 / 2  # right-toe-B



def simulate_robot(config):

    total_reward = 0

    ''' Gains '''
    # Kp_hip_roll = config["Kp_hip_roll"]
    # Kp_hip_yaw = config["Kp_hip_yaw"]
    # Kp_hip_pitch = config["Kp_hip_pitch"]
    # Kp_knee = config["Kp_knee"]
    # Kp_toe_pitch = config["Kp_toe_pitch"]

    # Kd_hip_roll = config["Kd_hip_roll"]
    # Kd_hip_yaw = config["Kd_hip_yaw"]
    # Kd_hip_pitch = config["Kd_hip_pitch"]
    # Kd_knee = config["Kd_knee"]
    # Kd_toe_pitch = config["Kd_toe_pitch"]
    # gains = np.array([Kp_hip_roll, Kp_hip_yaw, Kp_hip_pitch, Kp_knee, Kp_toe_pitch, Kd_hip_roll, Kd_hip_yaw, Kd_hip_pitch, Kd_knee, Kd_toe_pitch])
    kcp = config["kcp"]
    kcpv = config["kcpv"]

    data = mujoco.MjData(model) # creates the data

    set_start_position()

    # Perform an initial simulation step to update sensor data
    mujoco.mj_forward(model, data) # ich brauche das weil damit die Sensordata vor dem ersten Loop geupdatet ist

    max_torques = np.array([
        1.4 * 80, 
        1.4 * 50,
        12.5* 16,
        12.5* 16,
        0.9 * 50,
        #0.9 * 50,
    ]) 

    sensordata = data.sensordata.copy()

    torques_left = np.zeros(5)
    torques_right = np.zeros(5)
    last_tau_left = np.zeros_like(torques_left)
    last_tau_right = np.zeros_like(torques_right)
    errors_left = np.empty((10_000, 5))
    errors_right = np.empty((10_000, 5))
    pelvis_pos_plot = np.empty((10_000, 3))
    pelvis_orientations = np.empty((10_000, 3))
    pelvis_error_plot = np.empty((10_000, 3))


    tau_left = np.empty((10_000, 5))
    tau_right = np.empty((10_000,5))
    left_world = np.empty((10_000, 6))
    right_world = np.empty((10_000, 6))

    previous_value = 0

        # Kraftparameter ## das gleiche später nochmal für Y-Achse machen
    force = np.array([0, 50, 0])  # Beispielkraft in die z-Richtung in Newton z = -300
    force_duration = 0.1  # Dauer der Kraftanwendung in Sekunden
    force_start_time = 1  # Zeitpunkt der Kraftanwendung in Sekunden

        # simulate and render
    for i in range(10_000):


        ''' Get Configuration Space; Joint Space; Gelenkwinkel; all q '''
        configuration_space_vector = configuration_space_function(sensordata) 

        ''' Get Pelvis Orientation in World Frame '''        
        base_quat = sensordata[model.sensor_adr[mujoco.mj_name2id(model, 19, "base-quat")]:model.sensor_adr[mujoco.mj_name2id(model, 19, "base-quat")] + 4] # w x y z
        pelvis_orientation = qmt.eulerAngles(base_quat) # transform the quaternion to euler angles zyx
        pelvis_yaw_psi = pelvis_orientation[0]   
        pelvis_pitch_theta = pelvis_orientation[1]       
        pelvis_roll_phi = pelvis_orientation[2]
        #pelvis_orientations[i] = np.array([pelvis_yaw_psi, pelvis_pitch_theta, pelvis_roll_phi])

        ''' Get Pelvis Position '''
        pelvis_pos_world = sensordata[model.sensor_adr[mujoco.mj_name2id(model, 19, "base-pos")]:3] # Pelvis Position in World Frame        
        #pelvis_pos_plot[i] = pelvis_pos_world
        ''' Get Leg Pose in World and Body '''
        pose_left_leg_body  = forward_kinematic_left_body.pos_quaternion_fk_base_link_left_toe(configuration_space_vector) # get Pose of left leg, xyzwrpy FK takes configuration space as input
        pose_right_leg_body = forward_kinematic_right_body.pos_quaternion_fk_base_link_right_toe(configuration_space_vector) # get Pose of right leg, FK takes configuration space as input

        pose_left_leg_world  = forward_kinematic_left.pos_quaternion_fk_origin_left_toe(configuration_space_vector) # get Pose of left leg in World Frame
        pose_right_leg_world = forward_kinematic_right.pos_quaternion_fk_origin_right_toe(configuration_space_vector) # get Pose of right leg in World Frame
        

        ''' Rotate Pelvis Position around Z '''
        rot_z = np.array([[np.cos(pelvis_yaw_psi), -np.sin(pelvis_yaw_psi), 0],
                          [np.sin(pelvis_yaw_psi), np.cos(pelvis_yaw_psi), 0],
                          [0,0,1]]) 
        pelvis_pos_world = rot_z.T @ pelvis_pos_world               
        

        ''' Rotate Legs Position around Z '''
        pose_left_leg_world = pose_left_leg_world.at[:3].set((rot_z.T @ np.array(pose_left_leg_world[:3])))      
        pose_right_leg_world = pose_right_leg_world.at[:3].set((rot_z.T @ np.array(pose_right_leg_world[:3])))


        ''' Error Handling '''
        left_z_relative_desired = -0.95 # Left foot z world relative tor pelvis desired  # weiß nicht
        right_z_relative_desired = -0.95 # right foot z world relative tor pelvis desired

        left_x_world_desired =  -0.025 # - pelvis_pos_world[0]  
        right_x_world_desired = -0.025 # -  pelvis_pos_world[0] 
        left_y_world_desired =  0.15  # - pelvis_pos_world[1]
        right_y_world_desired = -0.15 # - pelvis_pos_world[1] 
        left_z_world_desired = (left_z_relative_desired)  + (+ np.arcsin(pelvis_roll_phi) * 0.135)   
        right_z_world_desired = (right_z_relative_desired) + (- np.arcsin(pelvis_roll_phi) * 0.135)   

                 
        left_world_desired_pos = np.array([left_x_world_desired, left_y_world_desired, left_z_world_desired]) #z = -0.95
        right_world_desired_pos = np.array([right_x_world_desired, right_y_world_desired, right_z_world_desired])

        # Umwandlung der Euler-Winkel der Pelvis Orientierung in Rotationsmatrix
        r_world_to_body = R.from_euler('zyx', pelvis_orientation).as_matrix()       
  
        pelvis_des = np.array([(pose_left_leg_world[0] + pose_right_leg_world[0]) / 2, (pose_left_leg_world[1] + pose_right_leg_world[1]) / 2, 0.95])

        pelvis_pos_world[2] = (pelvis_pos_world[2] - (pose_left_leg_world[2] + pose_right_leg_world[2]) / 2) # * 0.5 anstatt / 2 ## = ~0.95

        pelvis_error = pelvis_pos_world - pelvis_des
        #pelvis_error_plot[i] = pelvis_error

        left_body_pos = np.dot(r_world_to_body, (left_world_desired_pos + pelvis_error))
        right_body_pos = np.dot(r_world_to_body, (right_world_desired_pos + pelvis_error))
        #left_world[i,:3] = (left_world_desired_pos + pelvis_pos_world - pelvis_des )
        #right_world[i,:3] = (right_world_desired_pos + pelvis_pos_world  - pelvis_des)
       

        if np.isnan(left_body_pos).any():
            print(left_body_pos)
            break


        pose_left_leg_body_desired = np.zeros(6)        
        pose_left_leg_body_desired[:3] = left_body_pos
        
        pose_right_leg_body_desired = np.zeros(6)
        pose_right_leg_body_desired[:3] = right_body_pos


        ''' Inverse Kinematic '''
        configuration_space_vector_desired = inverse_kinematic.legs_levenberg_marquardt(configuration_space_vector.copy(), pose_left_leg_body_desired, pose_right_leg_body_desired)

        configuration_space_error_vector = configuration_space_error_function(configuration_space_vector_desired, configuration_space_vector, pelvis_pitch_theta) # new
        velocity_errors_vector = velocity_error_function(q_vels_desired, sensordata) # new     

        errors_left[i] = configuration_space_error_vector[:5] # fill array for plot#
        errors_right[i] = configuration_space_error_vector[5:] # fill array for plot

        ''' PD-Controller ''' 
        torques_left, torques_right = PD_Controller(gains, configuration_space_error_vector, velocity_errors_vector, torques_left, torques_right) # new
        
        ''' Gravity Compensation ''' 
        torques_gravity_left, torques_gravity_right = gravity_compensation(pelvis_pos_world[1] ,pose_left_leg_world, pose_right_leg_world, configuration_space_vector, previous_value, kcp, kcpv)

        ''' Combine torques '''
        torques_left += torques_gravity_left
        torques_right += torques_gravity_right

        ''' Clip Torques to real limits '''
        torques_left = np.clip(torques_left, -max_torques, max_torques)
        torques_right = np.clip(torques_right, -max_torques, max_torques)

        ''' Filter torques '''
        torques_left = first_order_filter(torques_left, last_tau_left, 0.5)
        torques_right = first_order_filter(torques_right, last_tau_right, 0.5)

        last_tau_left = torques_left
        last_tau_right = torques_right

        tau_left[i] = torques_left.squeeze()
        tau_right[i] = torques_right.squeeze()

        ''' Actuate motors '''
        actuate_motor(torques_left, torques_right)


        ''' Apply force at the specified time '''
        if force_start_time <= data.time < force_start_time + force_duration:
            # Apply force in the z-direction to the pelvis
            data.xfrc_applied[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base"), :3] = force

        # Reset the force after the duration
        if data.time >= force_start_time + force_duration:
            data.xfrc_applied[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base"), :3] = 0

        
        model.dof_damping[jnt_dofadr] = damping_limit * kd
        sensordata = data.sensordata.copy() # das braucht man anscheinend damit die Werte aktualisiert werden
        for _ in range(5):
            mujoco.mj_step(model, data)

            total_reward -= np.linalg.norm(pelvis_error)

        return {"total_reward": total_reward}
    

    # Define the search space for Ray Tune

search_space = {
    # "Kp_hip_roll": tune.uniform(500.0, 700.0),
    # "Kp_hip_yaw": tune.uniform(500.0, 700.0),
    # "Kp_hip_pitch": tune.uniform(500.0, 700.0),
    # "Kp_knee": tune.uniform(500.0, 700.0),
    # "Kp_toe_pitch": tune.uniform(400.0, 700.0),

    # "Kd_hip_roll": tune.uniform(50.0, 70.0),
    # "Kd_hip_yaw": tune.uniform(50.0, 70.0),
    # "Kd_hip_pitch": tune.uniform(50.0, 70.0),
    # "Kd_knee": tune.uniform(50.0, 70.0),
    # "Kd_toe_pitch": tune.uniform(40.0, 70.0),

    "kcp": tune.grid_search(np.arange(0.5, 1.5, 0.1)),
    "kcpv": tune.grid_search(np.arange(0.5, 1.5, 0.1))
}



# Run Ray Tune
analysis = tune.run(
    simulate_robot,
    config=search_space,
    num_samples=150,  # Adjust based on the available computational resources
    scheduler=ASHAScheduler(metric="total_reward", mode="max"),
)

