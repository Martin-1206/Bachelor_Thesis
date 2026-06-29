import os
import mujoco
import mujoco_viewer
import mujoco._render
import numpy as np
import qmt
import kinematics_and_jacobian.J_pos_quaternion_fk_base_link_left_toe_motor_jax as Jacobian_left
import kinematics_and_jacobian.J_pos_quaternion_fk_base_link_right_toe_motor_jax as Jacobian_right
import kinematics_and_jacobian.pos_quaternion_fk_base_link_left_toe_jax as forward_kinematic_left
import kinematics_and_jacobian.pos_quaternion_fk_base_link_right_toe_jax as forward_kinematic_right
import jax
from jax import jit
import jax.numpy as jnp
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler

model = mujoco.MjModel.from_xml_path('/home/martin/Bachelorarbeit/DIGIT/digit-v3_stl/digit-v3.xml') # create the Model from Path
#model = mujoco.MjModel.from_xml_path('/home/martin/Bachelorarbeit/DIGIT/digit-v3_stl/digit-v3_without_body.xml') # create the Model from Path
data = mujoco.MjData(model) # creates the data

''' Operator Input; Führungsgröße; Reference Input '''
pelvis_x_desired = -0.0 # Pelvis Position x desired
pelvis_y_desired = 0 # Pelvis Position y desired
pelvis_z_desired = 0.85#+0.4925 #1.159547 # 0.920218 + 0.259547 #1.2 # Pelvis Position z desired
pelvis_pitch_theta_desired = 0 # Desired Pelvis orientation around y-axis
pelvis_yaw_psi_desired = 0 # Desired Pelvis orientation around z-axis
pelvis_x_linvel_desired = 0 # Pelvis Velocity x desired    # später wenn D-Anteil dazukommt
pelvis_z_linvel_desired = 0 # Pelvis Velocity z desired    # später wenn D-Anteil dazukommt

reference_input_P_Control = np.array([pelvis_x_desired, pelvis_y_desired, pelvis_z_desired, pelvis_pitch_theta_desired, pelvis_yaw_psi_desired])
reference_input_PD_Control = np.append(reference_input_P_Control, (pelvis_x_linvel_desired, pelvis_z_linvel_desired))

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

jnt_dofadr= [model.jnt_dofadr[mujoco.mj_name2id(model,3, name)]for name in actuator_names]

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

def first_order_filter(tau, tau_old, alpha):
    return tau * (1-alpha) + tau_old * alpha

def set_start_position():
    ''' Sets the angle of the joints --> results in a Start Position '''
    data.qpos[2] = 0.91 #20218
    data.qpos[7] = np.deg2rad(18)  # left-hip-roll
    data.qpos[8] = 0.0  # left-hip-yaw
    data.qpos[9] = 0.17 # left-hip-pitch
    data.qpos[10] = 0.04 
    data.qpos[11] = 0.01
    data.qpos[34] = -np.deg2rad(18) # right-hip-roll
    data.qpos[35] = 0.0 # left-hip-yaw
    data.qpos[36] = -0.17 # left-hip-pitch
    data.qpos[37] = -0.04
    data.qpos[38] = 0.01



def get_Jacobian_and_FK_input(sensordata):
    
    base_pos = sensordata[model.sensor_adr[mujoco.mj_name2id(model, 19, "base-pos")]:3] # Sensor_adress 1st to 3rd

    base_quat = sensordata[model.sensor_adr[mujoco.mj_name2id(model, 19, "base-quat")]:model.sensor_adr[mujoco.mj_name2id(model, 19, "base-quat")] + 4] # +4 bcs the quaternion has 4 items
    pelvis_orientation = qmt.eulerAngles(base_quat) # transform the quaternion to euler angles z y x 

    joint_poses = sensordata[7:35]
    
    jacobian_input_vector = np.concatenate((base_pos, pelvis_orientation, joint_poses), axis=0)

    return jacobian_input_vector

def simulate_robot(config):

    total_reward = 0 # damit es nicht vor assignement refernziert wird

    gain_pos_x = config["gain_pos_x"]
    gain_pos_y = config["gain_pos_y"]
    gain_pos_z = config["gain_pos_z"]
    gain_ori_w = config["gain_ori_w"]
    gain_ori_x = config["gain_ori_x"]
    gain_ori_y = config["gain_ori_y"]
    gain_ori_z = config["gain_ori_z"]
    gain_vel_x = config["gain_vel_x"]
    gain_vel_y = config["gain_vel_y"]
    gain_vel_z = config["gain_vel_z"]
    Kcp = config["k_cp"]

    ''' Gain Matrix'''

    gain_matrix_kp = np.array([
        [gain_pos_x, 0, 0, 0, 0, 0, 0],
        [0, gain_pos_y, 0, 0, 0, 0, 0],
        [0, 0, gain_pos_z, 0, 0, 0, 0],
        [0, 0, 0, gain_ori_w, 0, 0, 0],
        [0, 0, 0, 0, gain_ori_x, 0, 0],
        [0, 0, 0, 0, 0, gain_ori_y, 0],
        [0, 0, 0, 0, 0, 0, gain_ori_z]
    ])


    gain_matrix_kd = np.array([
        [gain_vel_x, 0, 0, 0, 0, 0, 0],
        [0, gain_vel_y, 0, 0, 0, 0, 0],
        [0, 0, gain_vel_z, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ])
    
    data = mujoco.MjData(model) # creates the data

    ''' Start Position '''
    set_start_position()

    # Perform an initial simulation step to update sensor data
    mujoco.mj_forward(model, data) # ich brauche das weil damit die Sensordata vor dem ersten Loop geupdatet ist


    left_yaw_psi_list = []
    right_yaw_psi_list = []

    errors = np.empty((10_000, 7))
    tau_left = np.empty((10_000, 6))
    tau_right= np.empty((10_000, 6))

    pos_z = np.empty(10_000)

    max_torques = np.array([
        1.4 * 80, 
        1.4 * 50,
        12.5* 16,
        12.5* 16,
        0.9 * 50,
        0.9 * 50,
    ]) 
    sensordata = data.sensordata.copy()
    # simulate and render
    for i in range(1_000):
            

        ''' Get Position and Orientation '''       
        # Get Orientation
        base_quat = sensordata[model.sensor_adr[mujoco.mj_name2id(model, 19, "base-quat")]:model.sensor_adr[mujoco.mj_name2id(model, 19, "base-quat")] + 4] # w x y z
        pelvis_orientation = qmt.eulerAngles(base_quat) # transform the quaternion to euler angles zyx
        pelvis_yaw_psi = pelvis_orientation[0]   
        pelvis_pitch_theta = pelvis_orientation[1]        
        pelvis_roll_phi = pelvis_orientation[2]
            
        # Get Position with FK
        forward_kinematic_input = get_Jacobian_and_FK_input(sensordata)
        forward_kinematic_left_leg = forward_kinematic_left.pos_quaternion_fk_base_link_left_toe(forward_kinematic_input)
        forward_kinematic_right_leg = forward_kinematic_right.pos_quaternion_fk_base_link_right_toe(forward_kinematic_input)


        forward_kinematic_left_leg = forward_kinematic_left_leg.at[:3].set(qmt.rotate(base_quat, np.array(forward_kinematic_left_leg[:3])))        
        forward_kinematic_right_leg = forward_kinematic_right_leg.at[:3].set(qmt.rotate(base_quat, np.array(forward_kinematic_right_leg[:3])))

        forward_kinematic_left_leg = forward_kinematic_left_leg.at[3:].set(qmt.qmult(base_quat, np.array(forward_kinematic_left_leg[3:])))        
        forward_kinematic_right_leg = forward_kinematic_right_leg.at[3:].set(qmt.qmult(base_quat, np.array(forward_kinematic_right_leg[3:])))


        pelvis_x = (forward_kinematic_left_leg[0] + forward_kinematic_right_leg[0]) / 2     # equation (7)
        pelvis_y = (forward_kinematic_left_leg[1] + forward_kinematic_right_leg[1]) / 2     # equation (7)

        z_left_leg = forward_kinematic_left_leg[2]
        z_right_leg = forward_kinematic_right_leg[2]
        pelvis_z = np.absolute(min(z_left_leg, z_right_leg)) # The height of the pelvis is the foot with the largest z-distance # use min() bcs its negative and we want the one closer to zero

            
        #Test # ich glaube das bringt nix
        #print('pelvis_pitch_theta: ' ,pelvis_pitch_theta)
        #pelvis_x += 0.4925 * np.cos(-np.deg2rad(90) + pelvis_pitch_theta) # test mit lidar als base # das funktioniert einfach nicht richtig da der winkel immer sehr klein ist.
        #pelvis_z += 0.4925 * np.sin(np.deg2rad(90) + pelvis_pitch_theta) # test mit lidar als base # keine ahnung waurm so aber so stimmt es wohl
        

        pos_z[i] = pelvis_z

        # Velocities
        base_linvel = sensordata[model.sensor_adr[mujoco.mj_name2id(model, 19, "base-linvel")]:model.sensor_adr[mujoco.mj_name2id(model, 19, "base-linvel")] + 3] # vel in x y z  ich denke
        pelvis_x_linvel = base_linvel[0]
        pelvis_z_linvel = base_linvel[2]
    

        ''' Calculate errors; Function coming soon '''    
        pelvis_x_error = pelvis_x_desired - pelvis_x
        pelvis_y_error = pelvis_y_desired - pelvis_y
        pelvis_z_error = pelvis_z_desired - pelvis_z          
        pelvis_pitch_theta_error = pelvis_pitch_theta_desired - pelvis_pitch_theta 
        pelvis_yaw_psi_error = pelvis_yaw_psi_desired - pelvis_yaw_psi
            
        # Velocities
        pelvis_x_linvel_error = pelvis_x_linvel_desired - pelvis_x_linvel
        pelvis_z_linvel_error = pelvis_z_linvel_desired - pelvis_z_linvel
            
        # Create Quaternion for Orientation before going in Controller
        pelvis_orientation_error_quat = qmt.quatFromEulerAngles(np.array([0, pelvis_pitch_theta_error , 0])) # only pitch in this branch      
    
        errors_vector = np.array([pelvis_x_error, 0, pelvis_z_error, pelvis_orientation_error_quat[0], pelvis_orientation_error_quat[1], pelvis_orientation_error_quat[2], pelvis_orientation_error_quat[3]]) 

        errors[i] = errors_vector # von Manu

        error_vector_linvelocities = np.array([pelvis_x_linvel_error, 0, pelvis_z_linvel_error, 0, 0, 0, 0])   



        ''' Roll / Yaw Correction '''
        # Modifying x-Rotation
        y_h_distance = 0.091 # 0.135
        y_f = y_h_distance - ((np.absolute(forward_kinematic_left_leg[1]) + np.absolute(forward_kinematic_right_leg[1])) / 2) # equation (14)

        q_left_1cd = np.arctan(y_f / np.absolute(forward_kinematic_left_leg[2])) # equation (15)
        q_right_1cd = np.arctan(y_f / np.absolute(forward_kinematic_right_leg[2]))

        q_left_1yd = np.arctan(pelvis_y_desired / np.absolute(forward_kinematic_left_leg[2])) # equation (16)
        q_right_1yd = np.arctan(pelvis_y_desired / np.absolute(forward_kinematic_right_leg[2]))

        q_left_1d = q_left_1yd + q_left_1cd # equation (17)
        q_right_1d = q_right_1yd + q_right_1cd

        errors_vector_left = errors_vector
        errors_vector_right = errors_vector
            
        errors_vector_left[4] = q_left_1d        
        errors_vector_right[4] = q_right_1d

        # Modifying z-Rotation
        left_yaw_psi_desired = pelvis_yaw_psi_desired
        right_yaw_psi_desired = left_yaw_psi_desired

        left_yaw_psi_error = left_yaw_psi_desired - forward_kinematic_left_leg[6] # equation (18)
        right_yaw_psi_error = right_yaw_psi_desired - forward_kinematic_right_leg[6]

        left_yaw_psi = forward_kinematic_left_leg[6] # für plot erstellt
        right_yaw_psi = forward_kinematic_right_leg[6] # für plot erstellt
        left_yaw_psi_list.append(left_yaw_psi)
        right_yaw_psi_list.append(right_yaw_psi)

        errors_vector_left[6] = left_yaw_psi_error
        errors_vector_right[6] = right_yaw_psi_error


        errors_vector_left = errors_vector_left - np.array([0, 0, 0, 1, 0, 0, 0])
        errors_vector_right = errors_vector_right - np.array([0, 0, 0, 1, 0, 0, 0])            


        ''' Forces, P Control'''
        forces_left = gain_matrix_kp @ errors_vector_left # P-Controller # Im Moment nur ein Force Vektor da die Center of pressure control noch nicht mit drin ist
        forces_right = gain_matrix_kp @ errors_vector_right


        # Mit velocities, damping # PD Controll
        forces_left =  gain_matrix_kp @ errors_vector_left + gain_matrix_kd @ error_vector_linvelocities 
        forces_right =  gain_matrix_kp @ errors_vector_right + gain_matrix_kd @ error_vector_linvelocities



        ''' Center of pressure Control '''
        weight_robot = 48 * (-9.81)
        Kcp = 2
        y_center_pressure = pelvis_y - (Kcp * pelvis_y_error)  # + bcs I calcuate error different than them    # ehrlich nicht sicher in welche Richtung die Vorzeichen wirken
        Cr = (y_center_pressure - forward_kinematic_right_leg[1]) / (np.absolute(forward_kinematic_left_leg[1]) + np.absolute(forward_kinematic_right_leg[1])) # ratio weight distribution # equation (10) # muss schauen wegen Betrag
        #print('cr: ', Cr) 
        force_right_z = weight_robot * Cr # equation (11)
        force_left_z = weight_robot - force_right_z # equation (12)

        #forces_left = np.zeros_like(forces_left) # show gravity compensation
        #forces_right = np.zeros_like(forces_left) # show gravity compensation

        forces_left[2] += force_left_z  # made it + bcs the gravity compensation -9.81
        forces_right[2] += force_right_z # made it + bcs the gravity compensation -9.81

            

        ''' Jacobian '''
        jacobian_leg_input = get_Jacobian_and_FK_input(sensordata) # ist auch schon oben also doppelt, überlegen wie sinnvoll machen
        Jacobian_matrix_left = Jacobian_left.J_pos_quaternion_fk_base_link_left_toe_motor(jacobian_leg_input)        
        Jacobian_matrix_right = Jacobian_right.J_pos_quaternion_fk_base_link_right_toe_motor(jacobian_leg_input)

        Jacobian_left_transpose = np.transpose(Jacobian_matrix_left)
        Jacobian_right_transpose = np.transpose(Jacobian_matrix_right)

        # Multiply Jacobian by forces to get torque
        torques_left  = np.array(Jacobian_left_transpose  @ forces_left)  # get 6 dimensional torque vector
        torques_right = np.array(Jacobian_right_transpose @ forces_right)

        last_tau_left = data.ctrl[:6].copy()
        last_tau_right = data.ctrl[6:12].copy()

        torques_left = first_order_filter(torques_left, last_tau_left, 0.3)
        torques_right = first_order_filter(torques_right, last_tau_right, 0.3)

        torques_left = np.clip(torques_left, -max_torques, max_torques)
        torques_right = np.clip(torques_right, -max_torques, max_torques)


        tau_left[i] = torques_left
        tau_right[i] = torques_right

        ''' Actuate motors '''
        data.ctrl[0] = torques_left[0] / 80 # left-hip-roll
        data.ctrl[1] = torques_left[1] / 50 # left-hip-yaw
        data.ctrl[2] = torques_left[2] / 16 # left-hip-pitch #! PD?
        data.ctrl[3] = torques_left[3] / 16 # left-knee
        data.ctrl[4] = torques_left[4] / 50  / 2 # left-toe-A
        data.ctrl[5] = -torques_left[4] / 50 / 2  # left-toe-Bprint('forward_kinematic_left_leg: ', forward_kinematic_left_leg)


        data.ctrl[6] = torques_right[0] / 80 # right-hip-roll # warum waren die 2 weg?? # es passiert aber nur murks jetzt
        data.ctrl[7] = torques_right[1] / 50 # right-hip-yaw # warum waren die 2 weg??
        data.ctrl[8]  = torques_right[2] / 16 # right-hip-pitch
        data.ctrl[9]  = torques_right[3] / 16 # right-knee
        data.ctrl[10] = torques_right[4] / 50   / 2 # right-toe-A
        data.ctrl[11] = -torques_right[4] / 50 / 2  # right-toe-B      

        model.dof_damping[jnt_dofadr] = damping_limit * kd

        sensordata = data.sensordata.copy()
        for _ in range(5):
            mujoco.mj_step(model, data)


        total_reward -= np.linalg.norm(errors_vector[:3])



    return {"total_reward": total_reward}

# Define the search space for Ray Tune

search_space = {
    "gain_pos_x": tune.uniform(-2500.0, 2500.0),
    "gain_pos_y": tune.uniform(-2500.0, 2500.0),
    "gain_pos_z": tune.uniform(-2500.0, 2500.0),
    "gain_ori_w": tune.uniform(-2500.0, 2500.0),
    "gain_ori_x": tune.uniform(-2500.0, 2500.0),
    "gain_ori_y": tune.uniform(-2500.0, 2500.0),
    "gain_ori_z": tune.uniform(-2500.0, 2500.0),
    "gain_vel_x": tune.uniform(-250.0, 250.0),
    "gain_vel_y": tune.uniform(-250.0, 250.0),
    "gain_vel_z": tune.uniform(-250.0, 250.0),
    "k_cp": tune.uniform(-10.0, 10.0),
}



# Run Ray Tune
analysis = tune.run(
    simulate_robot,
    config=search_space,
    num_samples=50,  # Adjust based on the available computational resources
    scheduler=ASHAScheduler(metric="total_reward", mode="max"),
)

print("Best configuration found: ", analysis.best_config)

# search_space = {
#     "gain_pos_x": tune.grid_search([-1000.0,  0, 1000.0, 2000]),
#     "gain_pos_y": tune.grid_search([-1000.0,  0, 1000.0, 2000]),
#     "gain_pos_z": tune.grid_search([-1000.0,  0, 1000.0, 2000]),
#     "gain_ori_w": tune.grid_search([-1000.0,  0, 1000.0, 2000]),
#     "gain_ori_x": tune.grid_search([-1000.0,  0, 1000.0, 2000]),
#     "gain_ori_y": tune.grid_search([-1000.0,  0, 1000.0, 2000]),
#     "gain_ori_z": tune.grid_search([-1000.0,  0, 1000.0, 2000]),
#     "gain_vel_x": tune.grid_search([-100.0,  0, 100.0, 200]),
#     "gain_vel_y": tune.grid_search([-100.0,  0, 100.0, 200]),
#     "gain_vel_z": tune.grid_search([-100.0,  0, 100.0, 200]),
#     "k_cp": tune.grid_search([-10.0,  0, 10.0]),
# }
# tuner = tune.Tuner(simulate_robot, param_space=search_space)  # ③

# results = tuner.fit()
# print(results.get_best_result(metric="total_reward", mode="max").config)