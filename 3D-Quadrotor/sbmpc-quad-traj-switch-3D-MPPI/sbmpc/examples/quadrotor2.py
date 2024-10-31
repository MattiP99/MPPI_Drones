import time, os

import jax
import jax.numpy as jnp


os.environ['XLA_FLAGS'] = (
        '--xla_gpu_triton_gemm_any=True '
    )

import matplotlib.pyplot as plt
#plt.rcParams.update({
#    'text.usetex' : True
#})

import sys
sys.path.append('/home/mpiras/MPPI/sbmpc-quad-traj-switch-3D-MPPI/sbmpc')
from sbmpc.model import Model, ModelMjx
from sbmpc.solvers import SbMPC, BaseObjective
from sbmpc.utils.settings import ConfigMPC, ConfigGeneral
from sbmpc.utils.geometry import skew, quat_product, quat2rotm, quat_inverse , rotation_matrix_around_x
import sbmpc.utils.simulation as simulation
import sbmpc.utils.trapezoidal_traj as trapezoidal_traj



#from jax.config import config 
#config.update("jax_debug_nans", True)
#jax.config.update("jax_debug_nans", True)


MODEL = "classic"

input_max = jnp.array([60,24,24,24])
input_min = jnp.array([0,0,0,0])

mass = 2.7
mass_payload = 0.25
cable_length = 1
arm_length = 0.4
gravity = 9.81
inertia = jnp.array([2.45e-2, 2.45e-2, 1.383e-2], dtype=jnp.float32)
inertia_mat = jnp.diag(inertia)

spatial_inertia_mat = jnp.diag(jnp.concatenate([mass*jnp.ones(3, dtype=jnp.float32), inertia]))
spatial_inertia_mat_inv = jnp.linalg.inv(spatial_inertia_mat)

#inertia_slack = 1/12 * (mass) * drone_length**2
#inertia_taut = 1/12 * (mass+ mass_payload) * drone_length**2

e3 = jnp.array([0.,0.,1.],dtype=jnp.float32)
#spatial_inertia_mat = jnp.diag(jnp.concatenate([mass*jnp.ones(3, dtype=jnp.float32), inertia]))
#spatial_inertia_mat_inv = jnp.linalg.inv(spatial_inertia_mat)

input_hover = jnp.array([(mass+mass_payload)*gravity, 0.0, 0.0, 0.0], dtype=jnp.float32)
global tension
global is_slack_dynamic
tension = []
nq = 10

dt = 0.02
acc_vector = []
### DEFINE SYSTEM STATE and INPUTS ####
# X = [x, x_L, dotx, dotx_L ] = [12,1]
#     [3   3 ,   3      3    ]
# x_L, dotx_L= payload position and its velocity
# x, dotx = drone position and its velocity


# U = [F] 
# F = total thrust


# Definition of switching mode, I couls define a variable to do the switch
######## SLACK #############
# Sz , the system dynamics model will transition from Σp to Σz via the transition map Deltap-> z , which is an identity map
        
        
######## TAUT #############
# Sp represents the system state when the cable becomes taut. When the system state reaches Sp, the system will trigger 
# an inelastic collision between the payload and the robot along the cable direction and transition to Σp via the reset
# transition map Deltaz->p


def normalize_angle(angle):
    """
    Normalize an angle to be within the range [-pi, pi].

    Parameters:
        angle (float): The angle to normalize, in radians.

    Returns:
        float: The normalized angle within the range [-pi, pi].
    """
    normalized_angle = jnp.mod(angle + jnp.pi, 2 * jnp.pi) - jnp.pi
    return normalized_angle

def func_slack(state,inputs,csi,csi_dot): 

    ########## SET ORIGINAL EQUATIONS ###############
    print("INSIDE SLACK")
    print("csi SLACK",csi)
    print("state SLACK",state[0:6])
    print("vel SLACK",state[10:16])       

    acc_L =  - jnp.array([0.,0.,gravity])

    # Obtain Quadrotor Force Vector

    # ALREADY IN RADIANTS??
    #quad_force_vector =  F * rotation_matrix_around_x(normalize_angle(state[6])) @ e3
    #quad_force_vector =  F * rotation_matrix_around_x((state[6] * jnp.pi)/180) @ e3
    quat = state[6:10]
    ang_vel = state[16:19]
    ang_vel_quat = jnp.array([0., state[16], state[17], state[18]])
    orientation_mat = quat2rotm(quat)

    #quad_force_vector = F * quat2rotm(quat) @ e3  
    total_force = jnp.array([0., 0., inputs[0]]) - mass*gravity*orientation_mat[2, :]  # transpose + 3rd col = 3rd row

    print("total_force slack",total_force)

    # Solving for Quadrotor Acceleration
    total_torque = inputs[1:4] - skew(ang_vel) @ inertia_mat @ ang_vel
    #acc = quad_force_vector/mass - jnp.array([0.,0.,gravity])
    acc = spatial_inertia_mat_inv @ jnp.concatenate([total_force, total_torque]) 

    #acc_rot = (drone_length * (inputs[0] - inputs[1])) / inertia_slack 
    #pqrdot   = invI @ (total_torque - jnp.reshape(jnp.cross(qd_omega, I @ qd_omega, axisa=0, axisb=0), (3,1)))
    
    print("acc slack",acc)
    print("acc_L slack",acc_L)

    #################### EQUATIONS QUADROTOR #######################

    """
    quat = state[3:7]
    ang_vel = state[10:13]

    orientation_mat = quat2rotm(quat)
    ang_vel_quat = jnp.array([0., state[10], state[11], state[12]])

    total_force = jnp.array([0., 0., inputs[0]]) - mass*gravity*orientation_mat[2, :]  # transpose + 3rd col = 3rd row

    total_torque = 1e-3*inputs[1:4] - skew(ang_vel) @ inertia_mat @ ang_vel  # multiplication by normalization factor

    acc = spatial_inertia_mat_inv @ jnp.concatenate([total_force, total_torque])

    state_dot = jnp.concatenate([state[7:10],
                                 0.5 * quat_product(quat, ang_vel_quat),
                                 orientation_mat @ acc[:3],
                                 acc[3:6]])

    return state_dot
    """

    #################### state_dot ##############################
    state_dot =  jnp.concatenate([state[10:13],
                            state[13:16],
                            0.5 * quat_product(quat, ang_vel_quat),
                            orientation_mat @ acc[:3],
                            acc_L,
                            acc[3:6]])

    print("state_dot slack",state_dot)
    return state_dot

def baumgarte_stabilization(state):
    
    alpha = 50
    beta = alpha*jnp.sqrt(2)

    
    # Calculate the current constraint (distance error)
    diff_pos = state[0:3] - state[3:6]
    current_distance = jnp.linalg.norm(diff_pos)
    C = current_distance - cable_length  # Position constraint violation

    # Velocity constraint violation (time derivative of position constraint)
    relative_vel = state[10:13] - state[13:16]
    C_dot = jnp.dot(diff_pos / current_distance, relative_vel)
    
    # Desired corrective acceleration using Baumgarte stabilization
    correction_term = -2 * alpha * C_dot - beta**2 * C

    # Directional vector from payload to drone
    direction = diff_pos / current_distance 
    return correction_term * direction

def func_taut(state,inputs,csi,csi_dot):

    ########## SET ORIGINAL EQUATIONS ###############
    print("INSIDE TAUT")
    
    print("state taut",state[0:6])
    print("vel taut",state[10:19])
    
    #csi = (state[3:6] - state[0:3])/cable_length

    #csi_dot = (state[10:13] - state[7:10])/cable_length
    
    print("csi taut",csi)
    print("csi_dot taut",csi_dot)
    print("csi NORM taut",jnp.linalg.norm(csi))
    print("csi_dot NORM taut",jnp.linalg.norm(csi_dot))

    
    quat = state[6:10]
    ang_vel = state[16:19]
    ang_vel_quat = jnp.array([0., state[16], state[17], state[18]])
    orientation_mat = quat2rotm(quat)

    #quad_force_vector = F * quat2rotm(quat) @ e3  
    total_force = jnp.array([0., 0., inputs[0]])# - mass*gravity*orientation_mat[2, :]  # transpose + 3rd col = 3rd row
    total_force_world = orientation_mat @ total_force
    print("inputs[0] taut",inputs[0])
    print("inputs[1] taut",inputs[1])
    print("F taut",total_force)
    print("THETA taut",state[6])
    # Obtain Quadrotor Force Vector

    # ALREADY IN RADIANTS??
    
    #quad_force_vector = F * rotation_matrix_around_x(state[6]) @ e3  

    print("quad_force_vector taut",total_force_world)
    quad_centrifugal_f = (mass/jnp.linalg.norm(state[0:3] - state[3:6] )) * jnp.linalg.norm(jnp.array(state[10:13] - state[13:16]))**2
    print("quad_centrifugal_f taut",quad_centrifugal_f)

    #tension_vector = mass_payload * (-csi.reshape(1,3) @ total_force_world + quad_centrifugal_f) * csi.reshape(3,1) / (mass+mass_payload)
    tension_vector =  (mass_payload/ (mass+mass_payload)) * (( jnp.dot(-csi,total_force_world)) - quad_centrifugal_f) * csi 
    #tension_vector = mass_payload * (-csi.reshape(1,3) @ total_force + quad_centrifugal_f) * csi.reshape(3,1) / (mass+mass_payload)
    print("COMPONENT FORCE ALONG CABLE",-csi.reshape(1,3) @ total_force_world)
    print("COMPONENT FORCE ALONG CABLE 2",-csi.reshape(1,3) @ total_force_world * csi.reshape(3,1))
    print("tension_vectortaut",tension_vector)
    # Solving for Load Acceleration
    
    
    acc_L =  -jnp.transpose(tension_vector).reshape(3,) / mass_payload - jnp.array([0.,0.,gravity])
    print("tension_vector taut TRANSPOST",jnp.transpose(tension_vector).reshape(3,))
    print("acc_L taut",acc_L)
    # Solving for Quadrotor Acceleration
    acc = (total_force_world + jnp.transpose(tension_vector).reshape(3,)) / mass - jnp.array([0.,0.,gravity])
    #acc = (total_force + jnp.transpose(tension_vector).reshape(3,)) / mass - jnp.array([0.,0.,gravity])
    print("acc taut",acc)
    acc_L  = acc_L.reshape(3,)
    acc  = acc.reshape(3,)

    # Acceleration constraint with Baumgarte correction
    acc_L = acc_L - baumgarte_stabilization(state)


    #acc_rot = (drone_length * (inputs[0] - inputs[1])) / (inertia_taut)
    total_torque = inputs[1:4] - skew(ang_vel) @ inertia_mat @ ang_vel
    #acc = quad_force_vector/mass - jnp.array([0.,0.,gravity])
    acc_rot = spatial_inertia_mat_inv[3:6,3:6] @ total_torque 
    print("acc_rot taut",acc_rot)
    """
    acc_L = 1/(mass+mass_payload) * ((jnp.dot(csi, jnp.array([0.,0.,inputs[0]])) - mass*cable_length*jnp.dot(csi_dot, csi_dot)) * csi) - jnp.array([0,0,gravity])
    #acc = spatial_inertia_mat_inv @ jnp.concatenate([total_force, total_torque])
    print("acc_L taut",acc_L)
    ################ Problem with cross product in 2 dimensions !!!!!!!!!!!!!!!!!!!!!!!!! ###############
    csi_ddot = 1/(mass*cable_length) *  jnp.cross(csi,jnp.cross(csi,jnp.array([0.,0.,inputs[0]])))  -  jnp.dot(csi_dot,csi_dot) * csi

    ## In this case the first term doesn't have to be trasposed in world frame since theh drone is always straight ###
    #acc = acc_L -  cable_length * csi_ddot
    acc = (jnp.array([0.,0.,inputs[0]]) - 1/(mass+mass_payload) * mass_payload *((jnp.dot(csi, jnp.array([0.,0.,inputs[0]])) - mass*cable_length*jnp.dot(csi_dot, csi_dot)) * csi))/mass - jnp.array([0,0,gravity])
    """
    state_dot = jnp.concatenate([state[10:13],
                            #0.5 * quat_product(quat, ang_vel_quat),
                            state[13:16],
                            0.5 * quat_product(quat, ang_vel_quat),
                            #orientation_mat @ acc[:3],
                            acc[:3],
                            acc_L,
                            acc_rot])
        
    #v_kp1 = state[self.nq:] + self.bt * state_dot[-6:]#self.dynamics(state, inputs)[self.nq:]
        
    return state_dot


def check_distance(state, csi,csi_dot,d,d_dot):
    global is_slack_dynamic
    is_slack = is_slack_dynamic
    print("INSIDE CHECK DISTANCE")

    uav_attach_vector =  state[0:3] - state[3:6]  
    # uav_attach_distance = jnp.linalg.norm(uav_attach_vector)
    uav_attach_distance = jnp.linalg.norm(uav_attach_vector)
    if (is_slack == 1.0) & (jnp.linalg.norm(state[0:3] - state[3:6]) < cable_length - 0.001):# & (d_dot < 0.001):
            print("CHECK 3")
            is_slack = 1.0
        
    elif (is_slack == 1.0) & (jnp.linalg.norm(state[0:3] - state[3:6]) >= cable_length - 0.001):# & (d_dot >=  0.001):
            print("CHECK 4")
            #state = state.at[0:3].set(state[3:6] + (cable_length - 0.001) * csi)
            #state = state.at[3:6].set(state[0:3] + (cable_length - 0.001) * csi)
            
        

            ################ Inelastic Collision ###################
            cable_direction_projmat = csi.reshape((3,1)) @ csi.reshape((1,3))
            vDrone_proj = cable_direction_projmat @ state[10:13]
            vPayload_proj = cable_direction_projmat @ state[13:16]

            v_kp1_parallel_drone = (mass * vDrone_proj + mass_payload * vPayload_proj)/(mass_payload + mass)
            v1 = v_kp1_parallel_drone + state[10:13] - vDrone_proj
            v2 = v_kp1_parallel_drone + state[13:16] - vPayload_proj
            print("v_1 taut",v1)
            print("v_2 taut",v2)
            
            # Mi piacerebbe avere una cosa del genere avere una cosa del genere e se fossimo pií avere una cosa 
            
            state = state.at[10:13].set(v1)
            state = state.at[13:16].set(v2)
            
            #################### State Reset ###############################
            #state = state.at[3:6].set(state[0:3] - (cable_length - 0.001) * csi)
            
            #csi = uav_attach_vector/uav_attach_distance
            d_initial = uav_attach_distance
    
            # Desired final distance
            d_final = cable_length #- 0.001
            delta_d = jnp.abs(d_initial - d_final)
            
            delta_p1 = (mass_payload / (mass + mass_payload)) * delta_d * csi
            #print("delta_p1 taut",delta_p1)
            delta_p2 = (mass / (mass + mass_payload)) * delta_d * csi
            #print("delta_p2 taut",delta_p2)

            #print("drone before reset taut",state[0:3])
            #print("payload before reset taut",state[3:6])
            state = state.at[0:3].set(state[0:3] - delta_p1)
            state = state.at[3:6].set(state[3:6] + delta_p2)
            #print("drone after reset taut",state[0:3])
            #print("payload after reset taut",state[3:6])
            

            is_slack = 0.0
    return state, is_slack


"""
def handle_collision(state,d,d_dot, csi,csi_dot):
    condition = jnp.logical_and(d  >=  cable_length - 0.001,d_dot >= -0.01)
    if condition:
        vel_taut = handle_collision_taut(state,d,d_dot, csi,csi_dot)
        return vel_taut
    else:
        vel_slack = handle_collision_slack(state,d,d_dot, csi,csi_dot)
        return vel_slack
"""
def check_dynamics(state,inputs,csi,csi_dot):
    print("INSIDE CHECK DYNAMICS")
    print("e3 CHECK DYNAMICS",e3 )
    print("csi CHECK DYNAMICS",csi )
    global is_slack_dynamic
    print("is_slack_dynamic CHECK DYNAMICS",is_slack_dynamic )
    
    if  (is_slack_dynamic == 0.0):# & (d_dot >= 0.001):
            result_taut = func_taut(state,inputs,csi,csi_dot)
            quat = state[6:10]
            orientation_mat = quat2rotm(quat)

            #quad_force_vector = F * quat2rotm(quat) @ e3  
            total_force = jnp.array([0., 0., inputs[0]])# - mass*gravity*orientation_mat[2, :]  # transpose + 3rd col = 3rd row
            #total_force =  inputs[0] * orientation_mat[2, :]  # transpose + 3rd col = 3rd row
            print("total_force taut",total_force)
            #print("total_force2 taut",total_force2)
            total_force_world = orientation_mat @ total_force
            print("total_force_world taut",total_force_world)
            #total_force_world2 = orientation_mat @ total_force2
            #print("total_force_world2 taut",total_force_world2)
            total_force_world = total_force_world.reshape(3,)
        
            #################### WHICH CONDITION OF THE TENSION?????????? ##################
            tension_vector = jnp.dot((-total_force_world/mass + ( result_taut[10:13] - (result_taut[13:16] + baumgarte_stabilization(state)))),csi)*csi/(1/mass + 1/mass_payload)
            #tension_vector = jnp.dot((-quad_force_vector/mass + ( result_taut[7:10] - result_taut[10:13])),csi)*(csi)/(1/mass + 1/mass_payload)
            print("tension_vector DYNAMIC", tension_vector)
            #if jnp.dot(tension_vector ,e3) <= 0:
            if jnp.dot(tension_vector,csi) <= 0.0:
                print("INSIDE TAUT TO SLACK CHECK DYNAMICS", tension_vector)
                is_slack_dynamic = 1.0

@jax.jit
def quadrotor_dynamics(state: jnp.array, inputs: jnp.array) -> jnp.array:

    """
    Simple quadrotor dynamics model with CoM placed at the geometric center

    Parameters
    ----------
    state : jnp.array
        state vector [pos (world frame),
                      attitude (unit quaternion [w, x, y, z]),
                      vel (world frame),
                      angular_velocity (body frame)]
    inputs : jnp.array):
        input vector [thrust (along the body-frame z axis), torque (body frame)]
    Returns
    -------
    state_dot :jnp.array
        time derivative of state with given inputs
    """
    #csi = (state[3:6] - state[0:3])/cable_length

    
    csi = (state[0:3] - state[3:6] )/jnp.linalg.norm(state[0:3] - state[3:6] )
    csi_dot = 1/jnp.linalg.norm(state[0:3] - state[3:6] ) * (jnp.diag(jnp.array([1,1,1])) - ((csi).reshape(3,1) @ (csi).reshape(1,3))) @ (state[10:13] - state[13:16]).reshape(3,1)
    csi_dot = csi_dot.reshape(3,)
    
    print("csi dynamics",csi)
    print("csi_dot dynamics",csi_dot)
    print("csi NORM dynamics",jnp.linalg.norm(csi))
    print("csi_dot NORM dynamics",jnp.linalg.norm(csi_dot))
    print("INPUTS DYNAMICS",inputs)
    
    global is_slack_dynamic

    d  = jnp.linalg.norm(state[3:6] - state[0:3])
    #d_dot = ((state[3:6]  - state[0:3]).transpose() @ (state[10:13] - state[7:10]))/((jnp.linalg.norm(state[3:6] - state[0:3])))
    # Compute the dot product of v and dv/dt
    v_dot_dvdt = jnp.dot(state[3:6]  - state[0:3], state[13:16] - state[10:13])
    
    # Compute the derivative of the norm
    d_dot = v_dot_dvdt / d
    print("d_dot dynamic", d_dot)
    
    condition = (is_slack_dynamic == 0.0)
    result_taut = func_taut(state,inputs,csi,csi_dot)
    result_slack = func_slack(state,inputs,csi,csi_dot)
    #print("condition 1 ",condition)
    ######## state_dot ########
    return jax.lax.select(condition,
                          result_taut,
                          result_slack
                         
                         )



class Objective(BaseObjective):
    """ Cost function for the Quadrotor regulation task"""
        
    def compute_state_error(self, state: jnp.array, state_ref : jnp.array) -> jnp.array:
        print("state",state.shape)
        print("state_ref",state_ref.shape)
        pos_err = state[0:3] - state_ref[0:3]
        ang_err = quat_product(quat_inverse(state[6:10]), state_ref[6:10])[1:4]
        vel_err = state[10:13] - state_ref[10:13]
        vel_L_err = state[13:16] - state_ref[13:16]
        att_vel_err = state[16:19] - state_ref[16:19]

        return pos_err,  vel_err , ang_err , att_vel_err , vel_L_err

    def running_cost(self, state: jnp.array, inputs: jnp.array, reference) -> jnp.float32:
        state_ref = reference[:19]
        input_ref = reference[19:]
        
        #pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, state_ref)
        pos_err, vel_err , ang_err, att_vel_err, vel_L_err = self.compute_state_error(state, state_ref)    
        return (150 * pos_err.transpose() @ pos_err +
                60 * ang_err.transpose() @ ang_err +
                50* vel_L_err.transpose() @ vel_L_err +
                50 * vel_err.transpose() @ vel_err +
                10 * att_vel_err.transpose() @  att_vel_err +
                #0.01*(inputs-input_ref).transpose() @ (inputs-input_ref))
                (inputs-input_hover).transpose() @ jnp.diag(jnp.array([1, 0.05, 0.05, 2])) @ (inputs-input_hover))

    def final_cost(self, state, state_ref):
        #pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, reference[:13])
        pos_err, vel_err , ang_err , att_vel_err, vel_L_err = self.compute_state_error(state, reference[:19])
        return (200 * pos_err.transpose() @ pos_err +
                80 * ang_err.transpose() @ ang_err +
                80 * vel_L_err.transpose() @ vel_L_err +
                80 * vel_err.transpose() @ vel_err +
                15 * att_vel_err.transpose() @  att_vel_err)


class Simulation(simulation.Simulator):
    def __init__(self, initial_state, model, controller, is_slack,reference, num_iterations):
        super().__init__(initial_state, model,controller, is_slack,reference, num_iterations)
        
        ############# TRAJECTORY GENERATION ################
        q_des = jnp.array([0.0, 3.0, 5.0, 0.0, 3.0, 4.0,1.0,0.0,0.0,0.0], dtype=jnp.float32)  # hovering position
        
        #self.reference = jnp.zeros((T, x_init.size + input_hover.size),dtype=jnp.float32)
        #calculator = trapezoidal_traj.Trapeizoidal_Trajectory(q_init[3:6], q_des[3:6], 20, self.num_iter + self.controller.horizon + 1)
        #self.reference = calculator.compute_sinusoidal_trajectory_up_lateral3()
        

        ################# FIXED REFERENCE ##################
        x_des = jnp.concatenate([q_des, jnp.zeros(self.model.nv, dtype=jnp.float32)], axis=0)

        reference = jnp.concatenate((x_des, input_hover))
        self.reference = reference 
        
        
    
    
    def update(self):
        print("ITER UPDATE", self.iter)
        print("IS_SLALCK UDPATE BEFORE",self.is_slack)
        global is_slack_dynamic
        if self.iter == 0:
            
            self.is_slack =  is_slack_dynamic
        else:
            is_slack_dynamic = self.is_slack

        if self.is_slack == 1:
            is_slack_number = 1
            
        else:
            is_slack_number = 0
        self.vector_isslack[self.iter] = is_slack_number
        q_des = jnp.array([0.0, 3.0, 5.0, 0.0, 3.0, 4.0,1.0,0.0,0.0,0.0], dtype=jnp.float32)  # hovering position
        x_des = jnp.concatenate([q_des, jnp.zeros(self.model.nv, dtype=jnp.float32)], axis=0)
        # Compute the optimal input sequence
        reference = jnp.concatenate((x_des, input_hover))
        #print("reference:", reference)
        
        # Compute the optimal input sequence
        time_start = time.time_ns()
        
        ##### FIXED REFERENCE #####
        input_sequence = self.controller.compute_control_action(self.current_state_vec(), reference, num_steps=1).block_until_ready()
        
        ##### TAJECTORY REFERENCE #####

        #input_sequence = self.controller.compute_control_action(self.current_state_vec(), self.reference[self.iter:self.iter + self.controller.horizon ,:], num_steps=1).block_until_ready()

        print("computation time: {:.3f} [ms]".format(1e-6 * (time.time_ns() - time_start)))
        ctrl = input_sequence[:self.model.nu]

        self.input_traj[self.iter, :] = ctrl

        """
        # In the original they use
        d = jnp.linalg.norm(self.current_state[3:6] - self.current_state[0:3])
        #d = jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6])
        #d_dot = jnp.linalg.norm((self.current_state[0:3] - self.current_state[3:6])/(jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6])))

        # Consider time partial derivative !!!!!!!
        # In the original they use
        d_dot = (self.current_state[3:6] - self.current_state[0:3] ) @ (self.current_state[9:12] - self.current_state[6:9] )/((jnp.linalg.norm(self.current_state[3:6] - self.current_state[0:3])))
        #d_dot = (self.current_state[0:3] - self.current_state[3:6]) @ (self.current_state[6:9] - self.current_state[9:12] )/((jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6])))

        #csi = (self.current_state[3:6] - self.current_state[0:3])/cable_length
        #csi_dot = (self.current_state[9:12] - self.current_state[6:9])/cable_length

        csi = (self.current_state[3:6] - self.current_state[0:3])/jnp.linalg.norm(self.current_state[3:6] - self.current_state[0:3])
        csi_dot = ((self.current_state[9:12] - self.current_state[6:9])*jnp.linalg.norm(self.current_state[3:6] - self.current_state[0:3])-((self.current_state[3:6] - self.current_state[0:3]) * (self.current_state[3:6] - self.current_state[0:3]) * (1/jnp.linalg.norm(self.current_state[3:6] - self.current_state[0:3])) * (self.current_state[9:12] - self.current_state[6:9])))/(jnp.linalg.norm(self.current_state[3:6] - self.current_state[0:3]))**2
        
        csi2 = (self.current_state[3:6] - self.current_state[0:3])/cable_length
        csi_dot2 = (self.current_state[9:12] - self.current_state[6:9])/cable_length
        """
        print("CURRENT_STATE_DRONE", self.current_state[0:3])
        print("CURRENT_STATE_PAYLOAD", self.current_state[3:6])
        d = jnp.linalg.norm(self.current_state[3:6] - self.current_state[0:3])
       
        # Compute the dot product of v and dv/dt
        v_dot_dvdt = jnp.dot(self.current_state[3:6]  - self.current_state[0:3], self.current_state[13:16] - self.current_state[10:13])
        
        # Compute the derivative of the norm
        d_dot = v_dot_dvdt / d
        
        
        
        # Simulate the dynamics
        csi = (self.current_state[0:3] - self.current_state[3:6] )/jnp.linalg.norm(self.current_state[0:3]- self.current_state[3:6] )
        csi_dot = 1/jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6] ) * (jnp.diag(jnp.array([1,1,1])) - ((csi).reshape(3,1) @ (csi).reshape(1,3))) @ (self.current_state[10:13] - self.current_state[13:16]).reshape(3,1)
        csi_dot = csi_dot.reshape(3,)
        
        print("csi update",csi)
        print("csi_dot update",csi_dot)
        print("csi NORM update",jnp.linalg.norm(csi))
        print("csi_dot NORM update",jnp.linalg.norm(csi_dot))
        #primo = (self.current_state[7:10] - self.current_state[10:13] )*jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6] )
        #print("PRIMO update",primo)
        #secondo = ((self.current_state[0:3] - self.current_state[3:6] ) * (self.current_state[0:3] - self.current_state[3:6] ) * (1/jnp.linalg.norm(self.current_state[0:3]- self.current_state[3:6] )) * (self.current_state[7:10] - self.current_state[10:13] ))
        #print("SECONDO update",secondo)
        #terzo = (jnp.linalg.norm(self.current_state[3:6] - self.current_state[0:3]))**2
        #print("TERZO update",terzo)
        #v_kp1 , pos_payload = handle_collision(self.current_state, d,d_dot, csi,csi_dot)
        check_dynamics(self.current_state, ctrl,csi,csi_dot)
        

        self.current_state   = self.model.integrate(self.current_state, ctrl, dt)
        csi = (self.current_state[0:3] - self.current_state[3:6] )/jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6] )
        csi_dot = 1/jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6] ) * (jnp.diag(jnp.array([1,1,1])) - ((csi).reshape(3,1) @ (csi).reshape(1,3))) @ (self.current_state[10:13] - self.current_state[13:16]).reshape(3,1)
        csi_dot = csi_dot.reshape(3,)
        
        d = jnp.linalg.norm(self.current_state[3:6] - self.current_state[0:3])
       
        # Compute the dot product of v and dv/dt
        v_dot_dvdt = jnp.dot(self.current_state[3:6]  - self.current_state[0:3], self.current_state[13:16] - self.current_state[10:13])
        
        # Compute the derivative of the norm
        d_dot = v_dot_dvdt / d
        self.current_state, self.is_slack = check_distance(self.current_state, csi,csi_dot,d, d_dot)
        is_slack_dynamic = self.is_slack
        
        
        #print("cureent state1",self.current_state)
        # Check for collision and handle it
        print("IS_SLALCK UDPATE AFTER",self.is_slack)
        print("current_state UDPATE AFTER",self.current_state)
        

        # After integration bt = bt + dt or bt = 0 
        #self.bt = bt
        #self.state_traj[self.iter + 1, :] = self.current_state_vec()
        self.state_traj[self.iter + 1, :] = self.current_state_vec()

if __name__ == "__main__":

    mpc_config = ConfigMPC(0.02,
                           50,
                           jnp.array([5,0.2,0.2,0.05]),
                           num_parallel_computations=10000,
                           initial_guess=input_hover)
    gen_config = ConfigGeneral("float32", jax.devices("gpu")[0])

    if MODEL == "classic":
        system = Model(quadrotor_dynamics, 10, 9, 4, [input_min, input_max])
        q_init = jnp.array([0.0, 0.0, 5.0, 0.0, 0.0, 4.05, 1.,0.,0.,0.], dtype=jnp.float32)  # hovering position
        x_init = jnp.concatenate([q_init, jnp.array([0,0,0,0,0,0,0,0,0])])#(system.nv, dtype=jnp.float32)], axis=0)
        state_init = x_init
    elif MODEL == "mjx":
        system = ModelMjx("bitcraze_crazyflie_2/cf2.xml")
        q_init = system.data.qpos
        x_init = jnp.concatenate([q_init, jnp.zeros(system.nv, dtype=jnp.float32)], axis=0)
        state_init = system.data
    else:
        raise ValueError("Model must be either 'classic' or 'mjx'")
    
    #helper  = base_helper.Helper()
    solver = SbMPC(system, Objective(), mpc_config, gen_config)#, helper)

    iterations = 1000
    T = 1000+25+1
    
    #dim = x_init.size + input_hover.size = 17
    
    q_des = jnp.array([0.0, 3.0, 5.0, 0.0, 3.0, 4.0,1.0,0.0,0.0,0.0], dtype=jnp.float32)  # hovering position
    reference = jnp.concatenate((x_init, input_hover))
    #reference = jnp.zeros((T, x_init.size + input_hover.size),dtype=jnp.float32)
    #calculator = trapezoidal_traj.Trapeizoidal_Trajectory(q_init[0:3], q_des[0:3], 10, T)
    #reference = calculator.compute_linear_trajectory()
    #for i in range(T):
    #    reference = reference.at[i,0:3].set(trajectory[i,:])
    #    reference = reference.at[i,3].set(1)
    #    reference = reference.at[i,-4].set(mass*gravity) 

    #reference = jnp.concatenate((x_init, input_hover))

    
    # dummy for jitting
    

    is_slack = 0.0
    if jnp.linalg.norm(q_init[0:3] - q_init[3:6]) < cable_length - 0.001:
        is_slack = 1.0
    # Setup and run the simulation
    is_slack_dynamic = is_slack

    input_sequence = solver.compute_control_action(x_init, reference).block_until_ready()

    sim = Simulation(state_init, system, solver, is_slack, reference, iterations)
    #sim = Simulation(state_init, system, 1000)
    sim.simulate()

    ax = plt.figure().add_subplot(projection='3d')
    
    # Plot x-y-z position of the robot
    ax.plot(sim.state_traj[:, 0], sim.state_traj[:, 1],sim.state_traj[:, 2])
    ax.plot(sim.state_traj[:, 3], sim.state_traj[:, 4],sim.state_traj[:, 5], color  = 'black')
    
    
    plt.figure()
    plt.plot(sim.state_traj[:, 0:3])
    plt.plot(sim.state_traj[:, 3:6])
    plt.legend(["x", "y", "z" , "x_L", "y_L", "z_L"])
    plt.grid()
    

    plt.figure()
    plt.plot(sim.state_traj[:, 6:10])# * jnp.pi/180)
    plt.legend(["q0","q1","q2","q3"])
    plt.grid()
   

    plt.figure()
    plt.plot(sim.state_traj[:, 10:13])
    plt.legend(["x_dot", "y_dot", "z_dot"])
    plt.grid()
    

    plt.figure()
    plt.plot(sim.state_traj[:, 13:16])
    plt.legend(["x_Ldot", "y_Ldot", "z_Ldot"])
    plt.grid()
    

    plt.figure()
    plt.plot(sim.state_traj[:, 16:19])
    plt.legend(["omegax", "omegay", "omegaz"])
    plt.grid()
   

    plt.figure()
    plt.plot(jnp.linalg.norm(sim.state_traj[:, 0:3] - sim.state_traj[:, 3:6], axis = 1))
    plt.legend(["Cable_Length"])
    plt.grid()
    

    plt.figure()
    # Plot the input trajectory
    plt.plot(sim.input_traj)
    plt.legend(["f", "t1", "t2", "t3"])
    
    plt.show()