import time, os

import jax
import jax.numpy as jnp
import numpy as np


os.environ['XLA_FLAGS'] = (
        '--xla_gpu_triton_gemm_any=True '
    )

import matplotlib.pyplot as plt
#plt.rcParams.update({
#    'text.usetex' : True
#})

import sys
sys.path.append('/home/mpiras/MPPI/sbmpc-quad-traj-switch-2D-MPPI/sbmpc')
from sbmpc.model import Model, ModelMjx
from sbmpc.solvers import SbMPC, BaseObjective
from sbmpc.utils.settings import ConfigMPC, ConfigGeneral
from sbmpc.utils.geometry import skew, quat_product, quat2rotm, quat_inverse , rotation_matrix_around_x
import sbmpc.utils.simulation as simulation
import sbmpc.utils.trapezoidal_traj as trapezoidal_traj



#from jax.config import config 
#config.update("jax_debug_nans", True)
#jax.config.update("jax_debug_nans", True)

# Deefine the quadrootor dynamic and the quadrotro variables

MODEL = "classic"

input_max = jnp.array([50,50])
input_min = jnp.array([0,0])

mass = 2.7
mass_payload = 0.25
cable_length = 0.5
arm_length = 0.5
gravity = 9.81
#is_slack = True
#inertia = jnp.array([2.3951e-5, 2.3951e-5, 3.2347e-5], dtype=jnp.float32)
#inertia_mat = jnp.diag(inertia)

inertia_slack = 1/12 * (mass) * (2*arm_length)**2
inertia_taut = 1/12 * (mass+ mass_payload) * (2*arm_length)**2

e3 = jnp.array([0.,0.,1.],dtype=jnp.float32)
#spatial_inertia_mat = jnp.diag(jnp.concatenate([mass*jnp.ones(3, dtype=jnp.float32), inertia]))
#spatial_inertia_mat_inv = jnp.linalg.inv(spatial_inertia_mat)

input_hover = 0.5 * jnp.array([(mass+mass_payload)*gravity, (mass+mass_payload)*gravity], dtype=jnp.float32)
nq = 7

dt = 0.02
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
    print("vel SLACK",state[7:13])       

    acc_L =  - jnp.array([0.,0.,gravity])

    F = inputs[0]+inputs[1]
    # Obtain Quadrotor Force Vector

    # ALREADY IN RADIANTS??
    #quad_force_vector =  F * rotation_matrix_around_x(normalize_angle(state[6])) @ e3
    #quad_force_vector =  F * rotation_matrix_around_x((state[6] * jnp.pi)/180) @ e3
    quad_force_vector = F * rotation_matrix_around_x(state[6]) @ e3  
    print("quad_force_vector slack",quad_force_vector)

    # Solving for Quadrotor Acceleration
    acc = quad_force_vector/mass - jnp.array([0.,0.,gravity])
    acc_rot = (2*arm_length * (inputs[0] - inputs[1])) / inertia_slack 
    
    """  
    acc =  (1/mass)*jnp.array([0.,0.,inputs[0]]) - jnp.array([0.,0.,gravity])
    """
    print("acc slack",acc)
    print("acc_L slack",acc_L)
    print("acc_rot slack",acc_rot)

    #print("inputs",inputs)

    ######## state_dot ########
    state_dot =  jnp.concatenate([state[7:10],
                            #0.5 * quat_product(quat, ang_vel_quat),
                            state[10:13],
                            state[13].reshape(1,),
                            #orientation_mat @ acc[:3],
                            acc,
                            acc_L,
                            acc_rot.reshape(1,)])

    print("state_dot slack",state_dot)
    return state_dot


def func_taut(state,inputs,csi,csi_dot):

    ########## SET ORIGINAL EQUATIONS ###############
    print("INSIDE TAUT")
    
    print("state taut",state[0:6])
    print("vel taut",state[7:13])
    
    #csi = (state[3:6] - state[0:3])/cable_length

    #csi_dot = (state[10:13] - state[7:10])/cable_length
    
    print("csi taut",csi)
    print("csi_dot taut",csi_dot)
    print("csi NORM taut",jnp.linalg.norm(csi))
    print("csi_dot NORM taut",jnp.linalg.norm(csi_dot))

    csi_omega = jnp.cross(csi, csi_dot)
    

    F =  inputs[0]+inputs[1]
    print("inputs[0] taut",inputs[0])
    print("inputs[1] taut",inputs[1])
    print("F taut",F)
    print("THETA taut",state[6])
    # Obtain Quadrotor Force Vector
    # ALREADY IN RADIANTS??
    #quad_force_vector = F * rotation_matrix_around_x(normalize_angle(state[6])) @ e3  
    #quad_force_vector = F * rotation_matrix_around_x((state[6] * jnp.pi)/180) @ e3  
    quad_force_vector = F * rotation_matrix_around_x(state[6]) @ e3  

    print("quad_force_vector taut",quad_force_vector)
    quad_centrifugal_f = mass * cable_length * (csi_omega @ csi_omega)

    print("quad_centrifugal_f taut",quad_centrifugal_f)

    tension_vector = mass_payload * (-csi @ quad_force_vector + quad_centrifugal_f) * csi / (mass+mass_payload)
    print("tension_vectortaut",tension_vector)
    # Solving for Load Acceleration
    
    
    #acc_L =  jnp.transpose(tension_vector).reshape(3,) / mass_payload - jnp.array([0.,0.,gravity])
    acc_L = -tension_vector / mass_payload - jnp.array([0.,0.,gravity])
    
    print("acc_L taut",acc_L)
    # Solving for Quadrotor Acceleration
    #acc = (quad_force_vector - jnp.transpose(tension_vector).reshape(3,)) / mass - jnp.array([0.,0.,gravity])
    acc = (quad_force_vector + tension_vector) / mass - jnp.array([0.,0.,gravity])
    print("acc taut",acc)
    acc_L  = acc_L.reshape(3,)
    acc  = acc.reshape(3,)
    #acc_rot = (drone_length * (inputs[0] - inputs[1])) / (inertia_taut)
    acc_rot = (2*arm_length * (inputs[0] - inputs[1])) / (inertia_slack)
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
    state_dot = jnp.concatenate([state[7:10],
                            #0.5 * quat_product(quat, ang_vel_quat),
                            state[10:13],
                            state[13].reshape(1,),
                            #orientation_mat @ acc[:3],
                            acc,
                            acc_L,
                            acc_rot.reshape(1,)])
        
    #v_kp1 = state[self.nq:] + self.bt * state_dot[-6:]#self.dynamics(state, inputs)[self.nq:]
        
    return state_dot

def handle_collision_taut(state,d,d_dot, csi,csi_dot):
        cable_direction_projmat = csi.reshape((3,1)) @ csi.reshape((1,3))
        vDrone_proj = cable_direction_projmat @ state[7:10]
        vPayload_proj = cable_direction_projmat @ state[10:13]

        v_kp1_parallel_drone = (mass * vDrone_proj + mass_payload * vPayload_proj)/(mass_payload + mass)
        v1 = v_kp1_parallel_drone + state[7:10] - vDrone_proj
        v2 = v_kp1_parallel_drone + state[10:13] - vPayload_proj

        print("v1 taut",v1)
        print("v2 taut",v2)
        """
        v_kp1_parallel_drone = (1/(mass+mass_payload)) * ( mass* (csi @ jnp.transpose(csi)) * state[6:9] + mass_payload * (csi @ jnp.transpose(csi)) * state[9:12])
        

       
        print("v_kp1_parallel_drone taut",v_kp1_parallel_drone)
        v_kp1_parallel_payload =  v_kp1_parallel_drone

        v_orthogoal_drone = state[6:9] - state[6:9]*(csi @ csi)
        v_orthogoal_payload = state[9:12] - state[6:9]*(csi @ csi)
        
        v =   v_kp1_parallel_drone + v_orthogoal_drone #+ proj_of_v_on_v_orth 
        v_payload =  v_kp1_parallel_payload + v_orthogoal_payload #+ proj_of_v_payload_on_v_payload_orth 
        """           
            
        v_kp1 = state[nq:nq+7]
        v_kp1 =  v_kp1.at[0:3].set(v1) 

        #v_kp1 = v_kp1 + v_kp1.at[3:6].set(v_payload)
        v_kp1 = v_kp1.at[3:6].set(v2) 
        #state   = state.at[3:6].set(state[0:3] + cable_length * csi)
        print("v_kp1 taut",v_kp1)
        
        

        #return  v_kp1, state[3:6]
        return  v_kp1

def handle_collision_slack( state,d,d_dot, csi,csi_dot):
    v_kp1 = state[nq:nq+7]
    print("v_kp1 slack",v_kp1)
    print("d handle slack",d)
    print("d_dot handle slack",d_dot)
    return  v_kp1


def handle_collision(state,d,d_dot, csi,csi_dot,is_slack):
    #condition = jnp.logical_and(d  >  cable_length - 0.001, d_dot > -0.001)
    condition = (d_dot  >  0.001) & (is_slack == 0.0) # & (d_dot > -0.001):   

    #condition = (d  > cable_length - 0.001)
    result_taut = handle_collision_taut(state,d,d_dot, csi,csi_dot)
    result_slack = handle_collision_slack(state,d,d_dot, csi,csi_dot)

    print("condition 2 ",condition)
    print("result_taut 2",result_taut)
    print("result_slack 2",result_slack)
    return jax.lax.select(condition,
                          #### Cable Taut ####
                         result_taut,
                         result_slack
                         )

"""
def func_reset_state_taut(state,csi,is_slack):
    uav_attach_vector =  state[3:6] - state[0:3] 
    uav_attach_distance = jnp.linalg.norm(uav_attach_vector)
    
    csi = uav_attach_vector/uav_attach_distance
    state = state.at[0:3].set(state[3:6] + (cable_length-0.001) * csi)

    condition = (uav_attach_distance > cable_length - 0.001)
    
    is_slack_result_reset_taut = func_reset_is_slack_taut(state,csi,is_slack)
    is_slack_result_reset_slack = func_reset_is_slack_slack(state,csi,is_slack)
    return_is_slack = jax.lax.select(condition,is_slack_result_reset_taut, is_slack_result_reset_slack)
    state_and_is_slack = jnp.concatenate([state, return_is_slack.reshape(1,)],dtype=jnp.float32)
    return state_and_is_slack

def func_reset_state_slack(state,csi,is_slack):
    uav_attach_vector =  state[3:6] - state[0:3] 
    uav_attach_distance = jnp.linalg.norm(uav_attach_vector)
    condition = (uav_attach_distance > cable_length - 0.001)

    is_slack_result_reset_taut = func_reset_is_slack_taut(state,csi,is_slack)
    is_slack_result_reset_slack = func_reset_is_slack_slack(state,csi,is_slack)
    return_is_slack = jax.lax.select(condition,is_slack_result_reset_taut, is_slack_result_reset_slack)
    state_and_is_slack = jnp.concatenate([state, return_is_slack.reshape(1,)],dtype=jnp.float32)
    
    return state_and_is_slack

def func_reset_is_slack_taut(state,csi,is_slack):
    return is_slack

def func_reset_is_slack_slack(state,csi,is_slack):
    return 1.0 - is_slack

def check_distance(state, csi,csi_dot,is_slack):
    
    condition = (is_slack == 0.0)
    state_result_reset_taut = func_reset_state_taut(state,csi,is_slack)
    state_result_reset_slack = func_reset_state_slack(state,csi,is_slack)
    
    
    #print("condition 1 ",condition)
    ######## state_dot ########
    return_state_and_is_slack =  jax.lax.select(condition,state_result_reset_taut, state_result_reset_slack) 
    return return_state_and_is_slack[:-1], return_state_and_is_slack[-1]
"""
def check_distance(state, csi,csi_dot,is_slack):
    uav_attach_vector =  state[0:3] - state[3:6]  
    uav_attach_distance = jnp.linalg.norm(uav_attach_vector)
    if is_slack == 0.0:
        if uav_attach_distance > cable_length - 0.001:
            csi = uav_attach_vector/uav_attach_distance
            state = state.at[0:3].set(state[3:6] + (cable_length - 0.001) * csi)
            #state = state.at[3:6].set(state[0:3] + cable_length * csi)
            
        else:
            is_slack = 1.0
        
    else:
        if uav_attach_distance <= cable_length - 0.001:
            is_slack = 1.0
        else:
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

    csi_dot = (state[7:10] - state[10:13] )/cable_length
    csi = (state[0:3] - state[3:6] )/jnp.linalg.norm(state[0:3] - state[3:6] )
    #csi_dot = (((state[10:13] - state[7:10])*jnp.linalg.norm(state[3:6] - state[0:3]))-((state[3:6] - state[0:3]) * (state[3:6] - state[0:3]) * (1/jnp.linalg.norm(state[3:6] - state[0:3])) * (state[10:13] - state[7:10])))/(jnp.linalg.norm(state[3:6] - state[0:3]))**2
    #csi_dot = csi_dot/jnp.linalg.norm(csi_dot)
    #csi_dot = (state[10:13] - state[7:10])/jnp.linalg.norm(state[10:13] - state[7:10])
    print("csi dynamics",csi)
    print("csi_dot dynamics",csi_dot)
    print("csi NORM dynamics",jnp.linalg.norm(csi))
    print("csi_dot NORM dynamics",jnp.linalg.norm(csi_dot))
    print("INPUTS DYNAMICS",inputs)
    

    d = jnp.linalg.norm(state[0:3]  - state[3:6])
    d_dot = ((state[3:6] - state[0:3]) @ (state[10:13] - state[7:10]))/((jnp.linalg.norm(state[3:6] - state[0:3]  )))
    # Inputs are said to be total force and total torque but I already have this computation in order 
    # to consider as inputs the f forces and moments in the body frame
    
    
    
    
    ######## state_dot ########
    #return jnp.select([d - cable_length  < 0.001,  #### Cable Slack ####
    #                     jnp.logical_and(d - cable_length >= 0.001 , d_dot >= -0.01)],
    #                      #### Cable Taut ####
    #                     [func_slack(state,inputs),
    #                     func_taut(state,inputs,csi,csi_dot)]
    #                     )
    #condition = jnp.logical_and(d  > cable_length - 0.001, d_dot > -0.001)
    #condition = (d  >  cable_length - 0.001) #&  (d_dot > 0.001)
    condition = (d  > cable_length - 0.001)
    result_taut = func_taut(state,inputs,csi,csi_dot)
    result_slack = func_slack(state,inputs,csi,csi_dot)
    #print("condition 1 ",condition)
    ######## state_dot ########
    return jax.lax.select(condition,
                          result_taut,
                          result_slack
    )
 
"""                        

###### LINEAR TRAJECTORY POINT FOR PAYLOAD AS REFERENCE #####
# Standard deviation = [0.2 , 0.2]
# For the other examples is [0.1,0.1]
class Objective(BaseObjective):
    #Cost function for the Quadrotor regulation task
        
    def compute_state_error(self, state: jnp.array, state_ref : jnp.array) -> jnp.array:
        print("state",state.shape)
        print("state_ref",state_ref.shape)
        pos_err = state[0:3] - state_ref[0:3]
        pos_L__err = state[3:6] - state_ref[3:6]
        att_vel_err = state[13] - state_ref[13]
        vel_err = state[7:10] - state_ref[7:10]
        vel_L_err = state[10:13] - state_ref[10:13]
        ang_err = state[6] - state_ref[6]

        return pos_err, pos_L__err,  vel_err , ang_err , att_vel_err , vel_L_err

    def running_cost(self, state: jnp.array, inputs: jnp.array, reference) -> jnp.float32:
        state_ref = reference[:14]
        input_ref = reference[14:]
        
        #pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, state_ref)
        pos_err, pos_L_err , vel_err , ang_err, att_vel_err, vel_L_err = self.compute_state_error(state, state_ref)       
        return (#80 * pos_err.transpose() @ pos_err +
                95 * pos_L_err.transpose() @ pos_L_err +
                0.5 * att_vel_err *  att_vel_err +
                95 * vel_L_err.transpose() @ vel_L_err +
                #35 * vel_err.transpose() @ vel_err +
                10 * ang_err * ang_err +
                1*(inputs-input_ref).transpose() @ (inputs-input_ref))
                #1*(inputs-input_hover).transpose() @ (inputs-input_hover))

    def final_cost(self, state, reference):
        #pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, reference[:13])
        pos_err, pos_L_err,  vel_err , ang_err , att_vel_err , vel_L_err = self.compute_state_error(state, reference[:14])
        return (#150 * pos_err.transpose() @ pos_err +
                170 * pos_L_err.transpose() @ pos_L_err +
                25 * ang_err * ang_err +
                170 * vel_L_err.transpose() @ vel_L_err +
                #55 * vel_err.transpose() @ vel_err +
                1 * att_vel_err *  att_vel_err)
"""    
"""
###### CONSTANT POINT FOR PAYLOAD AS REFERENCE #####
class Objective(BaseObjective):
    #Cost function for the Quadrotor regulation task
        
    def compute_state_error(self, state: jnp.array, state_ref : jnp.array) -> jnp.array:
        print("state",state.shape)
        print("state_ref",state_ref.shape)
        pos_err = state[0:3] - state_ref[0:3]
        pos_L__err = state[3:6] - state_ref[3:6]
        att_vel_err = state[13] - state_ref[13]
        vel_err = state[7:10] - state_ref[7:10]
        vel_L_err = state[10:13] - state_ref[10:13]
        ang_err = state[6] - state_ref[6]

        return pos_err, pos_L__err,  vel_err , ang_err , att_vel_err , vel_L_err

    def running_cost(self, state: jnp.array, inputs: jnp.array, reference) -> jnp.float32:
        state_ref = reference[:14]
        input_ref = reference[14:]
        
        #pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, state_ref)
        pos_err, pos_L_err , vel_err , ang_err, att_vel_err, vel_L_err = self.compute_state_error(state, state_ref)       
        return (#80 * pos_err.transpose() @ pos_err +
                100 * pos_L_err.transpose() @ pos_L_err +
                0.5 * att_vel_err *  att_vel_err +
                35 * vel_L_err.transpose() @ vel_L_err +
                #35 * vel_err.transpose() @ vel_err +
                0.1 * ang_err * ang_err +
                #1*(inputs-input_ref).transpose() @ (inputs-input_ref))
                1*(inputs-input_hover).transpose() @ (inputs-input_hover))

    def final_cost(self, state, reference):
        #pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, reference[:13])
        pos_err, pos_L_err,  vel_err , ang_err , att_vel_err , vel_L_err = self.compute_state_error(state, reference[:14])
        return (#150 * pos_err.transpose() @ pos_err +
                200 * pos_L_err.transpose() @ pos_L_err +
                5 * ang_err * ang_err +
                75 * vel_L_err.transpose() @ vel_L_err +
                #55 * vel_err.transpose() @ vel_err +
                1 * att_vel_err *  att_vel_err)
"""  
"""
###### HOVERING FOR PAYLOAD AS REFERENCE #####
class Objective(BaseObjective):
    #Cost function for the Quadrotor regulation task
        
    def compute_state_error(self, state: jnp.array, state_ref : jnp.array) -> jnp.array:
        print("state",state.shape)
        print("state_ref",state_ref.shape)
        pos_err = state[0:3] - state_ref[0:3]
        pos_L__err = state[3:6] - state_ref[3:6]
        att_vel_err = state[13] - state_ref[13]
        vel_err = state[7:10] - state_ref[7:10]
        vel_L_err = state[10:13] - state_ref[10:13]
        ang_err = state[6] - state_ref[6]

        return pos_err, pos_L__err,  vel_err , ang_err , att_vel_err , vel_L_err

    def running_cost(self, state: jnp.array, inputs: jnp.array, reference) -> jnp.float32:
        state_ref = reference[:14]
        input_ref = reference[14:]
        
        #pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, state_ref)
        pos_err, pos_L_err , vel_err , ang_err, att_vel_err, vel_L_err = self.compute_state_error(state, state_ref)       
        return (#80 * pos_err.transpose() @ pos_err +
                80 * pos_L_err.transpose() @ pos_L_err +
                0.5 * att_vel_err *  att_vel_err +
                10 * vel_L_err.transpose() @ vel_L_err +
                #35 * vel_err.transpose() @ vel_err +
                0.1 * ang_err * ang_err +
                #1*(inputs-input_ref).transpose() @ (inputs-input_ref))
                1*(inputs-input_hover).transpose() @ (inputs-input_hover))

    def final_cost(self, state, reference):
        #pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, reference[:13])
        pos_err, pos_L_err,  vel_err , ang_err , att_vel_err , vel_L_err = self.compute_state_error(state, reference[:14])
        return (#150 * pos_err.transpose() @ pos_err +
                150 * pos_L_err.transpose() @ pos_L_err +
                5 * ang_err * ang_err +
                25 * vel_L_err.transpose() @ vel_L_err +
                #55 * vel_err.transpose() @ vel_err +
                1 * att_vel_err *  att_vel_err)
"""   

# SQUARE TRAJECTORY
class Objective(BaseObjective):
    #Cost function for the Quadrotor regulation task
        
    def compute_state_error(self, state: jnp.array, state_ref : jnp.array) -> jnp.array:
        print("state",state.shape)
        print("state_ref",state_ref.shape)
        pos_err = state[0:3] - state_ref[0:3]
        pos_L__err = state[3:6] - state_ref[3:6]
        att_vel_err = state[13] - state_ref[13]
        vel_err = state[7:10] - state_ref[7:10]
        vel_L_err = state[10:13] - state_ref[10:13]
        ang_err = state[6] - state_ref[6]

        return pos_err, pos_L__err,  vel_err , ang_err , att_vel_err , vel_L_err

    def running_cost(self, state: jnp.array, inputs: jnp.array, reference) -> jnp.float32:
        state_ref = reference[:14]
        input_ref = reference[14:]
        
        #pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, state_ref)
        pos_err, pos_L_err , vel_err , ang_err, att_vel_err, vel_L_err = self.compute_state_error(state, state_ref)       
        return (100 * pos_err.transpose() @ pos_err +
                #80 * pos_L_err.transpose() @ pos_L_err +
                0.5 * att_vel_err *  att_vel_err +
                15 * vel_L_err.transpose() @ vel_L_err +
                15 * vel_err.transpose() @ vel_err +
                0.1 * ang_err * ang_err +
                1*(inputs-input_ref).transpose() @ (inputs-input_ref))
                #1*(inputs-input_hover).transpose() @ (inputs-input_hover))

    def final_cost(self, state, reference):
        #pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, reference[:13])
        pos_err, pos_L__err,  vel_err , ang_err , att_vel_err , vel_L_err = self.compute_state_error(state, reference[:14])
        return (200 * pos_err.transpose() @ pos_err +
                5 * ang_err * ang_err +
                200 * vel_L_err.transpose() @ vel_L_err +
                200 * vel_err.transpose() @ vel_err +
                1 * att_vel_err *  att_vel_err)


"""
##### WEIGHTS FOR LINEAR TRAJECTORY #######

class Objective(BaseObjective):
    #Cost function for the Quadrotor regulation task
        
    def compute_state_error(self, state: jnp.array, state_ref : jnp.array) -> jnp.array:
        print("state",state.shape)
        print("state_ref",state_ref.shape)
        pos_err = state[0:3] - state_ref[0:3]
        pos_L__err = state[3:6] - state_ref[3:6]
        att_vel_err = state[13] - state_ref[13]
        vel_err = state[7:10] - state_ref[7:10]
        vel_L_err = state[10:13] - state_ref[10:13]
        ang_err = state[6] - state_ref[6]

        return pos_err, pos_L__err,  vel_err , ang_err , att_vel_err , vel_L_err

    def running_cost(self, state: jnp.array, inputs: jnp.array, reference) -> jnp.float32:
        state_ref = reference[:14]
        input_ref = reference[14:]
        
        #pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, state_ref)
        pos_err, pos_L_err , vel_err , ang_err, att_vel_err, vel_L_err = self.compute_state_error(state, state_ref)       
        return (80 * pos_err.transpose() @ pos_err +
                0.5 * att_vel_err *  att_vel_err +
                35 * vel_L_err.transpose() @ vel_L_err +
                35 * vel_err.transpose() @ vel_err +
                0.1 * ang_err * ang_err +
                1*(inputs-input_ref).transpose() @ (inputs-input_ref))
                #1*(inputs-input_hover).transpose() @ (inputs-input_hover))

    def final_cost(self, state, reference):
        #pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, reference[:13])
        pos_err, pos_L_err,  vel_err , ang_err , att_vel_err , vel_L_err = self.compute_state_error(state, reference[:14])
        return (150 * pos_err.transpose() @ pos_err +
                #150 * pos_L_err.transpose() @ pos_L_err +
                5 * ang_err * ang_err +
                55 * vel_L_err.transpose() @ vel_L_err +
                55 * vel_err.transpose() @ vel_err +
                1 * att_vel_err *  att_vel_err)

"""
class Simulation(simulation.Simulator):
    def __init__(self, initial_state, model, controller, is_slack,reference, num_iterations):
        super().__init__(initial_state, model,controller, is_slack,reference, num_iterations)
        
        ############# TRAJECTORY GENERATION ################
        q_des = jnp.array([0.0, 4.0, 9.0, 0.0, 4.0, 8.5,0.0], dtype=jnp.float32)  # hovering position
        self.reference = jnp.zeros((T, x_init.size + input_hover.size),dtype=jnp.float32)
        calculator = trapezoidal_traj.Trapeizoidal_Trajectory(q_init[3:6], q_des[3:6], 30, self.num_iter + self.controller.horizon + 1)
        self.reference = calculator.compute_square_trajectory()
        

        ################# FIXED REFERENCE ##################
        #x_des = jnp.concatenate([q_des, jnp.zeros(self.model.nv, dtype=jnp.float32)], axis=0)

        #reference = jnp.concatenate((x_des, input_hover))
        #self.reference = reference
        
        
        
    
    
    def update(self):
        q_des = jnp.array([0.0, 4.0, 9.0, 0.0, 4.0, 8.5,0.0], dtype=jnp.float32)  # hovering position
        x_des = jnp.concatenate([q_des, jnp.zeros(self.model.nv, dtype=jnp.float32)], axis=0)
        # Compute the optimal input sequence
        reference = jnp.concatenate((x_des, input_hover))
        #print("reference:", reference)
        
        # Compute the optimal input sequence
        time_start = time.time_ns()
        
        ##### FIXED REFERENCE #####
        #input_sequence = self.controller.compute_control_action(self.current_state_vec(), reference, num_steps=1).block_until_ready()
        
        ##### TAJECTORY REFERENCE #####

        input_sequence = self.controller.compute_control_action(self.current_state_vec(), self.reference[self.iter:self.iter + self.controller.horizon ,:], num_steps=1).block_until_ready()
        #print("REFERENCE",self.reference[self.iter:self.iter + self.controller.horizon ,:])
        print("computation time: {:.3f} [ms]".format(1e-6 * (time.time_ns() - time_start)))
        ctrl = input_sequence[:self.model.nu]

        self.input_traj[self.iter, :] = ctrl

        
        print("CURRENT_STATE_DRONE", self.current_state[0:3])
        print("CURRENT_STATE_PAYLOAD", self.current_state[3:6])
        d = jnp.linalg.norm(self.current_state[3:6] - self.current_state[0:3]  )
        d_dot = ((self.current_state[3:6] - self.current_state[0:3]) @ (self.current_state[10:13] - self.current_state[7:10]))/(jnp.linalg.norm(self.current_state[3:6] - (self.current_state[0:3])))
        print("d update",d)
        print("d_dot update",d_dot)
        
        #csi = (self.current_state[3:6] - self.current_state[0:3])/cable_length
        csi_dot = (self.current_state[7:10]  - self.current_state[10:13] )/cable_length
        # Simulate the dynamics
        csi = (self.current_state[0:3] - self.current_state[3:6] )/jnp.linalg.norm(self.current_state[0:3]- self.current_state[3:6] )

        print(" in update FUNCTION")
        print("d",d)
        print("d_dot",d_dot)
        print("csi",csi)
        print("csi_dot",csi_dot)
        
        # Simulate the dynamics
        
        v_kp1  = handle_collision(self.current_state, d,d_dot, csi,csi_dot,self.is_slack)
        self.current_state = self.current_state.at[nq:nq+7].set(v_kp1)
        #self.current_state = self.current_state.at[3:6].set(state_L)
        print("current state2",self.current_state)

        self.current_state   = self.model.integrate(self.current_state, ctrl, self.controller.dt)
        
        self.current_state, self.is_slack = check_distance(self.current_state, csi,csi_dot,self.is_slack)
        print("IS_SLALCK UDPATE",self.is_slack)
        print("current state1",self.current_state)
        # Check for collision and handle it
        
        self.state_traj[self.iter + 1, :] = self.current_state_vec()

if __name__ == "__main__":

    mpc_config = ConfigMPC(0.02,
                           25,
                           jnp.array([0.1,0.1]),
                           num_parallel_computations=10000,
                           initial_guess=input_hover)
    gen_config = ConfigGeneral("float32", jax.devices("gpu")[0])

    if MODEL == "classic":
        system = Model(quadrotor_dynamics, 7, 7, 2, [input_min, input_max])
        q_init = jnp.array([0.0, 0.0, 5.0, 0.0, 0.0, 4.5, 0], dtype=jnp.float32)  # hovering position
        #q_init = jnp.array([0.0, 0.0, 9.0, 0.0, 0.0, 9.1, 0], dtype=jnp.float32)  # hovering position
        x_init = jnp.concatenate([q_init, jnp.array([0,0,0,0,0,0,0])])#(system.nv, dtype=jnp.float32)], axis=0)
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
    T = iterations+25+1
    
    #dim = x_init.size + input_hover.size = 17
    
    q_des = jnp.array([0.0, 4.0, 9.0, 0.0, 4.0, 8.5,0.0], dtype=jnp.float32)  # hovering position
    #reference = jnp.concatenate([q_des, jnp.zeros(nq, dtype=jnp.float32)], axis=0) 
    #reference = jnp.concatenate([reference, jnp.array([(mass+mass_payload)*gravity/2, (mass+mass_payload)*gravity/2], dtype=jnp.float32)], axis=0)
    #reference = jnp.concatenate((x_init, input_hover))
    reference = jnp.zeros((T, x_init.size + input_hover.size),dtype=jnp.float32)
    calculator = trapezoidal_traj.Trapeizoidal_Trajectory(q_init[3:6], q_des[3:6], 30, T)
    reference = calculator.compute_square_trajectory()
    

    reference = jnp.concatenate((x_init, input_hover))
    
    
    # dummy for jitting
    input_sequence = solver.compute_control_action(x_init, reference).block_until_ready()

    is_slack = 0.0
    if jnp.linalg.norm(q_init[0:3] - q_init[3:6]) < cable_length:
        is_slack = 1.0
    # Setup and run the simulation
    sim = Simulation(state_init, system, solver, is_slack, reference, iterations)
    #sim = Simulation(state_init, system, 1000)
    sim.simulate()

    ############################################### PLOTTING ###########################################################
    import csv
    import seaborn as sns
    import pandas as pd
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection


    title = ['x', 'y', 'z', 'x_L', 'y_L', 'z_L', 'theta', 'x_dot', 'y_dot', 'z_dot', 'x_Ldot', 'y_Ldot', 'z_Ldot', 'theta_dot', 'Cable_Length', 'u1','u2']
    x = sim.state_traj[:, 0]
    y = sim.state_traj[:, 1]
    z = sim.state_traj[:, 2]
    x_L = sim.state_traj[:, 3]
    y_L = sim.state_traj[:, 4]
    z_L = sim.state_traj[:, 5]
    theta = sim.state_traj[:, 6]
    x_dot = sim.state_traj[:, 7]
    y_dot = sim.state_traj[:, 8]
    z_dot = sim.state_traj[:, 9]
    x_Ldot = sim.state_traj[:, 10]
    y_Ldot = sim.state_traj[:, 11]
    z_Ldot = sim.state_traj[:, 12]
    theta_dot = sim.state_traj[:, 13]
    Cable_Length = jnp.linalg.norm(sim.state_traj[:, 0:3] - sim.state_traj[:, 3:6], axis = 1)
    u1 = sim.input_traj[:,0]
    u2 = sim.input_traj[:,1]
    
    elements = jnp.concatenate([x.reshape(iterations+1,1),y.reshape(iterations+1,1),z.reshape(iterations+1,1),
                                x_L.reshape(iterations+1,1), y_L.reshape(iterations+1,1), z_L.reshape(iterations+1,1),
                                theta.reshape(iterations+1,1),
                                x_dot.reshape(iterations+1,1), y_dot.reshape(iterations+1,1), z_dot.reshape(iterations+1,1),
                                x_Ldot.reshape(iterations+1,1), y_Ldot.reshape(iterations+1,1), z_Ldot.reshape(iterations+1,1),
                                theta_dot.reshape(iterations+1,1),
                                Cable_Length.reshape(iterations+1,1),
                                jnp.concatenate([jnp.zeros(1,),u1],axis = 0).reshape(iterations+1,1),
                                jnp.concatenate([jnp.zeros(1,),u2],axis = 0).reshape(iterations+1,1)], axis = 1)
    
    print("ELEMENTS",elements.shape)

    list = []
    for i in range(iterations+1):
            elements_list =  elements[i,:].tolist()
            list.append(elements_list)
    
    data = [
        title,
        list
        ]


    with open('file_data1.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data[0])
        for x in data[1]:
            writer.writerow(x)

    filename = r'file_data1.csv'
    df = pd.read_csv(filename)

    x = df['x'].dropna()
    y = df['y'].dropna()
    z = df['z'].dropna()

    x_L = df['x_L'].dropna()
    y_L = df['y_L'].dropna()
    z_L = df['z_L'].dropna()

    theta = df['theta'].dropna()

    x_dot = df['x_dot'].dropna()
    y_dot = df['y_dot'].dropna()
    z_dot = df['z_dot'].dropna()

    x_Ldot = df['x_Ldot'].dropna()
    y_Ldot = df['y_Ldot'].dropna()
    z_Ldot = df['z_Ldot'].dropna()

    theta_dot = df['theta_dot'].dropna()
    Cable_Length = df['Cable_Length'].dropna()
    u1 = df['u1'].dropna()
    u2 = df['u2'].dropna()

    #bodacious colors
    colors=sns.color_palette("rocket",17) #personal fav 
    colors2=sns.color_palette("crest",17) #also nice

    #Ram's colors, if desired
    seshadri = ['#c3121e', '#0348a1', '#ffb01c', '#027608', '#0193b0', '#9c5300', '#949c01', '#7104b5']
    #            0sangre,   1neptune,  2pumpkin,  3clover,   4denim,    5cocoa,    6cumin,    7berry

    iterations_array = jnp.arange(0,iterations+1)
    #plot
    
    
    #Prepare multipanel plot 
    fig = plt.figure(1, figsize=(5, 5))
    gs = gridspec.GridSpec(4,4)
    gs.update(wspace=0.2, hspace=0.25)

    #xtr_subsplot= fig.add_subplot(gs[0:4,0:3])
    xtr_subsplot= fig.add_subplot(gs[0:4,0:3])

    plt.plot( iterations_array,x, linestyle='-', label='drone_x_position', color=colors[0], mfc='w', markersize=4) # plot data
    plt.plot(iterations_array,x_L, linestyle='-', label='payload_x_position', color=colors2[0], mfc='w', markersize=4) # plot data
    plt.plot(iterations_array,y, linestyle='-', label='drone_y_position', color=colors[16], mfc='w', markersize=4) # plot data
    plt.plot( iterations_array,y_L,linestyle='-',  label='payload_y_position', color=colors2[8], mfc='w', markersize=4) # plot data
    plt.plot( iterations_array,z,linestyle='-',  label='drone_z_position', color=colors[8], mfc='w', markersize=4) # plot data
    plt.plot( iterations_array,z_L, linestyle='-', label='payload_z_position', color=colors2[16], mfc='w', markersize=4) # plot data
   
    #plot params
    plt.xlim([0,iterations + 50])
    plt.ylim([-5,15])
    plt.minorticks_on()
    plt.tick_params(direction='in',right=True, top=True)
    plt.tick_params(labelsize=10)
    plt.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)
    #xticks = np.arange(0, 1e4,10)
    yticks = np.arange(0,15.1,1)

    plt.tick_params(direction='in',which='minor', length=5, bottom=True, top=True, left=True, right=True)
    plt.tick_params(direction='in',which='major', length=10, bottom=True, top=True, left=True, right=True)
    #plt.xticks(xticks)
    plt.yticks(yticks)

    plt.xlabel('Iteration', fontsize=14) 
    plt.ylabel('Position [m]',fontsize=14)  # label the y axis


    plt.legend(fontsize=14)  # add the legend (will default to 'best' location)
    
    #plot text and line on top of figure
    plt.axvline(x=10, linestyle='dotted', color='black')
    plt.text(20.5, 50, 'First Zoom', rotation=90)

    plt.axvline(x=600, linestyle='dotted', color='purple')
    plt.text(610.5, 250, 'Second Zoom', rotation=90)
    

    #generate second panel
    xtr_subsplot = fig.add_subplot(gs[0:2,3:4])
    plt.plot( iterations_array[0:100],x[0:100], linestyle='-', label='drone_x_position', color=colors[0], mfc='w', markersize=2) # plot data
    plt.plot(iterations_array[0:100],x_L[0:100], linestyle='-', label='payload_x_position', color=colors2[0], mfc='w', markersize=2) # plot data
    plt.plot(iterations_array[0:100],y[0:100], linestyle='-', label='drone_y_position', color=colors[8], mfc='w', markersize=2) # plot data
    plt.plot( iterations_array[0:100],y_L[0:100],linestyle='-',  label='payload_y_position', color=colors2[8], mfc='w', markersize=2) # plot data
    plt.plot( iterations_array[0:100],z[0:100],linestyle='-',  label='drone_z_position', color=colors[8], mfc='w', markersize=2) # plot data
    plt.plot( iterations_array[0:100],z_L[0:100], linestyle='-', label='payload_z_position', color=colors2[16], mfc='w', markersize=2) # plot data

    #Define tick parameters
    xticks2 = np.arange(0,100,10)
    yticks2 = np.arange(-5.1,15.1,1)
    plt.minorticks_on()
    plt.tick_params(direction='in',which='minor', length=5, 
                    bottom=True, top=True, left=True, right=True)
    plt.tick_params(direction='in',which='major', length=10, 
                    bottom=True, top=True, left=True, right=True)
    plt.tick_params(labelbottom=False, labeltop=False, 
                    labelright=True, labelleft=False)

    plt.xticks(xticks2)
    plt.yticks(yticks2)
    plt.legend()

    
    #generate third panel
    xtr_subsplot = fig.add_subplot(gs[2:4,3:4])
    plt.plot( iterations_array[590:650],x[590:650], linestyle='-', label='drone_x_position', color=colors[0], mfc='w', markersize=2) # plot data
    plt.plot(iterations_array[590:650],x_L[590:650], linestyle='-', label='payload_x_position', color=colors2[0], mfc='w', markersize=2) # plot data
    plt.plot(iterations_array[590:650],y[590:650], linestyle='-', label='drone_y_position', color=colors[8], mfc='w', markersize=2) # plot data
    plt.plot( iterations_array[590:650],y_L[590:650],linestyle='-',  label='payload_y_position', color=colors2[8], mfc='w', markersize=2) # plot data
    plt.plot( iterations_array[590:650],z[590:650],linestyle='-',  label='drone_z_position', color=colors[8], mfc='w', markersize=2) # plot data
    plt.plot( iterations_array[590:650],z_L[590:650], linestyle='-', label='payload_z_position', color=colors2[16], mfc='w', markersize=2) # plot data

    #Define tick parameters
    xticks3 = np.arange(590,651,5)
    yticks3 = np.arange(-5.1,15.1,1)
    plt.minorticks_on()
    plt.tick_params(direction='in',which='minor', length=5, 
                    bottom=True, top=True, left=True, right=True)
    plt.tick_params(direction='in',which='major', length=10, 
                    bottom=True, top=True, left=True, right=True)
    plt.tick_params(labelbottom=False, labeltop=False, 
                    labelright=True, labelleft=False)

    plt.xticks(xticks3)
    plt.yticks(yticks3)
    plt.legend()
    

    ########### VELOCITY PLOT ##########################################

    #Prepare multipanel plot 
    fig2 = plt.figure(2, figsize=(5, 5))
    gs2 = gridspec.GridSpec(4,4)
    gs2.update(wspace=0.2, hspace=0.25)

    xtr_subsplot2= fig2.add_subplot(gs2[0:4,0:2])

    plt.plot( iterations_array,x_dot, linestyle='-', label='drone_x_velocity', color=colors[0], mfc='w', markersize=4) # plot data
    #plt.plot(iterations_array,x_Ldot, linestyle='-', label='payload_x_velocity', color=colors2[0], mfc='w', markersize=4) # plot data
    plt.plot(iterations_array,y_dot, linestyle='-', label='drone_y_velocity', color=colors[16], mfc='w', markersize=4) # plot data
    #plt.plot( iterations_array,y_Ldot,linestyle='-',  label='payload_y_velocity', color=colors2[8], mfc='w', markersize=4) # plot data
    plt.plot( iterations_array,z_dot,linestyle='-',  label='drone_z_velocity', color=colors[8], mfc='w', markersize=4) # plot data
    #plt.plot( iterations_array,z_Ldot, linestyle='-', label='payload_z_velocity', color=colors2[16], mfc='w', markersize=4) # plot data
   
    #plot params
    #plt.xlim([0,iterations + 50])
    #plt.ylim([-100,100])
    #xticks = np.arange(0, 1e4,10)
    xticks = np.arange(0,1000.5,100)
    yticks = np.arange(-100,100,10)
    plt.minorticks_on()
    plt.tick_params(direction='in',which='minor', length=5, 
                    bottom=True, top=True, left=True, right=True)
    plt.tick_params(direction='in',which='major', length=10, 
                    bottom=True, top=True, left=True, right=True)
    plt.tick_params(labelbottom=False, labeltop=False, 
                    labelright=False, labelleft=True)
    

    plt.tick_params(direction='in',which='minor', length=5, bottom=True, top=True, left=True, right=True)
    plt.tick_params(direction='in',which='major', length=10, bottom=True, top=True, left=True, right=True)
    #plt.xticks(xticks)
    
    plt.xticks(xticks)

    plt.yticks(yticks)

    plt.xlabel('Iteration', fontsize=14) 
    plt.ylabel('Drone Velocity [m/s]',fontsize=14)  # label the y axis


    plt.legend()  # add the legend (will default to 'best' location)
    #plot text and line on top of figure
    
    xtr_subsplot2 = fig2.add_subplot(gs2[0:4,2:4])
    #plt.plot( iterations_array,x_dot, linestyle='-', label='drone_x_velocity', color=colors[0], mfc='w', markersize=4) # plot data
    plt.plot(iterations_array,x_Ldot, linestyle='-', label='payload_x_velocity', color=colors2[0], mfc='w', markersize=4) # plot data
    #plt.plot(iterations_array,y_dot, linestyle='-', label='drone_y_velocity', color=colors[16], mfc='w', markersize=4) # plot data
    plt.plot( iterations_array,y_Ldot,linestyle='-',  label='payload_y_velocity', color=colors2[8], mfc='w', markersize=4) # plot data
    #plt.plot( iterations_array,z_dot,linestyle='-',  label='drone_z_velocity', color=colors[8], mfc='w', markersize=4) # plot data
    plt.plot( iterations_array,z_Ldot, linestyle='-', label='payload_z_velocity', color=colors2[16], mfc='w', markersize=4) # plot data
   
    #Define tick parameters
    xticks2 = np.arange(0,1000.5,100)
    yticks2 = np.arange(-100,100,10)
    plt.minorticks_on()
    plt.tick_params(direction='in',which='minor', length=5, 
                    bottom=True, top=True, left=True, right=True)
    plt.tick_params(direction='in',which='major', length=10, 
                    bottom=True, top=True, left=True, right=True)
    plt.tick_params(labelbottom=False, labeltop=False, 
                    labelright=True, labelleft=False)

    plt.xticks(xticks2)
    plt.yticks(yticks2)
    plt.xlabel('Iteration', fontsize=14) 
    plt.ylabel('Payload Velocity [m/s]',fontsize=14)  # label the y axis

    plt.legend()



    ######################################### THETA PLOT #################################################################

    #Prepare multipanel plot 
    fig3 = plt.figure(3, figsize=(5, 5))
    gs3 = gridspec.GridSpec(4,4)
    gs3.update(wspace=0.2, hspace=0.25)

    xtr_subsplot3= fig3.add_subplot(gs2[0:4,0:2])

    plt.plot( iterations_array,theta, linestyle='-', label='theta', color=colors2[16], mfc='w', markersize=4) # plot data
    
    
   
    #plot params
    #plt.xlim([0,iterations + 50])
    #plt.ylim([-100,100])
    #xticks = np.arange(0, 1e4,10)
    xticks = np.arange(0,1000.5,100)
    yticks = np.arange(-6.28,6.28,0.4)
    plt.minorticks_on()
    plt.tick_params(direction='in',which='minor', length=5, 
                    bottom=True, top=True, left=True, right=True)
    plt.tick_params(direction='in',which='major', length=10, 
                    bottom=True, top=True, left=True, right=True)
    plt.tick_params(labelbottom=False, labeltop=False, 
                    labelright=False, labelleft=True)
    

    plt.tick_params(direction='in',which='minor', length=5, bottom=True, top=True, left=True, right=True)
    plt.tick_params(direction='in',which='major', length=10, bottom=True, top=True, left=True, right=True)
    #plt.xticks(xticks)
    
    plt.xticks(xticks)

    plt.yticks(yticks)

    plt.xlabel('Iteration', fontsize=14) 
    plt.ylabel('Theta [rad]',fontsize=14)  # label the y axis


    plt.legend()  # add the legend (will default to 'best' location)
    #plot text and line on top of figure
    
    xtr_subsplot3 = fig3.add_subplot(gs2[0:4,2:4])
    plt.plot(iterations_array,theta_dot, linestyle='-', label='theta_dot', color=colors[8], mfc='w', markersize=4) # plot data
   
    #Define tick parameters
    xticks2 = np.arange(0,1000.5,100)
    yticks2 = np.arange(-10.1,10.1,1)
    plt.minorticks_on()
    plt.tick_params(direction='in',which='minor', length=5, 
                    bottom=True, top=True, left=True, right=True)
    plt.tick_params(direction='in',which='major', length=10, 
                    bottom=True, top=True, left=True, right=True)
    plt.tick_params(labelbottom=False, labeltop=False, 
                    labelright=True, labelleft=False)

    plt.xticks(xticks2)
    plt.yticks(yticks2)
    plt.xlabel('Iterations', fontsize=14) 
    plt.ylabel('Theta Dot [rad/s]',fontsize=14)  # label the y axis

    plt.legend()


################################################  CABLE LENGTH PLOT ##########################################################

    #plot
    plt.figure(4, figsize=(5, 5))
    plt.plot( iterations_array,Cable_Length, linestyle='-', label='Cable Length', color=colors2[16], mfc='w', markersize=4) # plot data

    #plt.plot(x_fit, ffit(x_fit), linestyle='-', marker='None', label='fit', color=colors[0], markerfacecolor='white', markersize=8) # plot data

    #plot params
    xticks = np.arange(0,1000.5,100)
    yticks = np.arange(0,0.75,0.1)
    plt.minorticks_on()

    plt.tick_params(direction='in',which='minor', length=5, bottom=True, top=True, left=True, right=True)
    plt.tick_params(direction='in',which='major', length=10, bottom=True, top=True, left=True, right=True)
    plt.xticks(xticks)
    plt.yticks(yticks)


    #plt.text(1,325, f'y={Decimal(coefs[3]):.4f}x$^3$+{Decimal(coefs[2]):.2f}x$^2$+{Decimal(coefs[1]):.2f}x+{Decimal(coefs[0]):.1f}',fontsize =13)


    plt.xlabel('Iterations', fontsize=14) 
    plt.ylabel('Cable Length [m]',fontsize=14)  # label the y axis


    plt.legend(fontsize=14)  # add the legend (will default to 'best' location)

    ############################################## INPUT PLOT ############################################################

    #plot
    plt.figure(5, figsize=(5, 5))
    plt.plot( iterations_array,u1, linestyle='-', label='u1', color=colors2[16], mfc='w', markersize=4) # plot data
    plt.plot( iterations_array,u2, linestyle='-', label='u2', color=colors[8], mfc='w', markersize=4) # plot data

    #plt.plot(x_fit, ffit(x_fit), linestyle='-', marker='None', label='fit', color=colors[0], markerfacecolor='white', markersize=8) # plot data

    #plot params
    xticks = np.arange(0,1000.5,100)
    yticks = np.arange(-50,50,5)
    plt.minorticks_on()

    plt.tick_params(direction='in',which='minor', length=5, bottom=True, top=True, left=True, right=True)
    plt.tick_params(direction='in',which='major', length=10, bottom=True, top=True, left=True, right=True)
    plt.xticks(xticks)
    plt.yticks(yticks)


    #plt.text(1,325, f'y={Decimal(coefs[3]):.4f}x$^3$+{Decimal(coefs[2]):.2f}x$^2$+{Decimal(coefs[1]):.2f}x+{Decimal(coefs[0]):.1f}',fontsize =13)


    plt.xlabel('Iterations', fontsize=14) 
    plt.ylabel('Inputs',fontsize=14)  # label the y axis


    plt.legend(fontsize=14)  # add the legend (will default to 'best' location)

    ############################################## TORQUE PLOT ############################################################

    #plot
    plt.figure(6, figsize=(5, 5))
    plt.plot( iterations_array, 2*arm_length*(u1 - u2), linestyle='-', label='torque', color=colors2[16], mfc='w', markersize=4) # plot data

    #plt.plot(x_fit, ffit(x_fit), linestyle='-', marker='None', label='fit', color=colors[0], markerfacecolor='white', markersize=8) # plot data

    #plot params
    xticks = np.arange(0,1000.5,100)
    yticks = np.arange(-50,50,5)
    plt.minorticks_on()

    plt.tick_params(direction='in',which='minor', length=5, bottom=True, top=True, left=True, right=True)
    plt.tick_params(direction='in',which='major', length=10, bottom=True, top=True, left=True, right=True)
    plt.xticks(xticks)
    plt.yticks(yticks)


    #plt.text(1,325, f'y={Decimal(coefs[3]):.4f}x$^3$+{Decimal(coefs[2]):.2f}x$^2$+{Decimal(coefs[1]):.2f}x+{Decimal(coefs[0]):.1f}',fontsize =13)


    plt.xlabel('Iterations', fontsize=14) 
    plt.ylabel('Inputs torque [N m]',fontsize=14)  # label the y axis


    plt.legend(fontsize=14)  # add the legend (will default to 'best' location)

    ############################################## INPUT FORCE ############################################################

    #plot
    plt.figure(7, figsize=(5, 5))
    plt.plot( iterations_array, u1 + u2, linestyle='-', label='Force', color=colors2[16], mfc='w', markersize=4) # plot data
    

    #plt.plot(x_fit, ffit(x_fit), linestyle='-', marker='None', label='fit', color=colors[0], markerfacecolor='white', markersize=8) # plot data

    #plot params
    xticks = np.arange(0,1000.5,100)
    yticks = np.arange(-50,50,5)
    plt.minorticks_on()

    plt.tick_params(direction='in',which='minor', length=5, bottom=True, top=True, left=True, right=True)
    plt.tick_params(direction='in',which='major', length=10, bottom=True, top=True, left=True, right=True)
    plt.xticks(xticks)
    plt.yticks(yticks)


    #plt.text(1,325, f'y={Decimal(coefs[3]):.4f}x$^3$+{Decimal(coefs[2]):.2f}x$^2$+{Decimal(coefs[1]):.2f}x+{Decimal(coefs[0]):.1f}',fontsize =13)


    plt.xlabel('Iterations', fontsize=14) 
    plt.ylabel('Inputs Force [N]',fontsize=14)  # label the y axis


    plt.legend(fontsize=14)  # add the legend (will default to 'best' location)


########################################## 3D ###############################################################

    ax = plt.figure().add_subplot(projection='3d')
    # Plot x-y-z position of the robot
    ax.plot(sim.state_traj[:, 0], sim.state_traj[:, 1], sim.state_traj[:, 2])
    ax.plot(sim.state_traj[:, 3], sim.state_traj[:, 4], sim.state_traj[:, 5])
    
    # Vertices of the square
    square_vertices = np.array([
        [0, 0.0, 5.5],  # First corner
        [0, 1.0, 6.5],   # Second corner
        [0, 0.0, 7.5],    # Third corner
        [0, -1.0, 6.5],
        [0, 0.0, 5.5]    # Fourth corner
    ])

    # Extract the x, y, and z coordinates for plotting
    x = square_vertices[:, 0]
    y = square_vertices[:, 1]
    z = square_vertices[:, 2]

    # Plot the contour of the square by connecting vertices
    ax.plot(x, y, z, color='cyan', linewidth=1)
    
    plt.legend(["drone_position", "payload_position"])
    plt.grid()

    plt.show()



########################################## ANIMATION 3D ###############################################################

"""
import vpython
from vpython import *

#x1, y1, z1, x2, y2, z2 = np.load('..\\data\\3Dpen.npy')
ball1 = vpython.sphere(color = color.green, radius = 0.05, make_trail=True, retain=20)
ball2 = vpython.sphere(color = color.green, radius = 0.1, make_trail=True, retain=20)
rod1 = cylinder(pos=vector(0,0,0), axis=vector(0,0,0), radius=0.05)

rod2 = cylinder(pos=vector(0,0,0), axis=vector(0,0,0), radius=0.05)
rod3 = cylinder(pos=vector(0,0,0), axis=vector(0,0,0), radius=0.05)
# NB in the simulation y is the vertical direction
#FLOOR
base  = box(pos=vector(0,-4.25,0),axis=vector(1,0,0),
            size=vector(10,0.5,10) )

# SHADOWS
s1 = cylinder(pos=vector(0,-3.99,0),axis=vector(0,-0.1,0), radius=0.8, color=color.gray(luminance=0.7))
s2 = cylinder(pos=vector(0,-3.99,0),axis=vector(0,-0.1,0), radius=0.8, color=color.gray(luminance=0.7))

print('Start')

theta = np.asarray(theta)
for i in range(0,iterations+1):
    rate(50)
    ball1.pos = vector(x[i], z[i], y[i]) + vector(x_dot[i], z_dot[i], y_dot[i])*dt
    ball2.pos = vector(x_L[i], z_L[i], y_L[i]) + vector(x_Ldot[i], z_Ldot[i], y_Ldot[i])*dt
    #rod1.axis = vector(x[i], z[i], y[i])
    rod1.pos = vector(x[i], z[i], y[i]) + vector(x_dot[i], z_dot[i], y_dot[i])*dt
    rod1.axis = vector(x[i]-x_L[i], z[i]-z_L[i], y[i]-y_L[i]) + vector(x_dot[i], z_dot[i], y_dot[i])*dt
    

    rod2.pos = vector(x[i], z[i], y[i])+ vector(x_dot[i], z_dot[i], y_dot[i])*dt
    rod2.axis = vector(x[i], z[i]+drone_length*np.sin(theta), y[i]+drone_length*np.cos(theta)) + vector(theta_dot, 0, 0)*dt
    rod3.pos = vector(x[i], z[i], y[i])+ vector(x_dot[i], z_dot[i], y_dot[i])*dt
    rod3.axis = vector(x[i], z[i]-drone_length*np.sin(theta), y[i]-drone_length*np.cos(theta)) + vector(theta_dot, 0, 0)*dt
"""

import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
"""
xmin = -3
xmax = 3

ymin = -3
ymax = 3


fig = plt.figure()
# Plot x-y-z position of the robot


fig, ax = plt.subplots()
ax = plt.axes(xlim=(-3,3), ylim=(0, 600))
line1, = ax.plot([], [], lw=2)
line2, = ax.plot([], [], lw=2)
line3, = ax.plot([], [], lw=2)


def animate(i): 
    center = np.array([y[:i+1], z[:i+1]]) 
    center_payload = np.array([y_L[:i+1], z_L[:i+1]]) 
    #rod1.axis = vector(x[i], z[i], y[i])
    
    line1.set_data([center[0] - center_payload[0], center[1] - center_payload[1]]) 
    propeller1 = np.array([y[:i+1]+drone_length*np.cos(theta[i+1]), z[:i+1]+drone_length*np.sin(theta[i+1])]) 
    propeller2 = np.array([y[:i+1]-drone_length*np.cos(theta[i+1]) , z[:i+1]-drone_length*np.sin(theta[i+1])]) 
    line2.set_data(center[0] + propeller1[0],center[1] + propeller1[1])
    line3.set_data(center[0] + propeller2[0],center[1] + propeller2[1])
    return line1, line2, line3
 
ani = animation.FuncAnimation(fig, animate, frames=1000,
                              interval=1, blit=True, repeat=False)
ffmpeg_writer = animation.FFMpegFileWriter(fps = 30)
ani.save('/home/mpiras/MPPI/sbmpc-quad-traj-switch-2D-TEST/sbmpc/quadrotor.mp4', writer=ffmpeg_writer)
"""

from mpl_toolkits.mplot3d import Axes3D
import imageio



# Parameters for the output
output_folder = './frames_Hovering/'
output_video = 'drone_position_Hovering.mp4'

zoom_padding = 2.0  # You can adjust this to be more or less zoomed in
# Drone dimensions
propeller_size = 0.1  # Size of the propeller markers

# Create a 3D plot and save each frame
#for i, pos in enumerate(sim.state_traj[:, :3]):
for i, (drone_pos, payload_pos, angle) in enumerate(zip(sim.state_traj[:, 0:3], sim.state_traj[:, 3:6], sim.state_traj[:, 6])):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all previous points in blue
    #ax.scatter(sim.state_traj[:i+1, 0], sim.state_traj[:i+1, 1], sim.state_traj[:i+1, 2], color='blue')
    
    # Highlight the current position in red
    ax.scatter(drone_pos[0], drone_pos[1], drone_pos[2], color='red', s=5)

    # Plot the current payload position in blue
    ax.scatter(payload_pos[0], payload_pos[1], payload_pos[2], color='blue', s=2)

    # Draw a line representing the cable between the drone and the payload
    ax.plot(
        [drone_pos[0], payload_pos[0]], 
        [drone_pos[1], payload_pos[1]], 
        [drone_pos[2], payload_pos[2]], 
        color='black', linewidth=0.5, label='Cable'
    )
    
    # Calculate dynamic limits based on the current positions with padding
    x_min = drone_pos[0] - arm_length - zoom_padding
    x_max = drone_pos[0] + arm_length + zoom_padding
    y_min = min(drone_pos[1], payload_pos[1]) - arm_length - zoom_padding
    y_max = max(drone_pos[1], payload_pos[1]) + arm_length + zoom_padding
    z_min = min(drone_pos[2], payload_pos[2]) - arm_length - zoom_padding
    z_max = max(drone_pos[2], payload_pos[2]) + arm_length + zoom_padding

    # Apply the dynamic limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    # Define rotor positions (assuming arms are along the y-axis before rotation)
    rotor1_rel_pos = np.array([0, arm_length, 0])
    rotor2_rel_pos = np.array([0, -arm_length, 0])

    # Rotate the arm positions according to the drone's rotation angle around the x-axis
    rotor1_pos = drone_pos + np.dot(rotation_matrix_around_x(angle),rotor1_rel_pos)
    rotor2_pos = drone_pos + np.dot(rotation_matrix_around_x(angle),rotor2_rel_pos)

    # Draw the arms
    ax.plot(
        [drone_pos[0], rotor1_pos[0]], 
        [drone_pos[1], rotor1_pos[1]], 
        [drone_pos[2], rotor1_pos[2]], 
        color='gray', linewidth=1, label='Arm'
    )
    ax.plot(
        [drone_pos[0], rotor2_pos[0]], 
        [drone_pos[1], rotor2_pos[1]], 
        [drone_pos[2], rotor2_pos[2]], 
        color='gray', linewidth=1
    )
    
    # Draw the propellers
    ax.scatter(rotor1_pos[0], rotor1_pos[1], rotor1_pos[2], color='green', s=1, marker='o', label='Propeller 1')
    ax.scatter(rotor2_pos[0], rotor2_pos[1], rotor2_pos[2], color='green', s=1, marker='o', label='Propeller 2')

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Save the current frame
    plt.savefig(f'{output_folder}frame_{i:03d}.png')
    plt.close(fig)

# Convert the saved frames to a video
frames = []
for i in range(iterations+1):
    frames.append(imageio.imread(f'{output_folder}frame_{i:03d}.png'))

imageio.mimsave(output_video, frames, fps=50)  # Adjust fps as needed

print(f"Video saved as {output_video}")