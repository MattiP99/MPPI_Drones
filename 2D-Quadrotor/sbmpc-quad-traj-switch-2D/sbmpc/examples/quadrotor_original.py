import time, os

import jax
import jax.numpy as jnp
from scipy import signal
os.environ['XLA_FLAGS'] = (
        '--xla_gpu_triton_gemm_any=True '
    )

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import sys
sys.path.append('/home/mpiras/MPPI/sbmpc-quad-traj-switch-2D-TEST3/sbmpc')
from sbmpc.model import Model, ModelMjx
from sbmpc.solvers import SbMPC, BaseObjective
from sbmpc.utils.settings import ConfigMPC, ConfigGeneral
from sbmpc.utils.geometry import skew, quat_product, quat2rotm, quat_inverse, rotation_matrix_around_x
import sbmpc.utils.simulation as simulation
import sbmpc.utils.trapezoidal_traj as trapezoidal_traj




MODEL = "classic"

input_max = jnp.array([100,100])
input_min = jnp.array([0,0])

mass = 2.7
mass_payload = 0.25
cable_length = 0.5
arm_length = 0.4
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
global tension
global is_slack_dynamic
tension = []
nq = 7

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

"""
def func_slack(state,inputs):        
        acc_L =  - jnp.array([0.,0.,gravity])
        
        acc =  (1/mass)*jnp.array([0.,0.,inputs]) - jnp.array([0.,0.,gravity])
        print("acc slack",acc)
        print("acc_L slack",acc_L)
        print("inputs",inputs)

        ######## state_dot ########
        state_dot =  jnp.concatenate([state[6:9],
                            #0.5 * quat_product(quat, ang_vel_quat),
                            state[9:12],
                            #orientation_mat @ acc[:3],
                            acc,
                            acc_L])

        
        return state_dot
"""


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
    #print("csi SLACK",csi)
    #print("state SLACK",state[0:6])
    #print("vel SLACK",state[7:13])   
   
    acc_L =  - jnp.array([0.,0.,gravity])

    F = inputs[0]+inputs[1]
    # Obtain Quadrotor Force Vector

    # ALREADY IN RADIANTS??
    
    
    quad_force_vector = F * rotation_matrix_around_x(state[6]) @ e3  
    #quad_force_vector = (F * e3.reshape(1,3) @ rotation_matrix_around_x(state[6])).reshape(3,)
    #print("quad_force_vector slack",quad_force_vector)

    # Solving for Quadrotor Acceleration
    acc = quad_force_vector/mass - jnp.array([0.,0.,gravity])
    acc_rot = (arm_length * (inputs[0] - inputs[1])) / inertia_slack 

    global tension
    tension.append(jnp.array([0.0,0.0,0.0]))
   #print("TENSION SLACK",tension)
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


def func_acc_taut(quad_force_vector, tension_vector):
     
     acc_L =  -jnp.transpose(tension_vector).reshape(3,) / mass_payload - jnp.array([0.,0.,gravity]) 
     acc = (quad_force_vector + jnp.transpose(tension_vector).reshape(3,)) / mass  - jnp.array([0.,0.,gravity])
     return acc,acc_L

def baumgarte_stabilization(state):
    
    alpha = 5
    beta = alpha*jnp.sqrt(2)

    
    # Calculate the current constraint (distance error)
    diff_pos = state[0:3] - state[3:6]
    current_distance = jnp.linalg.norm(diff_pos)
    C = current_distance - cable_length  # Position constraint violation

    # Velocity constraint violation (time derivative of position constraint)
    relative_vel = state[7:10] - state[10:13]
    C_dot = jnp.dot(diff_pos / current_distance, relative_vel)
    
    # Desired corrective acceleration using Baumgarte stabilization
    correction_term = -2 * alpha * C_dot - beta**2 * C

    # Directional vector from payload to drone
    direction = diff_pos / current_distance if current_distance != 0 else np.array([0.0, 0.0, 0.0])
    return correction_term * direction

def func_taut(state,inputs,csi,csi_dot):

    ########## SET ORIGINAL EQUATIONS ###############
    print("INSIDE TAUT")
    print("DRONE VELOCITY CSI",jnp.dot(state[7:10],csi)*csi)
    print("PAYLOAD VELOCITY CSI",jnp.dot(state[10:13],csi)*csi)

    #print("state taut",state[0:6])
    #print("vel taut",state[7:13])
    
    #csi = (state[3:6] - state[0:3])/cable_length

    #csi_dot = (state[10:13] - state[7:10])/cable_length
    
    #print("csi taut",csi)
    #print("csi_dot taut",csi_dot)
    #print("csi NORM taut",jnp.linalg.norm(csi))
    #print("csi_dot NORM taut",jnp.linalg.norm(csi_dot))
    norm_csi_dot = jnp.linalg.norm(csi_dot)

    csi_omega = jnp.cross(csi, csi_dot)
    #csi_omega_norm = jnp.linalg.norm(csi_omega)
    #csi_omega = csi_omega / csi_omega_norm

    F =  inputs[0]+inputs[1]

    #print("inputs[0] taut",inputs[0])
    #print("inputs[1] taut",inputs[1])
    #print("F taut",F)
    #print("THETA taut",state[6])
    # Obtain Quadrotor Force Vector
    # ALREADY IN RADIANTS??
    #quad_force_vector = F * rotation_matrix_around_x(normalize_angle(state[6])) @ e3  
    #quad_force_vector = F * rotation_matrix_around_x((state[6] * jnp.pi)/180) @ e3  

    quad_force_vector = F * rotation_matrix_around_x(state[6]) @ e3  
    #quad_force_vector = F * (e3.reshape(1,3) @ rotation_matrix_around_x(state[6])).reshape(3,)  
    print("quad_force_vector taut",quad_force_vector)
   
    #quad_centrifugal_f = mass * cable_length* (jnp.dot(csi_omega, csi_omega))
    #quad_centrifugal_f = mass * (1/cable_length)* jnp.dot(jnp.array(state[7:10] - state[10:13]),jnp.array(state[7:10] - state[10:13]))
    quad_centrifugal_f = (mass/jnp.linalg.norm(state[0:3] - state[3:6] )) * jnp.linalg.norm(jnp.array(state[7:10] - state[10:13]))**2


    print("quad_centrifugal_f taut",quad_centrifugal_f)
    #tension_vector = mass_payload * ((-csi.reshape(1,3) @ quad_force_vector) + quad_centrifugal_f) * csi.reshape(3,1) / (mass+mass_payload)
    
    tension_vector =  (mass_payload/ (mass+mass_payload)) * (( jnp.dot(-csi,quad_force_vector)) - quad_centrifugal_f) * csi 

    print("tension_vector taut",tension_vector)
    # Solving for Load Acceleration
    global tension
    tension.append(tension_vector.reshape(3,))


    #acc,acc_L = func_acc_taut(quad_force_vector,tension_vector)
    acc_L =  -jnp.transpose(tension_vector).reshape(3,) / mass_payload - jnp.array([0.,0.,gravity]) 
    acc = (quad_force_vector + jnp.transpose(tension_vector).reshape(3,)) / mass  - jnp.array([0.,0.,gravity])
    acc_L_Control = jnp.dot(acc_L,csi) * csi
    vel_L_Control = jnp.dot(state[10:13],csi) * csi
    #vel_L_Control = (csi.reshape(3,1)@ csi.reshape(1,3)) @ state[10:13].reshape(3,1)
    # acc_L_Control = (-jnp.dot(tension_vector.reshape(3,),csi)/ mass_payload - jnp.dot(jnp.array([0.,0.,gravity]),csi))*csi

    print("acc_L taut",acc_L)
    print("acc_L_CONTROL taut",acc_L_Control)
    print("vel_L_CONTROL taut",vel_L_Control.reshape(3,))
    # Solving for Quadrotor Acceleration
    
    acc_Control = jnp.dot(acc,csi) * csi

    
    
    # Acceleration constraint with Baumgarte correction
    acc_L = acc_L - baumgarte_stabilization(state)
    

    acc_Control = jnp.dot(acc,csi) * csi
    vel_Control = jnp.dot(state[7:10],csi) * csi
    print("acc taut",acc)
    print("acc_CONTROL taut",acc_Control)
    print("vel_CONTROL taut",vel_L_Control.reshape(3,))
    # Solving for Quadrotor Acceleration
    #acc = (quad_force_vector + jnp.transpose(tension_vector).reshape(3,)) / mass  - jnp.array([0.,0.,gravity])
    
    if jnp.dot((acc - acc_L ),csi) == -jnp.linalg.norm(state[7:10] - state[10:13] )**2 / jnp.linalg.norm(state[0:3] - state[3:6] ):
        print("CONSTRAINT VALID")
    else:
        print("CONSTRAINT NOT VALID")
        print("ERROR CONSTRAINT", jnp.dot((acc - acc_L),csi) + jnp.linalg.norm(state[7:10] - state[10:13] )**2 / jnp.linalg.norm(state[0:3] - state[3:6] ))
    vel_Control = (csi.reshape(3,1)@ csi.reshape(1,3)) @ state[7:10].reshape(3,1)
    #vel_diff = jnp.dot(state[7:10],csi) - jnp.dot(state[10:13],csi)
    #acc_Control = ((jnp.dot(quad_force_vector,csi) + jnp.dot(tension_vector.reshape(3,),csi)) / mass - jnp.dot(jnp.array([0.,0.,gravity]),csi))*csi
    
    acc_L = acc_L - baumgarte_stabilization(state)
    #acc = acc_L - csi_ddot * cable_length 
    print("acc taut",acc)
    print("acc_CONTROL taut",acc_Control)
    print("vel_CONTROL taut",vel_Control.reshape(3,))
    #print("vel_DIFF taut",vel_diff)
    
    if  jax.numpy.array_equal(acc,acc_L):
        print("ACCELERATIONS EQUALS")
    else:
         print("ACCELERATIONS DIFFERENT")

    #acc_L = acc_L - baumgarte_stabilization(state)
      
    #acc_L  = acc_L.reshape(3,)
    #acc  = acc.reshape(3,)
    #acc_rot = (drone_length * (inputs[0] - inputs[1])) / (inertia_taut)
      
    acc_rot = (arm_length * (inputs[0] - inputs[1])) / (inertia_slack) 
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
    



def check_distance(state, csi,csi_dot,d,d_dot):
    global is_slack_dynamic
    is_slack = is_slack_dynamic
    print("INSIDE CHECK DISTANCE")

    uav_attach_vector =  state[0:3] - state[3:6]  
    # uav_attach_distance = jnp.linalg.norm(uav_attach_vector)
    uav_attach_distance = jnp.linalg.norm(uav_attach_vector)
    """
    if  (is_slack == False) & (jnp.linalg.norm(state[0:3] - state[3:6]) >= cable_length - 0.001) :
            
            
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
            
            print("CHECK 1")
            is_slack = False
    elif (is_slack == False) & (jnp.linalg.norm(state[0:3] - state[3:6]) < cable_length - 0.001):
            print("CHECK 2")
            is_slack = True
    """
    if (is_slack == True) & (jnp.linalg.norm(state[0:3] - state[3:6]) < cable_length - 0.001):# & (d_dot < 0.001):
            print("CHECK 3")
            is_slack = True
    elif (is_slack == True) & (jnp.linalg.norm(state[0:3] - state[3:6]) >= cable_length - 0.001):# & (d_dot >=  0.001):
            print("CHECK 4")
            #state = state.at[0:3].set(state[3:6] + (cable_length - 0.001) * csi)
            #state = state.at[3:6].set(state[0:3] + (cable_length - 0.001) * csi)
            
        

            ################ Inelastic Collision ###################
            cable_direction_projmat = csi.reshape((3,1)) @ csi.reshape((1,3))
            vDrone_proj = cable_direction_projmat @ state[7:10]
            vPayload_proj = cable_direction_projmat @ state[10:13]

            v_kp1_parallel_drone = (mass * vDrone_proj + mass_payload * vPayload_proj)/(mass_payload + mass)
            v1 = v_kp1_parallel_drone + state[7:10] - vDrone_proj
            v2 = v_kp1_parallel_drone + state[10:13] - vPayload_proj
            print("v_1 taut",v1)
            print("v_2 taut",v2)
            
            # Mi piacerebbe avere una cosa del genere avere una cosa del genere e se fossimo pií avere una cosa 
            
            state = state.at[7:10].set(v1)
            state = state.at[10:13].set(v2)
            
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
            

            is_slack = False
    return state, is_slack
    
#@jax.jit


def check_dynamics(state,inputs,csi,csi_dot):
    print("INSIDE CHECK DYNAMICS")
    print("e3 CHECK DYNAMICS",e3 )
    print("csi CHECK DYNAMICS",csi )
    global is_slack_dynamic
    print("is_slack_dynamic CHECK DYNAMICS",is_slack_dynamic )
    
    if  (is_slack_dynamic == False):# & (d_dot >= 0.001):
            result_taut = func_taut(state,inputs,csi,csi_dot)
            csi_omega = jnp.cross(csi, csi_dot)
        

            F =  inputs[0]+inputs[1]
            #quad_force_vector = (F * e3.reshape(1,3) @ rotation_matrix_around_x(state[6])).reshape(3,)
            quad_force_vector = F * rotation_matrix_around_x(state[6]) @ e3 
            #################### WHICH CONDITION OF THE TENSION?????????? ##################
            tension_vector = jnp.dot((-quad_force_vector/mass + ( result_taut[7:10] - (result_taut[10:13] + baumgarte_stabilization(state)))),csi)*csi/(1/mass + 1/mass_payload)
            #tension_vector = jnp.dot((-quad_force_vector/mass + ( result_taut[7:10] - result_taut[10:13])),csi)*(csi)/(1/mass + 1/mass_payload)
            print("tension_vector DYNAMIC", tension_vector)
            #if jnp.dot(tension_vector ,e3) <= 0:
            if jnp.dot(tension_vector,csi) >= 0.0:
                print("INSIDE TAUT TO SLACK CHECK DYNAMICS", tension_vector)
                is_slack_dynamic = True
    

    """
    else:
        if  (is_slack_dynamic == False):# & (d_dot >= 0.001):
            result_taut = func_taut(state,inputs,csi,csi_dot)
            csi_omega = jnp.cross(csi, csi_dot)
        

            F =  inputs[0]+inputs[1]
            quad_force_vector = F * rotation_matrix_around_x(state[6]) @ e3  
            quad_centrifugal_f = mass * cable_length * (csi_omega @ csi_omega)
            F_d = quad_force_vector + quad_centrifugal_f * csi - gravity * e3
            F_p =  quad_centrifugal_f * csi  - gravity * e3
            #tension_vector = ((F_d/ mass) * (-csi)  - (F_p/ mass_payload) * (-csi)  - (result_taut[7:10] - result_taut[10:13]) * (-csi)) / (1/mass + 1/mass_payload)
            #tension_vector = (jnp.dot((F_d/ mass) , (-csi))  - jnp.dot((F_p/ mass_payload) ,(-csi))  - jnp.dot((result_taut[7:10] - result_taut[10:13]) ,(-csi))) / (1/mass + 1/mass_payload) * (-csi)
            tension_vector = jnp.dot((-quad_force_vector/mass - (result_taut[10:13] - result_taut[7:10] )),csi)*csi/(1/mass + 1/mass_payload)

            print("tension_vector DYNAMIC 2", tension_vector)
            if jnp.dot(tension_vector ,e3)  >= 0:
            
                print("INSIDE TAUT TO SLACK CHECK DYNAMICS 2", tension_vector)
                #result_taut = func_slack(state,inputs,csi,csi_dot)
                #acc_vector.extend(result_taut[7:13])
                is_slack_dynamic = True
      """        

def quadrotor_dynamics(state: jnp.array, inputs: jnp.array , is_slack) -> jnp.array:

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
    #csi_dot = (((state[10:13] - state[7:10])*jnp.linalg.norm(state[3:6] - state[0:3]))-((state[3:6] - state[0:3]) * (state[3:6] - state[0:3]) * (1/jnp.linalg.norm(state[3:6] - state[0:3])) * (state[10:13] - state[7:10])))/(jnp.linalg.norm(state[3:6] - state[0:3]))**2

    #v_dot_dvdt = jnp.dot((state[0:3] - state[3:6]), (state[7:10] - state[10:13] ))
    #d_v_norm_dt = v_dot_dvdt / jnp.linalg.norm(state[0:3] - state[3:6])
    
    # Apply the formula: (dv/dt * v_norm - v * d(v_norm)/dt) / v_norm^2
    #csi_dot = ((state[7:10] - state[10:13] ) * jnp.linalg.norm(state[0:3] - state[3:6]) - (state[0:3] - state[3:6])  * d_v_norm_dt) / (jnp.linalg.norm(state[0:3] - state[3:6]) ** 2)
    #csi_dot = (state[7:10] - state[10:13] )/cable_length
    csi_dot = 1/jnp.linalg.norm(state[0:3] - state[3:6] ) * (jnp.diag(jnp.array([1,1,1])) - ((csi).reshape(3,1) @ (csi).reshape(1,3))) @ (state[7:10] - state[10:13]).reshape(3,1)
    csi_dot = csi_dot.reshape(3,)

    lilli = jnp.dot(csi,csi_dot)
    print("LILLLLLLIIIII",lilli)
    print("csi dynamics",csi)
    print("csi_dot dynamics",csi_dot)
    print("csi NORM dynamics",jnp.linalg.norm(csi))
    print("csi_dot NORM dynamics",jnp.linalg.norm(csi_dot))

    
    
    #print("INPUTS DYNAMICS",inputs)
    

    d = jnp.linalg.norm(state[3:6] - state[0:3])
    #d_dot = ((state[3:6]  - state[0:3]).transpose() @ (state[10:13] - state[7:10]))/((jnp.linalg.norm(state[3:6] - state[0:3])))
    # Compute the dot product of v and dv/dt
    v_dot_dvdt = jnp.dot(state[3:6]  - state[0:3], state[10:13] - state[7:10])
    
    # Compute the derivative of the norm
    d_dot = v_dot_dvdt / d
    

    #d = jnp.linalg.norm(state[0:3] - state[3:6] )
    #d_dot = ((state[0:3] - state[3:6]  ).transpose() @ (state[7:10] - state[10:13] ))/((jnp.linalg.norm(state[0:3] - state[3:6] )))
    # Inputs are said to be total force and total torque but I already have this computation in order 
    # to consider as inputs the f forces and moments in the body frame

    
    
    print("d_dot dynamic", d_dot)
    global is_slack_dynamic

    if  (is_slack_dynamic == False):# & (d_dot >= 0.001):
            result_taut = func_taut(state,inputs,csi,csi_dot)
            acc_vector.extend(result_taut[7:13])
            return result_taut
            
    else:
            result_taut = func_slack(state,inputs,csi,csi_dot)
            acc_vector.extend(result_taut[7:13])
            return result_taut
    
    

    

    


# NOTE: at the moment there is not reference? And there will be,will it be for the payload or for the drone? 
# Here I could generate the desired trajectory for the drone or the payload
class Objective(BaseObjective):
    """ Cost function for the Quadrotor regulation task"""
        
    def compute_state_error(self, state: jnp.array, state_ref : jnp.array) -> jnp.array:
        pos_err = state[0:2] - state_ref[0:2]
        #att_err = quat_product(quat_inverse(state[3:7]), state_ref[3:7])[1:4]
        vel_err = state[2:4] - state_ref[2:4]
        #ang_vel_err = state[16:] - state_ref[13:]

        return pos_err,  vel_err

    def running_cost(self, state: jnp.array, inputs: jnp.array, reference) -> jnp.float32:
        state_ref = reference[:14]
        input_ref = reference[14:]
        pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, state_ref)

        return (10 * pos_err.transpose() @ pos_err +
                0.01 * att_err.transpose() @ att_err +
                0.5 * vel_err.transpose() @ vel_err +
                0.1 * ang_vel_err.transpose() @ ang_vel_err +
                (inputs-input_ref).transpose() @ jnp.diag(jnp.array([0.1, 0.1, 0.1, 0.5])) @ (inputs-input_ref))

    def final_cost(self, state, reference):
        pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, reference[:13])
        return (100 * pos_err.transpose() @ pos_err +
                0.1 * att_err.transpose() @ att_err +
                5 * vel_err.transpose() @ vel_err +
                1 * ang_vel_err.transpose() @ ang_vel_err)


class Simulation(simulation.Simulator):
    #def __init__(self, initial_state, model, controller, reference, num_iterations):
    #    super().__init__(initial_state, model, controller,reference, num_iterations)
    def __init__(self, initial_state, model, is_slack, num_iterations):
        super().__init__(initial_state, model,is_slack, num_iterations)

        # DON´T NEED THE REFERENCE RIGHT NOW
        
        #self.reference = jnp.zeros((T, x_init.size + input_hover.size),dtype=jnp.float32)
        #calculator = trapezoidal_traj.Trapeizoidal_Trajectory(q_init[0:3], q_des[0:3], 0.2, self.num_iter + self.controller.horizon + 1)
        #self.reference = calculator.compute_trajectory2()
        
        
    
    def update(self):
        
        
        #print("reference:", reference)
        
        # Compute the optimal input sequence
        #time_start = time.time_ns()
        
        
        print("ITER UPDATE", self.iter)
        print("IS_SLALCK UDPATE BEFORE",self.is_slack)
        x = jnp.arange(0, 1000, 0.1) 
        x1 = jnp.arange(0, 100, 0.1) 
        global is_slack_dynamic
        if self.iter == 0:
            
            self.is_slack =  is_slack_dynamic
        else:
            is_slack_dynamic = self.is_slack

        if self.is_slack == True:
            is_slack_number = 1
            
        else:
            is_slack_number = 0
        self.vector_isslack[self.iter] = is_slack_number
        
        # INPUT HOVERING
        #input_sequence =  jnp.array([(mass+mass_payload)*gravity/2 , (mass + mass_payload)*gravity/2])

        # INPUT SINUSOIDAL
        #input_sequence =  jnp.array([2*(mass+mass_payload)*gravity + 0.5 * jnp.cos(0.5*x) , 2*(mass + mass_payload)*gravity - 0.5 * jnp.cos(0.5*x)])
        
        # INPUT ASCENDING WHILE TURNING IN ONE DIRECTION
        #input_sequence =  jnp.array([(mass+mass_payload)*gravity , 0.9* (mass + mass_payload)*gravity])
        
        # INPUT ASCENDING WHILE OSCILLATING VERTICALLY
        #input_sequence =  jnp.array([((mass+mass_payload)*gravity + 1)/2 + (20 * jnp.sin(0.1*x))/2 ,
        #                              ((mass+mass_payload)*gravity + 1)/2 + (20 * jnp.sin(0.1*x))/2])
        
        # INPUT ASCENDING WHILE OSCILLATING HORIZONTALLY 1
        #input_sequence =  jnp.array([((mass+mass_payload)*gravity + 6)/2 + (0.5 * jnp.sin(0.5*x))/2 ,
        #                             ((mass+mass_payload)*gravity + 6)/2 - (0.5 * jnp.sin(0.5*x))/2])

        
        
        # INPUT ASCENDING WHILE OSCILLATING HORIZONTALLY 2
        #input_sequence_1 =  jnp.array([(mass+mass_payload)*gravity/2 + 8 , (mass + mass_payload)*gravity/2 + 8 ])
        #input_sequence_1 = jnp.tile(input_sequence_1, (1000, 1)) 
        
        #input_sequence_1 =  jnp.array([((mass+mass_payload)*gravity + 1)/2 + (20 * jnp.sin(0.1*x1))/2 ,
        #                              ((mass+mass_payload)*gravity + 1)/2 - (20 * jnp.sin(0.1*x1))/2])
        #input_sequence_1 = input_sequence_1.reshape(1000,2)
        #input_sequence_2 =  jnp.array([((mass+mass_payload)*gravity + 1)/2 + (20 * jnp.sin(0.05*x))/2 ,
        #                              ((mass+mass_payload)*gravity + 1)/2 - (20 * jnp.sin(0.05*x))/2])
        #input_sequence_2 = input_sequence_2.reshape(2500,2)
        #input_sequence = jnp.concatenate([input_sequence_1,input_sequence_2],axis=0)
        
        # INPUT CASE SWITCHING
        
        input_sequence_1 =  jnp.array([(mass+mass_payload)*gravity/2 + 0.0 , (mass + mass_payload)*gravity/2 + 0.0])
        input_sequence_1 = jnp.tile(input_sequence_1, (350, 1)) 
        input_sequence_2 =  jnp.array([(mass+mass_payload)*gravity/2 + 0.3  , (mass + mass_payload)*gravity/2 + 0.1 ])
        input_sequence_2 = jnp.tile(input_sequence_2, (50, 1)) 
        input_sequence = jnp.concatenate([input_sequence_1,input_sequence_2], axis=0)
        input_sequence_3 =  jnp.array([(mass + mass_payload)*gravity/2 + 0.1 , (mass + mass_payload)*gravity/2 + 0.3 ])
        input_sequence_3 = jnp.tile(input_sequence_3, (100, 1)) 
        input_sequence = jnp.concatenate([input_sequence,input_sequence_3], axis=0)
        input_sequence_4 =  jnp.array([(mass + mass_payload)*gravity/2 + 0.3 , (mass + mass_payload)*gravity/2 + 0.1  ])
        input_sequence_4 = jnp.tile(input_sequence_4, (50, 1)) 
        input_sequence = jnp.concatenate([input_sequence,input_sequence_4], axis=0)
        input_sequence_5 =  jnp.array([(mass + mass_payload)*gravity/2 + 0.0 , (mass + mass_payload)*gravity/2 + 0.0  ])
        input_sequence_5 = jnp.tile(input_sequence_5, (450, 1)) 
        input_sequence = jnp.concatenate([input_sequence,input_sequence_5], axis=0)
        
        """
        input_sequence_1 =  jnp.array([(mass+mass_payload)*gravity/2 + 1 , (mass + mass_payload)*gravity/2 + 1])
        input_sequence_1 = jnp.tile(input_sequence_1, (600, 1)) 
        input_sequence_2 =  jnp.array([-(mass+mass_payload)*gravity/2 - 0.5 , -(mass + mass_payload)*gravity/2 - 0.5  ])
        input_sequence_2 = jnp.tile(input_sequence_2, (400, 1)) 
        input_sequence = jnp.concatenate([input_sequence_1,input_sequence_2], axis=0)
        """        
        
        # INPUT CASE SWITCHING 2
        #input_sequence =  jnp.array([-(mass+mass_payload)*gravity/2 - 8  , -(mass + mass_payload)*gravity/2 - 8  ])
        #input_sequence = jnp.tile(input_sequence, (1000, 1)) 


        #input_sequence = (mass+mass_payload)*gravity + signal.chirp(x, f0=(mass+mass_payload)*gravity + 100, f1=650, t1=1000, method='hyperbolic')

        #input_sequence = (mass+mass_payload)*gravity 
        #print("computation time: {:.3f} [ms]".format(1e-6 * (time.time_ns() - time_start)))

        
        #ctrl = input_sequence[:, self.iter]
        #ctrl = input_sequence
        
        
        # CONTROL FOR 2
        ctrl = input_sequence[self.iter, :]
        self.input_traj[self.iter, :] = ctrl
        #d = jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6])
        #d_dot = jnp.linalg.norm((self.current_state[0:3] - self.current_state[3:6])/((jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6]))))
        print("CURRENT_STATE_DRONE", self.current_state[0:3])
        print("CURRENT_STATE_PAYLOAD", self.current_state[3:6])
        print("CURRENT_VELOCITY_DRONE", self.current_state[7:10])
        print("CURRENT_VELOCITY_PAYLOAD", self.current_state[10:13])
        
        csi = (self.current_state[0:3] - self.current_state[3:6] )/jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6] )
        #v_dot_dvdt = jnp.dot((self.current_state[0:3] - self.current_state[3:6]), (self.current_state[7:10]  - self.current_state[10:13] ))
        #d_v_norm_dt = v_dot_dvdt / jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6])
        
        # Apply the formula: (dv/dt * v_norm - v * d(v_norm)/dt) / v_norm^2
        #csi_dot = ((self.current_state[7:10]  - self.current_state[10:13] ) * jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6]) - (self.current_state[0:3] - self.current_state[3:6])  * d_v_norm_dt) / (jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6]) ** 2)
        #csi_dot = (self.current_state[7:10]  - self.current_state[10:13])/ cable_length
        csi_dot = 1/jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6] ) * (jnp.diag(jnp.array([1,1,1])) - ((csi).reshape(3,1) @ (csi).reshape(1,3))) @ (self.current_state[7:10] - self.current_state[10:13]).reshape(3,1)
        csi_dot = csi_dot.reshape(3,)
        
        d = jnp.linalg.norm(self.current_state[3:6] - self.current_state[0:3])
       
        # Compute the dot product of v and dv/dt
        v_dot_dvdt = jnp.dot(self.current_state[3:6]  - self.current_state[0:3], self.current_state[10:13] - self.current_state[7:10])
        
        # Compute the derivative of the norm
        d_dot = v_dot_dvdt / d
        
        
        #print("csi update",csi)
        #print("csi_dot update",csi_dot)
        #print("csi NORM update",jnp.linalg.norm(csi))
        #print("csi_dot NORM update",jnp.linalg.norm(csi_dot))

        #primo = (self.current_state[7:10] - self.current_state[10:13] )*jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6] )
        #print("PRIMO update",primo)
        #secondo = ((self.current_state[0:3] - self.current_state[3:6] ) * (self.current_state[0:3] - self.current_state[3:6] ) * (1/jnp.linalg.norm(self.current_state[0:3]- self.current_state[3:6] )) * (self.current_state[7:10] - self.current_state[10:13] ))
        #print("SECONDO update",secondo)
        #terzo = (jnp.linalg.norm(self.current_state[3:6] - self.current_state[0:3]))**2
        #print("TERZO update",terzo)
        


        #v_kp1  = handle_collision(self.current_state, d,d_dot, csi,csi_dot,self.is_slack)
        #self.current_state = self.current_state.at[nq:nq+7].set(v_kp1)
        
        ######################### SWITCHING CONTROL ########################
        

        #check_dynamics(self.current_state, csi,csi_dot,d, d_dot)
        check_dynamics(self.current_state, ctrl,csi,csi_dot)
        
        print("GLOBAL",is_slack_dynamic)
        
        self.current_state  = self.model.integrate(self.current_state, ctrl, self.is_slack, dt)
        
        csi = (self.current_state[0:3] - self.current_state[3:6] )/jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6] )

        #v_dot_dvdt = jnp.dot((self.current_state[0:3] - self.current_state[3:6]), (self.current_state[7:10]  - self.current_state[10:13] ))
        #d_v_norm_dt = v_dot_dvdt / jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6])
        
        # Apply the formula: (dv/dt * v_norm - v * d(v_norm)/dt) / v_norm^2
        #csi_dot = ((self.current_state[7:10]  - self.current_state[10:13] ) * jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6]) - (self.current_state[0:3] - self.current_state[3:6])  * d_v_norm_dt) / (jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6]) ** 2)
        #csi_dot = (self.current_state[7:10]  - self.current_state[10:13])/cable_length
        
        csi_dot = 1/jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6] ) * (jnp.diag(jnp.array([1,1,1])) - ((self.current_state[0:3] - self.current_state[3:6]).reshape(3,1) @ (self.current_state[0:3] - self.current_state[3:6]).reshape(1,3))) @ (self.current_state[7:10] - self.current_state[10:13]).reshape(3,1)
        csi_dot = csi_dot.reshape(3,)
        
        d = jnp.linalg.norm(self.current_state[3:6] - self.current_state[0:3])
       
        # Compute the dot product of v and dv/dt
        v_dot_dvdt = jnp.dot(self.current_state[3:6]  - self.current_state[0:3], self.current_state[10:13] - self.current_state[7:10])
        
        # Compute the derivative of the norm
        d_dot = v_dot_dvdt / d
        print("is_slack_dynamic AFTER CHECK DYNAMICS",is_slack_dynamic)
        print("is_slack AFTER CHECK DYNAMICS",self.is_slack)
        
        self.current_state, self.is_slack = check_distance(self.current_state, csi,csi_dot,d, d_dot)
        is_slack_dynamic = self.is_slack
        print("is_slack_dynamic AFTER CHECK DISTANCE",is_slack_dynamic)
        print("is_slack AFTER CHECK DISTANCE",self.is_slack)
        #csi = (self.current_state[0:3] - self.current_state[3:6] )/jnp.linalg.norm(self.current_state[0:3]- self.current_state[3:6] )
        #csi_dot = ((self.current_state[10:13] - self.current_state[7:10])*jnp.linalg.norm(self.current_state[3:6] - self.current_state[0:3])-((self.current_state[3:6] - self.current_state[0:3]) * (self.current_state[3:6] - self.current_state[0:3]) * (1/jnp.linalg.norm(self.current_state[3:6] - self.current_state[0:3])) * (self.current_state[10:13] - self.current_state[7:10])))/(jnp.linalg.norm(self.current_state[3:6] - self.current_state[0:3]))**2
        
        # self.is_slack = check_distance(self.current_state, csi,csi_dot,self.is_slack)
        
        
        #print("cureent state1",self.current_state)
        # Check for collision and handle it
        
        print("current_state UDPATE AFTER",self.current_state)
        #print("IS_SLALCK UDPATE AFTER",self.is_slack)


        # After integration bt = bt + dt or bt = 0 .
        #self.bt = bt
        #self.state_traj[self.iter + 1, :] = self.current_state_vec()
        self.state_traj[self.iter + 1, :] = self.current_state_vec()



if __name__ == "__main__":

    mpc_config = ConfigMPC(0.001,
                           25,
                           #jnp.array([0.2, 0.3, 0.3, 0.15]),
                           jnp.array([0.2]),
                           num_parallel_computations=10000,
                           initial_guess=input_hover)
    gen_config = ConfigGeneral("float32", jax.devices("cpu")[0])

    if MODEL == "classic":
        # HERE WHAT VALUE OF nq,nv,nu SHOULD I USE?
        system = Model(quadrotor_dynamics, 7, 7, 2, [input_min, input_max])
        q_init = jnp.array([0.0, 0.0, 5.0, 0.0, 0.0, 4.6, 0], dtype=jnp.float32)  # hovering position
        x_init = jnp.concatenate([q_init, jnp.array([0,0,0,0,0,0,0])])#(system.nv, dtype=jnp.float32)], axis=0)
        state_init = x_init
    elif MODEL == "mjx":
        system = ModelMjx("bitcraze_crazyflie_2/cf2.xml")
        q_init = system.data.qpos
        x_init = jnp.concatenate([q_init, jnp.zeros(system.nv, dtype=jnp.float32)], axis=0)
        state_init = system.data
    else:
        raise ValueError("Model must be either 'classic' or 'mjx'")
    
    solver = SbMPC(system, Objective(), mpc_config, gen_config)

    T = 1000+25+1
   
    

    # NO REFERENCE RIGHT NOW

    #reference = jnp.zeros((T, x_init.size + input_hover.size),dtype=jnp.float32)
    #q_des = jnp.array([0.5, 0.5, 0.5, 1., 0., 0., 0.], dtype=jnp.float32)  # hovering position    t = jnp.arange(1,500+25+1)
    #q_des = jnp.array([0.0, 3.0], dtype=jnp.float32)  # hovering position    t = jnp.arange(1,500+25+1)

    #calculator = trapezoidal_traj.Trapeizoidal_Trajectory(q_init[0:2], q_des[0:2], 0.2, T)
    #reference = calculator.compute_trajectory2()
    

    # dummy for jitting
    #input_sequence = solver.compute_control_action(x_init, reference).block_until_ready()
    

    # Setup and run the simulation
    #sim = Simulation(state_init, system, solver, reference, 500)

    is_slack = False
    if jnp.linalg.norm(q_init[0:3] - q_init[3:6]) < cable_length:
        is_slack = True
    
    iterations = 1000
 
   
    is_slack_dynamic = is_slack

    sim = Simulation(state_init, system, is_slack, iterations)
    sim.simulate()

    print("switching trajecotry",sim.state_traj[600:640, 0:3])
    print("switching trajecotry payload",sim.state_traj[600:640, 3:6])
    print("switching velocity 600",sim.state_traj[600, 7:10])
    print("switching velocity payload 600",sim.state_traj[600, 10:13])

    print("switching velocity 601",sim.state_traj[601, 7:10])
    print("switching velocity payload 601",sim.state_traj[601, 10:13])

    print("switching velocity 614",sim.state_traj[614, 7:10])
    print("switching velocity payload 614",sim.state_traj[614, 10:13])

    print("switching velocity 615",sim.state_traj[615, 7:10])
    print("switching velocity payload 615",sim.state_traj[615, 10:13])

    print("DDOT 590", ((sim.state_traj[590,0:3] - sim.state_traj[590,3:6] ) @ (sim.state_traj[590,7:10] - sim.state_traj[590,10:13]))/((jnp.linalg.norm( sim.state_traj[590,0:3] - sim.state_traj[590,3:6]  ))))
    print("DDOT 600", ((sim.state_traj[600,0:3] - sim.state_traj[600,3:6] ) @ (sim.state_traj[600,7:10] - sim.state_traj[600,10:13]))/((jnp.linalg.norm( sim.state_traj[600,0:3] - sim.state_traj[600,3:6]  ))))
    print("DDOT 601", ((sim.state_traj[601,0:3] - sim.state_traj[601,3:6] ) @ (sim.state_traj[601,7:10] - sim.state_traj[601,10:13]))/((jnp.linalg.norm( sim.state_traj[601,0:3] - sim.state_traj[601,3:6]  ))))

    print("DDOT 602", ((sim.state_traj[602,0:3] - sim.state_traj[602,3:6] ) @ (sim.state_traj[602,7:10] - sim.state_traj[602,10:13]))/((jnp.linalg.norm( sim.state_traj[602,0:3] - sim.state_traj[602,3:6]  ))))
    print("DDOT 608", ((sim.state_traj[608,0:3] - sim.state_traj[608,3:6] ) @ (sim.state_traj[608,7:10] - sim.state_traj[608,10:13]))/((jnp.linalg.norm( sim.state_traj[608,0:3] - sim.state_traj[608,3:6]  ))))

    print("DDOT 610", ((sim.state_traj[610,0:3] - sim.state_traj[610,3:6] ) @ (sim.state_traj[610,7:10] - sim.state_traj[610,10:13]))/((jnp.linalg.norm( sim.state_traj[610,0:3] - sim.state_traj[610,3:6]  ))))
    print("DDOT 612", ((sim.state_traj[612,0:3] - sim.state_traj[612,3:6] ) @ (sim.state_traj[612,7:10] - sim.state_traj[612,10:13]))/((jnp.linalg.norm( sim.state_traj[612,0:3] - sim.state_traj[612,3:6]  ))))
    print("DDOT 613", ((sim.state_traj[613,0:3] - sim.state_traj[613,3:6] ) @ (sim.state_traj[613,7:10] - sim.state_traj[613,10:13]))/((jnp.linalg.norm( sim.state_traj[613,0:3] - sim.state_traj[613,3:6]  ))))
    print("DDOT 614", ((sim.state_traj[614,0:3] - sim.state_traj[614,3:6] ) @ (sim.state_traj[614,7:10] - sim.state_traj[614,10:13]))/((jnp.linalg.norm( sim.state_traj[614,0:3] - sim.state_traj[614,3:6]  ))))
    print("DDOT 615", ((sim.state_traj[615,0:3] - sim.state_traj[615,3:6] ) @ (sim.state_traj[615,7:10] - sim.state_traj[615,10:13]))/((jnp.linalg.norm( sim.state_traj[615,0:3] - sim.state_traj[615,3:6]  ))))

    print("DDOT 618", ((sim.state_traj[618,0:3] - sim.state_traj[618,3:6] ) @ (sim.state_traj[618,7:10] - sim.state_traj[618,10:13]))/((jnp.linalg.norm( sim.state_traj[618,0:3] - sim.state_traj[618,3:6]  ))))

    print("DDOT 620", ((sim.state_traj[620,0:3] - sim.state_traj[620,3:6] ) @ (sim.state_traj[620,7:10] - sim.state_traj[620,10:13]))/((jnp.linalg.norm( sim.state_traj[620,0:3] - sim.state_traj[620,3:6]  ))))
    print("DDOT 630", ((sim.state_traj[630,0:3] - sim.state_traj[630,3:6] ) @ (sim.state_traj[630,7:10] - sim.state_traj[630,10:13]))/((jnp.linalg.norm( sim.state_traj[630,0:3] - sim.state_traj[630,3:6]  ))))

    print("DDOT 640", ((sim.state_traj[640,0:3] - sim.state_traj[640,3:6] ) @ (sim.state_traj[640,7:10] - sim.state_traj[640,10:13]))/((jnp.linalg.norm( sim.state_traj[640,0:3] - sim.state_traj[640,3:6]  ))))
    print("DDOT 650", ((sim.state_traj[650,0:3] - sim.state_traj[650,3:6] ) @ (sim.state_traj[650,7:10] - sim.state_traj[650,10:13]))/((jnp.linalg.norm( sim.state_traj[650,0:3] - sim.state_traj[650,3:6]  ))))
    print("DDOT 700", ((sim.state_traj[700,0:3] - sim.state_traj[700,3:6] ) @ (sim.state_traj[700,7:10] - sim.state_traj[700,10:13]))/((jnp.linalg.norm( sim.state_traj[700,0:3] - sim.state_traj[700,3:6]  ))))
    print("DDOT 900", ((sim.state_traj[900,0:3] - sim.state_traj[900,3:6] ) @ (sim.state_traj[900,7:10] - sim.state_traj[900,10:13]))/((jnp.linalg.norm( sim.state_traj[900,0:3] - sim.state_traj[900,3:6]  ))))


    ############################################### PLOTTING ###########################################################
    import csv
    import seaborn as sns
    import pandas as pd
    import matplotlib.gridspec as gridspec


    title = ['x', 'y', 'z', 'x_L', 'y_L', 'z_L', 'theta', 'x_dot', 'y_dot', 'z_dot', 'x_Ldot', 'y_Ldot', 'z_Ldot', 'theta_dot', 'Cable_Length', 'vector_isslack', 'u1','u2']
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

    vector_isslack = sim.vector_isslack[:]
    
    elements = jnp.concatenate([x.reshape(iterations+1,1),y.reshape(iterations+1,1),z.reshape(iterations+1,1),
                                x_L.reshape(iterations+1,1), y_L.reshape(iterations+1,1), z_L.reshape(iterations+1,1),
                                theta.reshape(iterations+1,1),
                                x_dot.reshape(iterations+1,1), y_dot.reshape(iterations+1,1), z_dot.reshape(iterations+1,1),
                                x_Ldot.reshape(iterations+1,1), y_Ldot.reshape(iterations+1,1), z_Ldot.reshape(iterations+1,1),
                                theta_dot.reshape(iterations+1,1),
                                Cable_Length.reshape(iterations+1,1),
                                vector_isslack.reshape(iterations + 1,1),
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
    vector_isslack = df['vector_isslack'].dropna()
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

    xtr_subsplot= fig.add_subplot(gs[0:4,0:3])

    plt.plot( iterations_array,x, linestyle='-', label='drone_x_position', color=colors[0], mfc='w', markersize=4) # plot data
    plt.plot(iterations_array,x_L, linestyle='-', label='payload_x_position', color=colors2[0], mfc='w', markersize=4) # plot data
    plt.plot(iterations_array,y, linestyle='-', label='drone_y_position', color=colors[16], mfc='w', markersize=4) # plot data
    plt.plot( iterations_array,y_L,linestyle='-',  label='payload_y_position', color=colors2[8], mfc='w', markersize=4) # plot data
    plt.plot( iterations_array,z,linestyle='-',  label='drone_z_position', color=colors[8], mfc='w', markersize=4) # plot data
    plt.plot( iterations_array,z_L, linestyle='-', label='payload_z_position', color=colors2[16], mfc='w', markersize=4) # plot data
   
    #plot params
    plt.xlim([0,iterations + 50])
    plt.ylim([-130,20])
    plt.minorticks_on()
    plt.tick_params(direction='in',right=True, top=True)
    plt.tick_params(labelsize=10)
    plt.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)
    #xticks = np.arange(0, 1e4,10)
    yticks = np.arange(-130,20.1,10)

    plt.tick_params(direction='in',which='minor', length=5, bottom=True, top=True, left=True, right=True)
    plt.tick_params(direction='in',which='major', length=10, bottom=True, top=True, left=True, right=True)
    #plt.xticks(xticks)
    plt.yticks(yticks)

    plt.xlabel('Iteration', fontsize=14) 
    plt.ylabel('Position [m]',fontsize=14)  # label the y axis


    plt.legend(fontsize=14)  # add the legend (will default to 'best' location)
    #plot text and line on top of figure
    plt.axvline(x=10, linestyle='dotted', color='black')
    plt.text(20.5, 3, 'First Zoom', rotation=90)

    plt.axvline(x=600, linestyle='dotted', color='purple')
    plt.text(610.5, 3, 'Second Zoom', rotation=90)

    #generate second panel
    xtr_subsplot = fig.add_subplot(gs[0:2,3:4])
    #plt.plot( iterations_array[0:50],x[0:50], linestyle='-', label='drone_x_position', color=colors[0], mfc='w', markersize=4) # plot data
    #plt.plot(iterations_array[0:50],x_L[0:50], linestyle='-', label='payload_x_position', color=colors2[0], mfc='w', markersize=4) # plot data
    #plt.plot(iterations_array[0:50],y[0:50], linestyle='-', label='drone_y_position', color=colors[8], mfc='w', markersize=4) # plot data
    #plt.plot( iterations_array[0:50],y_L[0:50],linestyle='-',  label='payload_y_position', color=colors2[8], mfc='w', markersize=4) # plot data
    plt.plot( iterations_array[0:100],z[0:100],linestyle='-',  label='drone_z_position', color=colors[8], mfc='w', markersize=4) # plot data
    plt.plot( iterations_array[0:100],z_L[0:100], linestyle='-', label='payload_z_position', color=colors2[16], mfc='w', markersize=4) # plot data

    #Define tick parameters
    xticks2 = np.arange(0,100,10)
    yticks2 = np.arange(3,7,0.25)
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
    #plt.plot( iterations_array[590:650],x[590:650], linestyle='-', label='drone_x_position', color=colors[0], mfc='w', markersize=4) # plot data
    #plt.plot(iterations_array[590:650],x_L[590:650], linestyle='-', label='payload_x_position', color=colors2[0], mfc='w', markersize=4) # plot data
    #plt.plot(iterations_array[590:650],y[590:650], linestyle='-', label='drone_y_position', color=colors[8], mfc='w', markersize=4) # plot data
    #plt.plot( iterations_array[590:650],y_L[590:650],linestyle='-',  label='payload_y_position', color=colors2[8], mfc='w', markersize=4) # plot data
    plt.plot( iterations_array[590:650],z[590:650],linestyle='-',  label='drone_z_position', color=colors[8], mfc='w', markersize=4) # plot data
    plt.plot( iterations_array[590:650],z_L[590:650], linestyle='-', label='payload_z_position', color=colors2[16], mfc='w', markersize=4) # plot data

    #Define tick parameters
    xticks3 = np.arange(590,650,5)
    yticks3 = np.arange(14,19.1,0.25)
   
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
    yticks = np.arange(-80,10.1,5)
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
    yticks2 = np.arange(-80,10.1,5)
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
    yticks = np.arange(-3.14,3.14,0.5)
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
    yticks2 = np.arange(-3.14,3.14,0.5)
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
    fig = plt.figure(4, figsize=(5, 5))
    plt.plot( iterations_array,Cable_Length, linestyle='-', label='Cable Length', color=colors2[16], mfc='w', markersize=4) # plot data

    #plt.plot(x_fit, ffit(x_fit), linestyle='-', marker='None', label='fit', color=colors[0], markerfacecolor='white', markersize=8) # plot data

    #plot params
    xticks = np.arange(0,1000.5,100)
    yticks = np.arange(0,0.55,0.1)
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
    fig = plt.figure(5, figsize=(5, 5))
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
    plt.ylabel('Inputs [N]',fontsize=14)  # label the y axis


    plt.legend(fontsize=14)  # add the legend (will default to 'best' location)

    ################################################ SLACK VECTOR PLOT ########################################################

    #plot
    fig = plt.figure(6, figsize=(5, 5))
    plt.plot( iterations_array,vector_isslack, linestyle='-', label='Slackness', color=colors2[16], mfc='w', markersize=4) # plot data

    #plt.plot(x_fit, ffit(x_fit), linestyle='-', marker='None', label='fit', color=colors[0], markerfacecolor='white', markersize=8) # plot data

    #plot params
    xticks = np.arange(0,1000.5,100)
    yticks = np.arange(-0.25,1.25,0.25)
    plt.minorticks_on()

    plt.tick_params(direction='in',which='minor', length=5, bottom=True, top=True, left=True, right=True)
    plt.tick_params(direction='in',which='major', length=10, bottom=True, top=True, left=True, right=True)
    plt.xticks(xticks)
    plt.yticks(yticks)


    #plt.text(1,325, f'y={Decimal(coefs[3]):.4f}x$^3$+{Decimal(coefs[2]):.2f}x$^2$+{Decimal(coefs[1]):.2f}x+{Decimal(coefs[0]):.1f}',fontsize =13)


    plt.xlabel('Iterations', fontsize=14) 
    plt.ylabel('Slackness ',fontsize=14)  # label the y axis


    plt.legend(fontsize=14)  # add the legend (will default to 'best' location)


########################################### ACCELERATION PLOT ##########################################################

# Ho 2*iterazioni*6 elements list
# I want to get rid of the repetitions (2), so I take every even number
# Now I have iterazioni*6 elements vector
# I want to have 3 elements per row, so I split the vector in chunks of 3 elements
#print("acc_vector",acc_vector)

# Step 1: Take 6 elements, skip the following 6
filtered_elements = []
for i in range(0, len(acc_vector)+1, 12):  # Increment by 12: 6 elements to take, 6 to skip
    filtered_elements.extend(acc_vector[i+6:i+12])

# Step 2: Convert to NumPy array
filtered_array = np.array(filtered_elements)

# Step 3: Reshape the array to have 6 elements per row
acc_array = filtered_array.reshape(-1, 6)




#Prepare multipanel plot 
fig2 = plt.figure(7, figsize=(5, 5))
gs2 = gridspec.GridSpec(4,4)
gs2.update(wspace=0.2, hspace=0.25)

xtr_subsplot2= fig2.add_subplot(gs2[0:4,0:2])

plt.plot( iterations_array[:-1], acc_array[:,0], linestyle='-', label='drone_x_acceleration', color=colors[0], mfc='w', markersize=4) # plot data
#plt.plot(iterations_array,x_Ldot, linestyle='-', label='payload_x_velocity', color=colors2[0], mfc='w', markersize=4) # plot data
plt.plot(iterations_array[:-1],acc_array[:,1], linestyle='-', label='drone_y_acceleration', color=colors[16], mfc='w', markersize=4) # plot data
#plt.plot( iterations_array,y_Ldot,linestyle='-',  label='payload_y_velocity', color=colors2[8], mfc='w', markersize=4) # plot data
plt.plot( iterations_array[:-1],acc_array[:,2],linestyle='-',  label='drone_z_acceleration', color=colors[8], mfc='w', markersize=4) # plot data
#plt.plot( iterations_array,z_Ldot, linestyle='-', label='payload_z_velocity', color=colors2[16], mfc='w', markersize=4) # plot data
   
#plot params
#plt.xlim([0,iterations + 50])
#plt.ylim([-100,100])
#xticks = np.arange(0, 1e4,10)
xticks = np.arange(0,1000.5,100)
yticks = np.arange(-21,3.1,1)
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
plt.ylabel('Drone Acceleration [m/s^2]',fontsize=14)  # label the y axis


plt.legend()  # add the legend (will default to 'best' location)
#plot text and line on top of figure
    
xtr_subsplot2 = fig2.add_subplot(gs2[0:4,2:4])
#plt.plot( iterations_array,x_dot, linestyle='-', label='drone_x_velocity', color=colors[0], mfc='w', markersize=4) # plot data
plt.plot(iterations_array[:-1],acc_array[:,3], linestyle='-', label='payload_x_acceleration', color=colors2[0], mfc='w', markersize=4) # plot data
#plt.plot(iterations_array,y_dot, linestyle='-', label='drone_y_velocity', color=colors[16], mfc='w', markersize=4) # plot data
plt.plot( iterations_array[:-1],acc_array[:,4],linestyle='-',  label='payload_y_acceleration', color=colors2[8], mfc='w', markersize=4) # plot data
#plt.plot( iterations_array,z_dot,linestyle='-',  label='drone_z_velocity', color=colors[8], mfc='w', markersize=4) # plot data
plt.plot( iterations_array[:-1],acc_array[:,5], linestyle='-', label='payload_z_acceleration', color=colors2[16], mfc='w', markersize=4) # plot data
   
#Define tick parameters
xticks2 = np.arange(0,1000.5,100)
yticks2 = np.arange(-21,3.1,1)
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
plt.ylabel('Payload Acceleration [m/s^2]',fontsize=14)  # label the y axis

plt.legend()
plt.show()

################################################ TENSION PLOT ########################################################
"""
# Step 1: Take 6 elements, skip the following 6
#print("(tension_plt) ", (tension_plt))


filtered_elements = []

tension_array = tension_plt[1::2]

# Step 2: Convert to NumPy array
#tension_array = np.array(filtered_elements)

#plot
fig = plt.figure(8, figsize=(5, 5))
plt.plot( iterations_array[:-1], tension_array, linestyle='-', label='tension', color=colors2[16], mfc='w', markersize=4) # plot data

#plt.plot(x_fit, ffit(x_fit), linestyle='-', marker='None', label='fit', color=colors[0], markerfacecolor='white', markersize=8) # plot data

#plot params
xticks = np.arange(0,1000.5,100)
yticks = np.arange(-5,5,0.5)
plt.minorticks_on()

plt.tick_params(direction='in',which='minor', length=5, bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in',which='major', length=10, bottom=True, top=True, left=True, right=True)
plt.xticks(xticks)
plt.yticks(yticks)


#plt.text(1,325, f'y={Decimal(coefs[3]):.4f}x$^3$+{Decimal(coefs[2]):.2f}x$^2$+{Decimal(coefs[1]):.2f}x+{Decimal(coefs[0]):.1f}',fontsize =13)


plt.xlabel('Iterations', fontsize=14) 
plt.ylabel('Cable Tension [N]',fontsize=14)  # label the y axis


plt.legend(fontsize=14)  # add the legend (will default to 'best' location)
"""
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
    rod2.axis = vector(x[i], z[i]+arm_length*np.sin(theta), y[i]+arm_length*np.cos(theta)) + vector(theta_dot, 0, 0)*dt
    rod3.pos = vector(x[i], z[i], y[i])+ vector(x_dot[i], z_dot[i], y_dot[i])*dt
    rod3.axis = vector(x[i], z[i]- arm_length*np.sin(theta), y[i]-arm_length*np.cos(theta)) + vector(theta_dot, 0, 0)*dt
"""
"""
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

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
    propeller1 = np.array([y[:i+1]+arm_length*np.cos(theta[i+1]), z[:i+1]+arm_length*np.sin(theta[i+1])]) 
    propeller2 = np.array([y[:i+1]-arm_length*np.cos(theta[i+1]) , z[:i+1]-arm_length*np.sin(theta[i+1])]) 
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
output_folder = './Oscillation_Hovering/'
output_video = 'Oscillation_Hovering.mp4'

zoom_padding = 2.0  # You can adjust this to be more or less zoomed in
# Drone dimensions
propeller_size = 0.1  # Size of the propeller markers

# Create a 3D plot and save each frame
#for i, pos in enumerate(sim.state_traj[:, :3]):
for i, (drone_pos, payload_pos, angle) in enumerate(zip(sim.state_traj[:, 0:3], sim.state_traj[:, 3:6], sim.state_traj[:, 6])):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Change the camera view angle
    elev = 1  # Elevation angle
    azim = 0#i * (360 / 200)  # Rotate 360 degrees over all frames
    ax.view_init(elev=elev, azim=azim)
    
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
    plt.savefig(f'{output_folder}frame_{i:03d}.png', dpi = 150)
    plt.close(fig)

# Convert the saved frames to a video
frames = []
for i in range(iterations+1):
    frames.append(imageio.imread(f'{output_folder}frame_{i:03d}.png'))

imageio.mimsave(output_video, frames, fps=50)  # Adjust fps as needed

print(f"Video saved as {output_video}")

"""
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

min_x = -3
max_x = +3
min_y = 0
max_y = 7
fig, ax = plt.subplots()
ax.set_xlim([min_x, max_x])
ax.set_ylim([min_y, max_y])
scat = ax.scatter(1, 1)
scat2 = ax.scatter(1, 1)

def animate(i):
    
    scat.set_offsets((y[i], z[i]))
    scat2.set_offsets((y_L[i], z_L[i]))
    #ax.set_ylim(y[i], y[i]+10)
    return scat, scat2 #,ax

ani = animation.FuncAnimation(fig, animate, repeat=False,
                                    frames=len(x) - 1, interval=20)

# To save the animation using Pillow as a gif
writer = animation.PillowWriter(fps=30,
                                 metadata=dict(artist='Me'),
                                 bitrate=1800)
ani.save('scatter.gif', writer=writer)

plt.show()
"""
"""
    # Output:
    # The CSV file named 'file.csv' will be created and three rows of data will be written to it.
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
    plt.plot(sim.state_traj[:, 3:6])
    plt.legend(["x_L", "y_L", "z_L"])
    plt.grid()
    

    plt.figure()
    plt.plot(sim.state_traj[:, 6])# * jnp.pi/180)
    plt.legend(["theta"])
    plt.grid()
   

    plt.figure()
    plt.plot(sim.state_traj[:, 7:10])
    plt.legend(["x_dot", "y_dot", "z_dot"])
    plt.grid()
    

    plt.figure()
    plt.plot(sim.state_traj[:, 10:13])
    plt.legend(["x_Ldot", "y_Ldot", "z_Ldot"])
    plt.grid()
    

    plt.figure()
    plt.plot(sim.state_traj[:, 13])
    plt.legend(["theta dot"])
    plt.grid()
   

    plt.figure()
    plt.plot(jnp.linalg.norm(sim.state_traj[:, 0:3] - sim.state_traj[:, 3:6], axis = 1))
    plt.legend(["Cable_Length"])
    plt.grid()
    

    plt.figure()
    # Plot the input trajectory
    plt.plot(sim.input_traj)
    plt.legend(["u1", "u2"])
    

    plt.figure()
    # Plot the input trajectory
    plt.plot(sim.input_traj[:,0] + sim.input_traj[:,1])
    plt.legend(["u1 + u2"])
    

    plt.figure()
    # Plot the input trajectory
    plt.plot((drone_length *(sim.input_traj[:,0] - sim.input_traj[:,1]))/inertia_slack)
    plt.legend(["torque"])
    plt.show()
"""
