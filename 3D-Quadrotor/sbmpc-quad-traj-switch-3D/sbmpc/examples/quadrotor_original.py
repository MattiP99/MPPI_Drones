import time, os

import jax
import jax.numpy as jnp
from scipy import signal
os.environ['XLA_FLAGS'] = (
        '--xla_gpu_triton_gemm_any=True '
    )

import matplotlib.pyplot as plt
from matplotlib import rc
import sys
sys.path.append('/home/mpiras/MPPI/sbmpc-quad-traj-switch-3D/sbmpc')
from sbmpc.model import Model, ModelMjx
from sbmpc.solvers import SbMPC, BaseObjective
from sbmpc.utils.settings import ConfigMPC, ConfigGeneral
from sbmpc.utils.geometry import skew, quat_product, quat2rotm, quat_inverse, rotation_matrix_around_x
import sbmpc.utils.simulation as simulation
import sbmpc.utils.trapezoidal_traj as trapezoidal_traj




MODEL = "classic"

input_max = jnp.array([60,24,24,24])
input_min = jnp.array([0,0,0,0])

mass = 2.7
mass_payload = 0.25
cable_length = 0.5
arm_length = 0.4
gravity = 9.81
inertia = jnp.array([2.45e-2, 2.45e-2, 1.383e-2], dtype=jnp.float32)
inertia_mat = jnp.diag(inertia)

spatial_inertia_mat = jnp.diag(jnp.concatenate([mass*jnp.ones(3, dtype=jnp.float32), inertia]))
spatial_inertia_mat_inv = jnp.linalg.inv(spatial_inertia_mat)



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
    print("csi SLACK",csi)
    print("state SLACK",state[0:6])
    print("vel SLACK",state[7:13])       

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
                            #0.5 * quat_product(quat, ang_vel_quat),
                            state[13:16],
                            0.5 * quat_product(quat, ang_vel_quat),
                            #orientation_mat @ acc[:3],
                            orientation_mat @ acc[:3],
                            acc_L,
                            acc[3:6]])

    print("state_dot slack",state_dot)
    return state_dot

def baumgarte_stabilization(state):
    
    alpha = 10
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
    direction = diff_pos / current_distance if current_distance != 0 else jnp.array([0.0, 0.0, 0.0])
    return correction_term * direction


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

    quat = state[6:10]
    ang_vel = state[16:19]
    ang_vel_quat = jnp.array([0., state[16], state[17], state[18]])
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
    print("inputs[1] taut",inputs[1])
    print("F taut",total_force)
    print("THETA taut",state[6])
    # Obtain Quadrotor Force Vector

    # ALREADY IN RADIANTS??
     
    #quad_force_vector = F * rotation_matrix_around_x(state[6]) @ e3  

    #print("quad_force_vector taut",total_force_world)
    #quad_centrifugal_f = mass * cable_length * (csi_omega @ csi_omega)
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
            

            is_slack = False
    return state, is_slack
    
def check_dynamics(state,inputs,csi,csi_dot):
    print("INSIDE CHECK DYNAMICS")
    print("e3 CHECK DYNAMICS",e3 )
    print("csi CHECK DYNAMICS",csi )
    global is_slack_dynamic
    print("is_slack_dynamic CHECK DYNAMICS",is_slack_dynamic )
    
    if  (is_slack_dynamic == False):# & (d_dot >= 0.001):
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
            if jnp.dot(tension_vector,csi) >= 0.0:
                print("INSIDE TAUT TO SLACK CHECK DYNAMICS", tension_vector)
                is_slack_dynamic = True

#@jax.jit
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

    #csi_dot = (state[10:13] - state[13:16] )/cable_length
    csi = (state[0:3] - state[3:6] )/jnp.linalg.norm(state[0:3] - state[3:6] )
    csi_dot = 1/jnp.linalg.norm(state[0:3] - state[3:6] ) * (jnp.diag(jnp.array([1,1,1])) - ((csi).reshape(3,1) @ (csi).reshape(1,3))) @ (state[10:13] - state[13:16]).reshape(3,1)
    csi_dot = csi_dot.reshape(3,)
    print("csi dynamics",csi)
    print("csi_dot dynamics",csi_dot)
    print("csi NORM dynamics",jnp.linalg.norm(csi))
    print("csi_dot NORM dynamics",jnp.linalg.norm(csi_dot))
    print("INPUTS DYNAMICS",inputs)
    

    d  = jnp.linalg.norm(state[3:6] - state[0:3])
    #d_dot = ((state[3:6]  - state[0:3]).transpose() @ (state[10:13] - state[7:10]))/((jnp.linalg.norm(state[3:6] - state[0:3])))
    # Compute the dot product of v and dv/dt
    v_dot_dvdt = jnp.dot(state[3:6]  - state[0:3], state[13:16] - state[10:13])
    
    # Compute the derivative of the norm
    d_dot = v_dot_dvdt / d
    # Inputs are said to be total force and total torque but I already have this computation in order 
    # to consider as inputs the f forces and moments in the body frame
    print("d_dot dynamic", d_dot)
    global is_slack_dynamic
   
    if  (is_slack_dynamic == False):# & (d_dot >= 0.001):
            result_taut = func_taut(state,inputs,csi,csi_dot)
            acc_vector.extend(result_taut[10:16])
            return result_taut
            
    else:
            result_taut = func_slack(state,inputs,csi,csi_dot)
            acc_vector.extend(result_taut[10:16])
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
        #q_des = jnp.array([0.0, 3.0], dtype=jnp.float32)  # hovering position
        #x_des = jnp.concatenate([q_des, jnp.zeros(self.model.nv, dtype=jnp.float32)], axis=0)
        
        #print("reference:", reference)
        
        # Compute the optimal input sequence
        #time_start = time.time_ns()
        #input_sequence = self.controller.compute_control_action(self.current_state_vec(), self.reference[self.iter:self.iter + self.controller.horizon ,:], num_steps=1).block_until_ready()
        
        # WHICH SHAPE DOES IT HAVE?
        #x = jnp.arange(0, 500, 0.1) 
        #CASE 2- OSCILLLATION HORIZONTALLY
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
        #input_sequence =  jnp.array([(mass+mass_payload)*gravity , 0.0,0.0,0.0 ])

        # INPUT SINUSOIDAL
        #input_sequence =  jnp.array([(mass+mass_payload)*gravity + 10 * jnp.sin(0.5*x) , 0.0 * x, 0.0 * x , 0.0 * x])
        
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
        #
        #input_sequence_1 =  jnp.array([((mass+mass_payload)*gravity + 1)/2 + (20 * jnp.sin(0.1*x1))/2 ,
        #                              ((mass+mass_payload)*gravity + 1)/2 - (20 * jnp.sin(0.1*x1))/2])
        #input_sequence_1 = input_sequence_1.reshape(1000,2)
        #input_sequence = jnp.concatenate([input_sequence_1,input_sequence_2],axis=0)
        

        # INPUT ASCENDING WHILE OSCILLATING HORIZONTALLY 3
        force = ((mass+mass_payload)*gravity) * jnp.ones((500,1))
        torque_x=  0 * jnp.ones((500,1))
        torque_y=  0 * jnp.ones((500,1))
        torque_z=  0 * jnp.ones((500,1))
        input_sequence1 =  jnp.concatenate([force,torque_x,torque_y,torque_z],axis=1)
        input_sequence1 = input_sequence1.reshape(500,4)
        force = (-(mass+mass_payload)*gravity)-0.5 * jnp.ones((500,1))
        torque_x=  0 * jnp.ones((500,1))
        torque_y=  0 * jnp.ones((500,1))
        torque_z=  0 * jnp.ones((500,1))
        input_sequence2 =  jnp.concatenate([force,torque_x,torque_y,torque_z],axis=1)
        input_sequence2 = input_sequence2.reshape(500,4)
        input_sequence = jnp.concatenate([input_sequence1,input_sequence2],axis=0)
        #input_sequence_2 =  jnp.array([((mass+mass_payload)*gravity + 1) * jnp.ones((900)),
        #                              -1 * 0.25 * jnp.sin(0.5*x2)/ gravity,
        #                              -1 * 0.25 * jnp.sin(0.5*x2)/ gravity,
        #                              0 * jnp.ones((900))])
        #input_sequence_2 = input_sequence_2.reshape(900,4)
        #input_sequence = jnp.concatenate([input_sequence_1,input_sequence_2],axis=0)

        # INPUT CASE SWITCHING
        #input_sequence_1 =  jnp.array([(mass+mass_payload)*gravity + 8 , 0,0,0 ])
        #input_sequence_1 = jnp.tile(input_sequence_1, (500, 1)) 
        #input_sequence_2 =  jnp.array([-(mass+mass_payload)*gravity - 8  , 0,0,0 ])
        #input_sequence_2 = jnp.tile(input_sequence_2, (500, 1)) 
        #input_sequence = jnp.concatenate([input_sequence_1,input_sequence_2],axis=0)

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
        d = jnp.linalg.norm(self.current_state[3:6] - self.current_state[0:3])
       
        # Compute the dot product of v and dv/dt
        v_dot_dvdt = jnp.dot(self.current_state[3:6]  - self.current_state[0:3], self.current_state[13:16] - self.current_state[10:13])
        
        # Compute the derivative of the norm
        d_dot = v_dot_dvdt / d
        
        #csi = (self.current_state[3:6] - self.current_state[0:3])/cable_length
        
        # Simulate the dynamics
        csi = (self.current_state[0:3] - self.current_state[3:6] )/jnp.linalg.norm(self.current_state[0:3]- self.current_state[3:6] )
        csi_dot = 1/jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6] ) * (jnp.diag(jnp.array([1,1,1])) - ((csi).reshape(3,1) @ (csi).reshape(1,3))) @ (self.current_state[10:13] - self.current_state[13:16]).reshape(3,1)
        csi_dot = csi_dot.reshape(3,)
        #csi_dot = (self.current_state[10:13]  - self.current_state[13:16] )/cable_length

        #csi_dot = (self.current_state[10:13] - self.current_state[7:10])/jnp.linalg.norm(self.current_state[10:13] - self.current_state[7:10])
        print("csi update",csi)
        print("csi_dot update",csi_dot)
        print("csi NORM update",jnp.linalg.norm(csi))
        print("csi_dot NORM update",jnp.linalg.norm(csi_dot))
        
        check_dynamics(self.current_state, ctrl,csi,csi_dot)
        
        self.current_state   = self.model.integrate(self.current_state, ctrl, dt)
        
        csi = (self.current_state[0:3] - self.current_state[3:6] )/jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6] )

        #v_dot_dvdt = jnp.dot((self.current_state[0:3] - self.current_state[3:6]), (self.current_state[7:10]  - self.current_state[10:13] ))
        #d_v_norm_dt = v_dot_dvdt / jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6])
        
        # Apply the formula: (dv/dt * v_norm - v * d(v_norm)/dt) / v_norm^2
        #csi_dot = ((self.current_state[7:10]  - self.current_state[10:13] ) * jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6]) - (self.current_state[0:3] - self.current_state[3:6])  * d_v_norm_dt) / (jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6]) ** 2)
        #csi_dot = (self.current_state[7:10]  - self.current_state[10:13])/cable_length
        
        csi_dot = 1/jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6] ) * (jnp.diag(jnp.array([1,1,1])) - ((csi).reshape(3,1) @ (csi).reshape(1,3))) @ (self.current_state[10:13] - self.current_state[13:16]).reshape(3,1)
        csi_dot = csi_dot.reshape(3,)
        
        d = jnp.linalg.norm(self.current_state[3:6] - self.current_state[0:3])
       
        # Compute the dot product of v and dv/dt
        v_dot_dvdt = jnp.dot(self.current_state[3:6]  - self.current_state[0:3], self.current_state[13:16] - self.current_state[10:13])
        
        # Compute the derivative of the norm
        d_dot = v_dot_dvdt / d
        print("is_slack_dynamic AFTER CHECK DYNAMICS",is_slack_dynamic)
        print("is_slack AFTER CHECK DYNAMICS",self.is_slack)
        
        self.current_state, self.is_slack = check_distance(self.current_state, csi,csi_dot,d, d_dot)
        is_slack_dynamic = self.is_slack
        print("is_slack_dynamic AFTER CHECK DISTANCE",is_slack_dynamic)
        print("is_slack AFTER CHECK DISTANCE",self.is_slack)
        #self.is_slack = check_distance(self.current_state, csi,csi_dot,self.is_slack)
        
        
        #print("cureent state1",self.current_state)
        # Check for collision and handle it
        print("IS_SLALCK UDPATE AFTER",self.is_slack)
        print("current_state UDPATE AFTER",self.current_state)
        

        # After integration bt = bt + dt or bt = 0 .
        #self.bt = bt
        #self.state_traj[self.iter + 1, :] = self.current_state_vec()
        self.state_traj[self.iter + 1, :] = self.current_state_vec()
        

if __name__ == "__main__":

    mpc_config = ConfigMPC(0.02,
                           50,
                           #jnp.array([0.2, 0.3, 0.3, 0.15]),
                           jnp.array([0.2,0.2,0.2,0.2]),
                           num_parallel_computations=10000,
                           initial_guess=input_hover)
    gen_config = ConfigGeneral("float32", jax.devices("cpu")[0])

    if MODEL == "classic":
        # HERE WHAT VALUE OF nq,nv,nu SHOULD I USE?
        system = Model(quadrotor_dynamics, 10, 9, 4, [input_min, input_max])
        q_init = jnp.array([0.0, 0.0, 5.0, 0.0, 0.0, 4.8, 1.,0.,0.,0.], dtype=jnp.float32)  # hovering position
        x_init = jnp.concatenate([q_init, jnp.array([0,0,0,0,0,0,0,0,0])])#(system.nv, dtype=jnp.float32)], axis=0)
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

    is_slack_dynamic = is_slack
    sim = Simulation(state_init, system, is_slack, 1000)
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
    plt.legend(["quaternion"])
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
    plt.legend(["omega1","omega2","omega3"])
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
    

