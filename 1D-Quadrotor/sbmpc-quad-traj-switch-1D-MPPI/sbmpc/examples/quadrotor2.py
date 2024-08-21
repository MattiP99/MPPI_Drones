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
sys.path.append('/home/mpiras/MPPI/sbmpc-quad-traj-switch-1D-MPPI/sbmpc')
from sbmpc.model import Model, ModelMjx
from sbmpc.solvers import SbMPC, BaseObjective
from sbmpc.utils.settings import ConfigMPC, ConfigGeneral
from sbmpc.utils.geometry import skew, quat_product, quat2rotm, quat_inverse
import sbmpc.utils.simulation as simulation
import sbmpc.utils.trapezoidal_traj as trapezoidal_traj



#from jax.config import config 
#config.update("jax_debug_nans", True)
#jax.config.update("jax_debug_nans", True)


MODEL = "classic"

# Input max for the force was 1
input_max = jnp.array([2000.0],dtype=jnp.float32)
input_min = jnp.array([0.0],dtype=jnp.float32)

mass = 2.7
mass_payload = 0.25
cable_length = 0.5

gravity = 9.81
inertia = jnp.array([2.3951e-5, 2.3951e-5, 3.2347e-5], dtype=jnp.float32)
inertia_mat = jnp.diag(inertia)

spatial_inertia_mat = jnp.diag(jnp.concatenate([mass*jnp.ones(3, dtype=jnp.float32), inertia]))
spatial_inertia_mat_inv = jnp.linalg.inv(spatial_inertia_mat)

input_hover = jnp.array([(mass+mass_payload)*gravity], dtype=jnp.float32)
nq = 6


def func_slack(state,inputs,csi,csi_dot):        
    acc_L =  - jnp.array([0.,0.,gravity])

    F =  jnp.array([0,0,inputs[0]])
    # Obtain Quadrotor Force Vector
    quad_force_vector = F 

    # Solving for Quadrotor Acceleration
    acc = quad_force_vector/mass - jnp.array([0.,0.,gravity])
    
    """  
    acc =  (1/mass)*jnp.array([0.,0.,inputs[0]]) - jnp.array([0.,0.,gravity])
    """
    print("acc slack",acc)
    print("acc_L slack",acc_L)
    #print("inputs",inputs)

    ######## state_dot ########
    state_dot =  jnp.concatenate([state[6:9],
                            #0.5 * quat_product(quat, ang_vel_quat),
                            state[9:12],
                            #orientation_mat @ acc[:3],
                            acc,
                            acc_L])

        
    return state_dot


def func_taut(state,inputs,csi,csi_dot):

    
    ########## SET ORIGINAL EQUATIONS ###############
    
    print("inputs[0] taut",inputs[0])
    #csi = -(state[3:6] - state[0:3])/cable_length

    #csi_dot = -(state[9:12] - state[6:9])/cable_length
    #jax.debug.print("{csi}", csi = csi)
    #jax.debug.print("{csi_dot}", csi_dot = csi_dot)
    csi_omega = jnp.cross(csi, csi_dot)
    

    F = jnp.array([0,0,inputs[0]])
    quad_force_vector = F 
    quad_centrifugal_f = mass * cable_length * (csi_omega @ csi_omega)
    tension_vector = mass_payload * (-csi.reshape(1,3) @ quad_force_vector + quad_centrifugal_f) * csi.reshape(3,1) / (mass+mass_payload)
    # Solving for Load Acceleration
    acc_L = - jnp.transpose(tension_vector) / mass_payload - jnp.array([0.,0.,gravity])
    # Solving for Quadrotor Acceleration
    acc = (quad_force_vector + jnp.transpose(tension_vector)) / mass - jnp.array([0.,0.,gravity])
    acc_L  = acc_L.reshape(3,)
    acc  = acc.reshape(3,)

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
    state_dot = jnp.concatenate([state[6:9],
                            #0.5 * quat_product(quat, ang_vel_quat),
                            state[9:12],
                            #orientation_mat @ acc[:3],
                            acc,
                            acc_L])
        
    #v_kp1 = state[self.nq:] + self.bt * state_dot[-6:]#self.dynamics(state, inputs)[self.nq:]
        
    return state_dot

def handle_collision_taut(state,d,d_dot, csi,csi_dot):
    # the whole dynamic has to change, also the angular velocity
        print("state handle taut",   state[:6])
        print("velocity handle taut", state[6:])
        print("d handle taut",d)
        print("d_dot handle taut",d_dot)

        #### SET NEW EQUATIONS ###############
    
        

        # TASK:  PROJECT VECTOR V_KP1_PARALLEL_DRONE ON VELOCITY DRONE OR I USE THE INVERSE OF THE EQUATION IN THE PAPER
        cable_direction_projmat = csi.reshape((3,1)) @ csi.reshape((1,3))
        vDrone_proj = cable_direction_projmat @ state[6:9]
        vPayload_proj = cable_direction_projmat @ state[9:12]

        v_kp1_parallel_drone = (mass * vDrone_proj + mass_payload * vPayload_proj)/(mass_payload + mass)
        v1 = v_kp1_parallel_drone + state[6:9] - vDrone_proj
        v2 = v_kp1_parallel_drone + state[9:12] - vPayload_proj

        print("v1",v1)
        print("v2",v2)
        """
        v_kp1_parallel_drone = (1/(mass+mass_payload)) * ( mass* (csi @ jnp.transpose(csi)) * state[6:9] + mass_payload * (csi @ jnp.transpose(csi)) * state[9:12])
        

       
        print("v_kp1_parallel_drone taut",v_kp1_parallel_drone)
        v_kp1_parallel_payload =  v_kp1_parallel_drone

        v_orthogoal_drone = state[6:9] - state[6:9]*(csi @ csi)
        v_orthogoal_payload = state[9:12] - state[6:9]*(csi @ csi)
        
        v =   v_kp1_parallel_drone + v_orthogoal_drone #+ proj_of_v_on_v_orth 
        v_payload =  v_kp1_parallel_payload + v_orthogoal_payload #+ proj_of_v_payload_on_v_payload_orth 
        """           
            
        v_kp1 = state[6:12]
        v_kp1 =  v_kp1.at[0:3].set(v1) 

        #v_kp1 = v_kp1 + v_kp1.at[3:6].set(v_payload)
        v_kp1 = v_kp1.at[3:6].set(v2) 
        print("v_kp1 taut",v_kp1)
        #print("v",v)
        #print("state[:self.nq]",state[:nq])
        

        return  v_kp1

def handle_collision_slack( state,d,d_dot, csi,csi_dot):
    v_kp1 = state[nq:nq+6]
    print("v_kp1 slack",v_kp1)
    print("d handle slack",d)
    print("d_dot handle slack",d_dot)
    return  v_kp1


def handle_collision(state,d,d_dot, csi,csi_dot):
    #condition = jnp.logical_and(d  >  cable_length - 0.001, d_dot > -0.001)
    condition = (d  >  cable_length - 0.001) &  (d_dot > -0.001)

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
    ########## YOU MODIFIED CSI and CSI_DOT NNNNNNBBBBBBB ################ 
    #csi = (state[3:6] - state[0:3])/cable_length
    csi = (state[3:6] - state[0:3])/jnp.linalg.norm(state[3:6] - state[0:3])
    #csi_dot = (state[9:12] - state[6:9])/cable_length
    csi_dot = (((state[9:12] - state[6:9])*jnp.linalg.norm(state[3:6] - state[0:3]))-((state[3:6] - state[0:3]) * (state[3:6] - state[0:3]) * (1/jnp.linalg.norm(state[3:6] - state[0:3])) * (state[9:12] - state[6:9])))/(jnp.linalg.norm(state[3:6] - state[0:3]))**2
  
    print("state[3:6] 1 ",state[3:6])
    print("state[0:3] 1 ",state[0:3])
    print("csi 1 ",csi)
    print("csi_dot 1 ",csi_dot)
    
    # In the original they use
    d = jnp.linalg.norm(state[3:6] - state[0:3] )
    
    #d = jnp.linalg.norm(state[0:3] - state[3:6])
    #d_dot = jnp.linalg.norm((state[0:3] - state[3:6])/((jnp.linalg.norm(state[0:3] - state[3:6]))))

    # Consider time partial derivative !!!!!!!
    # In the original they use
    d_dot = (state[3:6] - state[0:3] ) @ (state[9:12] - state[6:9] )/((jnp.linalg.norm(state[3:6] - state[0:3])))
    #d_dot = (state[0:3] - state[3:6]) @ (state[6:9] - state[9:12])/((jnp.linalg.norm(state[0:3] - state[3:6])))

    # Inputs are said to be total force and total torque but I already have this computation in order 
    # to consider as inputs the f forces and moments in the body frame

    #### ORIGINAL EQUATIONS FOR TOTAL FORCE AND TOTAL TORQUE ####
    #total_force = jnp.array([0., 0., inputs[0]]) - mass*gravity*orientation_mat[2, :]  # transpose + 3rd col = 3rd row
    #total_torque = 1e-3*inputs[1:4] - skew(ang_vel) @ inertia_mat @ ang_vel  # multiplication by normalization factor

    #quat = state[3:7]
    #ang_vel = state[10:13]
    #orientation_mat = quat2rotm(quat)
    
    # I'm not convinced about orientation_mat
    ##################  PROBLEM WITH THE INPUTS ###################### 
    ################### I CANNOT PUT INPUT[]   ######################

    # I think that it is because what it is written in update function
    # The problem is that I'm passing to the integrate a single number right now so inputs it is just a single value

    #print("inputs",inputs)
    #total_force = jnp.array([0., 0., inputs]) - mass*jnp.array([0., 0., gravity],dtype=jnp.float32)
    #total_torque = 1e-3*inputs[1:4] - skew(ang_vel) @ inertia_mat @ ang_vel  # multiplication by normalization factor
    
    
    
    
    ######## state_dot ########
    #return jnp.select([d - cable_length  < 0.001,  #### Cable Slack ####
    #                     jnp.logical_and(d - cable_length >= 0.001 , d_dot >= -0.01)],
    #                      #### Cable Taut ####
    #                     [func_slack(state,inputs),
    #                     func_taut(state,inputs,csi,csi_dot)]
    #                     )
    condition = jnp.logical_and(d  > cable_length - 0.001, d_dot > -0.001)
    #condition = (d  >  cable_length - 0.001) #&  (d_dot > 0.001)
    #condition = (d  > cable_length - 0.001)
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
        #att_err = quat_product(quat_inverse(state[3:7]), state_ref[3:7])[1:4]
        vel_err = state[6:9] - state_ref[6:9]
        #ang_vel_err = state[16:] - state_ref[13:]

        return pos_err,  vel_err

    def running_cost(self, state: jnp.array, inputs: jnp.array, reference) -> jnp.float32:
        state_ref = reference[:12]
        input_ref = reference[12:]
        
        #pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, state_ref)
        pos_err, vel_err = self.compute_state_error(state, state_ref)    
        return (4 * pos_err.transpose() @ pos_err +
                #0.01 * att_err.transpose() @ att_err +
                1 * vel_err.transpose() @ vel_err +
                #0.1 * ang_vel_err.transpose() @ ang_vel_err +
                #0.5*(inputs-input_ref).transpose() @ (inputs-input_ref))
                0.5*(inputs-input_hover).transpose() @ (inputs-input_hover))

    def final_cost(self, state, state_ref):
        #pos_err, att_err, vel_err, ang_vel_err = self.compute_state_error(state, reference[:13])
        pos_err, vel_err = self.compute_state_error(state, reference[:12])
        return (100 * pos_err.transpose() @ pos_err +
                #0.1 * att_err.transpose() @ att_err +
                100 * vel_err.transpose() @ vel_err)
                #1 * ang_vel_err.transpose() @ ang_vel_err)



class Simulation(simulation.Simulator):
    def __init__(self, initial_state, model, controller, num_iterations):
        super().__init__(initial_state, model, controller, num_iterations)
        
        ############# TRAJECTORY GENERATION ################
        q_des = jnp.array([0.0, 0.0, 9.0, 0.0, 0.0, 8.5], dtype=jnp.float32)  # hovering position
        #self.reference = jnp.zeros((T, x_init.size + input_hover.size),dtype=jnp.float32)
        #calculator = trapezoidal_traj.Trapeizoidal_Trajectory(q_init[0:3], q_des[0:3], 30, self.num_iter + self.controller.horizon + 1)
        #self.reference = calculator.compute_chirp_trajectory()
        

        ################# FIXED REFERENCE ##################
        self.reference = jnp.concatenate([q_des, jnp.zeros(self.model.nv, dtype=jnp.float32)], axis=0) 
        self.reference = jnp.concatenate([self.reference, jnp.array([(mass+mass_payload)*gravity], dtype=jnp.float32)], axis=0) 
        
        
    
    
    def update(self):
        q_des = jnp.array([0.0, 0.0, 9.0, 0.0, 0.0, 8.5], dtype=jnp.float32)  # hovering position
        x_des = jnp.concatenate([q_des, jnp.zeros(self.model.nv, dtype=jnp.float32)], axis=0)
        # Compute the optimal input sequence
        
        #print("reference:", reference)
        
        # Compute the optimal input sequence
        time_start = time.time_ns()

        ##### FIXED REFERENCE #####
        input_sequence = self.controller.compute_control_action(self.current_state_vec(), self.reference, num_steps=1).block_until_ready()
        
        ##### TAJECTORY REFERENCE #####

        #input_sequence = self.controller.compute_control_action(self.current_state_vec(), self.reference[self.iter:self.iter + self.controller.horizon ,:], num_steps=1).block_until_ready()

        print("computation time: {:.3f} [ms]".format(1e-6 * (time.time_ns() - time_start)))
        ctrl = input_sequence[:self.model.nu]

        self.input_traj[self.iter, :] = ctrl

        
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
        

        print(" in update FUNCTION")
        print("d",d)
        print("d_dot",d_dot)
        print("csi",csi)
        print("csi_dot",csi_dot)
        print("csi2",csi2)
        print("csi_dot2",csi_dot2)

        # Simulate the dynamics
        

        v_kp1 = handle_collision(self.current_state, d,d_dot, csi,csi_dot)
        self.current_state = self.current_state.at[nq:nq+6].set(v_kp1)
        #self.current_state = self.current_state.at[3:6].set(state_L)
        print("current state2",self.current_state)

        self.current_state   = self.model.integrate(self.current_state, ctrl, self.controller.dt)
        #self.current_state   = self.model.integrate(self.current_state, ctrl, 0.001)

        print("current state1",self.current_state)
        # Check for collision and handle it
        
        self.state_traj[self.iter + 1, :] = self.current_state_vec()

if __name__ == "__main__":

    mpc_config = ConfigMPC(0.001,
                           25,
                           jnp.array([0.2]),
                           num_parallel_computations=10000,
                           initial_guess=input_hover)
    gen_config = ConfigGeneral("float32", jax.devices("cpu")[0])

    if MODEL == "classic":
        system = Model(quadrotor_dynamics, 6, 6, 1, [input_min, input_max])
        q_init = jnp.array([0.0, 0.0, 5.0, 0.0, 0.0, 5.1], dtype=jnp.float32)  # hovering position
        #q_init = jnp.array([0.0, 0.0, 9.0, 0.0, 0.0, 9.1], dtype=jnp.float32)  # hovering position
        

        x_init = jnp.concatenate([q_init, jnp.zeros(system.nv, dtype=jnp.float32)], axis=0) #jnp.array([0.01,0.,0.,0.,0.,0.],)],axis = 0)#
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

    
    T = 1000+25+1
    
    #dim = x_init.size + input_hover.size = 17
    
    q_des = jnp.array([0.0, 0.0, 9.0, 0.0, 0.0, 8.5], dtype=jnp.float32)  # hovering position
    reference = jnp.concatenate([q_des, jnp.zeros(7, dtype=jnp.float32)], axis=0) 
    

    #reference = jnp.zeros((T, x_init.size + input_hover.size),dtype=jnp.float32)
    #calculator = trapezoidal_traj.Trapeizoidal_Trajectory(q_init[0:3], q_des[0:3], 10, T)
    #reference = calculator.compute_linear_trajectory()
    #for i in range(T):
    #    reference = reference.at[i,0:3].set(trajectory[i,:])
    #    reference = reference.at[i,3].set(1)
    #    reference = reference.at[i,-4].set(mass*gravity) 

    #reference = jnp.concatenate((x_init, input_hover))

    # dummy for jitting
    input_sequence = solver.compute_control_action(x_init, reference).block_until_ready()

    # Setup and run the simulation
    sim = Simulation(state_init, system, solver, 1000)
    #sim = Simulation(state_init, system, 1000)
    sim.simulate()

    ax = plt.figure().add_subplot(projection='3d')
    # Plot x-y-z position of the robot
    ax.plot(sim.state_traj[:, 0], sim.state_traj[:, 1],sim.state_traj[:, 2])
    ax.plot(sim.state_traj[:, 3], sim.state_traj[:, 4],sim.state_traj[:, 5])
    
    plt.figure()
    plt.plot(sim.state_traj[:, 0:3])
    plt.plot(sim.state_traj[:, 3:6])
    plt.legend(["x", "y", "z","x_L", "y_L", "z_L"])
    plt.grid()

    plt.figure()
    plt.plot(sim.state_traj[:, 6:9])
    plt.legend(["xdot", "ydot", "zdot"])
    plt.grid()

    plt.figure()
    plt.plot(sim.state_traj[:, 9:12])
    plt.legend(["x_Ldot", "y_Ldot", "z_Ldot"])
    plt.grid()

    plt.figure()
    plt.plot(sim.state_traj[:, 2] - sim.state_traj[:, 5])
    plt.legend(["diff_z"])
    plt.grid()
    plt.show()


    plt.figure()
    plt.plot(sim.input_traj)
    plt.legend(["F"])
    plt.grid()
    plt.show()

    print("valueDrone18",sim.state_traj[18, 0], sim.state_traj[18, 1],sim.state_traj[18, 2])
    print("valuePayload18",sim.state_traj[18, 3], sim.state_traj[18, 4],sim.state_traj[18, 5])

    print("valueDrone200",sim.state_traj[200, 0], sim.state_traj[200, 1],sim.state_traj[200, 2])
    print("valuePayload200",sim.state_traj[200, 3], sim.state_traj[200, 4],sim.state_traj[200, 5])

    print("valueDrone320",sim.state_traj[320, 0], sim.state_traj[320, 1],sim.state_traj[320, 2])
    print("valuePayload320",sim.state_traj[320, 3], sim.state_traj[320, 4],sim.state_traj[320, 5])

    print("valueDrone450",sim.state_traj[450, 0], sim.state_traj[450, 1],sim.state_traj[450, 2])
    print("valuePayload450",sim.state_traj[450, 3], sim.state_traj[450, 4],sim.state_traj[450, 5])

    print("valueDrone1000",sim.state_traj[1000, 0], sim.state_traj[1000, 1],sim.state_traj[1000, 2])
    print("valuePayload1000",sim.state_traj[1000, 3], sim.state_traj[1000, 4],sim.state_traj[1000, 5])