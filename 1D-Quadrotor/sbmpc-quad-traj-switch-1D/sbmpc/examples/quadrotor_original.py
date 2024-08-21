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
sys.path.append('/home/mpiras/MPPI/sbmpc-quad-traj-switch-1D/sbmpc')
from sbmpc.model import Model, ModelMjx
from sbmpc.solvers import SbMPC, BaseObjective
from sbmpc.utils.settings import ConfigMPC, ConfigGeneral
from sbmpc.utils.geometry import skew, quat_product, quat2rotm, quat_inverse
import sbmpc.utils.simulation as simulation
import sbmpc.utils.trapezoidal_traj as trapezoidal_traj



MODEL = "classic"

input_max = jnp.array([1])
input_min = jnp.array([0])

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

dt = 0.001
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
def func_slack(state,inputs,csi,csi_dot):        
    acc_L =  - jnp.array([0.,0.,gravity])

    F =  jnp.array([0,0,inputs])
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
    print("INSIDE TAUT")
    print("csi taut",csi)
    print("state taut",state[0:6])
    print("vel taut",state[6:12])
    
    #csi = -(state[3:6] - state[0:3])/cable_length

    #csi_dot = -(state[9:12] - state[6:9])/cable_length
    csi_omega = jnp.cross(csi, csi_dot)
    

    F = jnp.array([0,0,inputs])
    quad_force_vector = F 
    quad_centrifugal_f = mass * cable_length * (csi_omega @ csi_omega)

    print("quad_centrifugal_f taut",quad_centrifugal_f)

    tension_vector = mass_payload * (-csi.reshape(1,3) @ quad_force_vector + quad_centrifugal_f) * csi.reshape(3,1) / (mass+mass_payload)
    print("tension_vectortaut",tension_vector)
    # Solving for Load Acceleration
    ciao = -csi.reshape(1,3) @ quad_force_vector
    print("ciao",ciao)
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
"""
def func_taut(state,inputs,csi,csi_dot):

    
    ########## SET ORIGINAL EQUATIONS ###############
    acc_L = 1/(mass+mass_payload) * ((jnp.dot(csi, jnp.array([0.,0.,inputs])) - mass*cable_length*jnp.dot(csi_dot, csi_dot)) * csi) - jnp.array([0,0,gravity])
    #acc = spatial_inertia_mat_inv @ jnp.concatenate([total_force, total_torque])
    primo = jnp.dot(csi, jnp.array([0.,0.,inputs]))
    secondo = mass*cable_length*jnp.dot(csi_dot, csi_dot)
    print("acc_L taut",acc_L)
    print("cazzo taut",primo)
    print("dio taut",secondo)
    ################ Problem with cross product in 2 dimensions !!!!!!!!!!!!!!!!!!!!!!!!! ###############
    csi_ddot = 1/(mass*cable_length) *  jnp.cross(csi,jnp.cross(csi,jnp.array([0.,0.,inputs])))  -  jnp.dot(csi_dot,csi_dot) * csi

    ## In this case the first term doen't have to be trasposed in world frame since theh drone is always straight ###
    acc = acc_L -  cable_length * csi_ddot
    
    state_dot = jnp.concatenate([state[6:9],
                            #0.5 * quat_product(quat, ang_vel_quat),
                            state[9:12],
                            #orientation_mat @ acc[:3],
                            acc,
                            acc_L])
        
    #v_kp1 = state[self.nq:] + self.bt * state_dot[-6:]#self.dynamics(state, inputs)[self.nq:]
        
    return state_dot
"""

def handle_collision(state,d,d_dot, csi,csi_dot):
    
    if (d  > cable_length - 0.001) & (d_dot > -0.001):   
        
        cable_direction_projmat = csi.reshape((3,1)) @ csi.reshape((1,3))
        vDrone_proj = cable_direction_projmat @ state[6:9]
        vPayload_proj = cable_direction_projmat @ state[9:12]

        v_kp1_parallel_drone = (mass * vDrone_proj + mass_payload * vPayload_proj)/(mass_payload + mass)
        v1 = v_kp1_parallel_drone + state[6:9] - vDrone_proj
        v2 = v_kp1_parallel_drone + state[9:12] - vPayload_proj
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
        
        

        return  v_kp1
    else:
        
        v_kp1 = state[nq:nq+6]
        return  v_kp1

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
    #csi_dot = (state[9:12] - state[6:9])/cable_length
    csi = (state[3:6] - state[0:3])/jnp.linalg.norm(state[3:6] - state[0:3])
    csi_dot = (((state[9:12] - state[6:9])*jnp.linalg.norm(state[3:6] - state[0:3]))-((state[3:6] - state[0:3]) * (state[3:6] - state[0:3]) * (1/jnp.linalg.norm(state[3:6] - state[0:3])) * (state[9:12] - state[6:9])))/(jnp.linalg.norm(state[3:6] - state[0:3]))**2
   
    d = jnp.linalg.norm(state[3:6] - state[0:3] )
    d_dot = (state[3:6] - state[0:3] ) @ (state[9:12] - state[6:9] )/((jnp.linalg.norm(state[3:6] - state[0:3])))
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
    if (d  > cable_length - 0.001) & (d_dot > -0.001):
        return func_taut(state,inputs,csi,csi_dot)
    else:  
         return func_slack(state,inputs,csi,csi_dot)
    
    #elif jnp.logical_and(d - cable_length >= 0.001 , d_dot >= -0.01):   
    #     return func_taut(state,inputs,csi,csi_dot)


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
        state_ref = reference[:13]
        input_ref = reference[13:]
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
    def __init__(self, initial_state, model, num_iterations):
        super().__init__(initial_state, model, num_iterations)

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
        x = jnp.arange(0, 100, 0.1) 
         
        #input_sequence = 0.5*jnp.sin(0.05*x)
        #if(self.iter >= 300) & (self.iter < 325):
        #    input_sequence = 100
        input_sequence = (mass+mass_payload)*gravity #+ 1000*jnp.sin(0.1*x)
        #input_sequence = (mass+mass_payload)*gravity + signal.chirp(x, f0=(mass+mass_payload)*gravity + 100, f1=650, t1=1000, method='hyperbolic')

        #input_sequence = (mass+mass_payload)*gravity 
        #print("computation time: {:.3f} [ms]".format(1e-6 * (time.time_ns() - time_start)))

        
        ctrl = input_sequence#[self.iter]
        self.input_traj[self.iter, :] = ctrl
        #d = jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6])
        #d_dot = jnp.linalg.norm((self.current_state[0:3] - self.current_state[3:6])/((jnp.linalg.norm(self.current_state[0:3] - self.current_state[3:6]))))
        d = jnp.linalg.norm(self.current_state[3:6] - self.current_state[0:3])
        d_dot = (self.current_state[3:6] - self.current_state[0:3] ) @ (self.current_state[9:12] - self.current_state[6:9] )/((jnp.linalg.norm(self.current_state[3:6] - self.current_state[0:3])))

        #csi = (self.current_state[3:6] - self.current_state[0:3])/cable_length
        #csi_dot = (self.current_state[9:12] - self.current_state[6:9])/cable_length
        # Simulate the dynamics
        csi = (self.current_state[3:6] - self.current_state[0:3])/jnp.linalg.norm(self.current_state[3:6] - self.current_state[0:3])
        csi_dot = ((self.current_state[9:12] - self.current_state[6:9])*jnp.linalg.norm(self.current_state[3:6] - self.current_state[0:3])-((self.current_state[3:6] - self.current_state[0:3]) * (self.current_state[3:6] - self.current_state[0:3]) * (1/jnp.linalg.norm(self.current_state[3:6] - self.current_state[0:3])) * (self.current_state[9:12] - self.current_state[6:9])))/(jnp.linalg.norm(self.current_state[3:6] - self.current_state[0:3]))**2
        

        v_kp1 = handle_collision(self.current_state, d,d_dot, csi,csi_dot)
        self.current_state = self.current_state.at[nq:nq+6].set(v_kp1)
        #self.current_state = self.current_state.at[3:6].set(state_L)
        #print("cureent state2",self.current_state)

        self.current_state   = self.model.integrate(self.current_state, ctrl, dt)
        #print("cureent state1",self.current_state)
        # Check for collision and handle it
        
        

        # After integration bt = bt + dt or bt = 0 .
        #self.bt = bt
        self.state_traj[self.iter + 1, :] = self.current_state_vec()
        

if __name__ == "__main__":

    mpc_config = ConfigMPC(0.02,
                           25,
                           #jnp.array([0.2, 0.3, 0.3, 0.15]),
                           jnp.array([0.2]),
                           num_parallel_computations=10000,
                           initial_guess=input_hover)
    gen_config = ConfigGeneral("float32", jax.devices("cpu")[0])

    if MODEL == "classic":
        # HERE WHAT VALUE OF nq,nv,nu SHOULD I USE?
        system = Model(quadrotor_dynamics, 6, 6, 1, [input_min, input_max])
        q_init = jnp.array([0.0, 0.0,5.0, 0.0, 0.0, 5.1], dtype=jnp.float32)  # hovering position
        x_init = jnp.concatenate([q_init, jnp.array([0,0,0,0,0,0])])#(system.nv, dtype=jnp.float32)], axis=0)
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

   
    
    sim = Simulation(state_init, system, 1000)
    sim.simulate()

    
    
    ax = plt.figure().add_subplot(projection='3d')
    
    # Plot x-y-z position of the robot
    ax.plot(sim.state_traj[:, 0], sim.state_traj[:, 1],sim.state_traj[:, 2])
    ax.plot(sim.state_traj[:, 3], sim.state_traj[:, 4],sim.state_traj[:, 5], color  = 'black')
    plt.show()
    
    plt.plot(sim.state_traj[:, 0:3])
    plt.plot(sim.state_traj[:, 3:6])
    plt.legend(["x", "y", "z" , "x_L", "y_L", "z_L"])
    plt.grid()
    plt.show()

    plt.plot(sim.state_traj[:, 3:6])
    plt.legend(["x_L", "y_L", "z_L"])
    plt.grid()
    plt.show()

    plt.plot(sim.state_traj[:, 6:9])
    plt.legend(["x_dot", "y_dot", "z_dot"])
    plt.grid()
    plt.show()

    plt.plot(sim.state_traj[:, 9:12])
    plt.legend(["x_Ldot", "y_Ldot", "z_Ldot"])
    plt.grid()
    plt.show()

    plt.plot(sim.state_traj[:, 2] - sim.state_traj[:, 5])
    plt.legend(["diff_z"])
    plt.grid()
    plt.show()


    # Plot the input trajectory
    plt.plot(sim.input_traj)
    plt.legend(["F"])
    plt.show()
