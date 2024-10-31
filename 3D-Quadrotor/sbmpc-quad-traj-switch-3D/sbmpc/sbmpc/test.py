import time, os

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt
import sys

cable_length = 0.5
state = jnp.array([1., 0., 0., 0., 0., 0., 0., 0., 0. , 0., 0., 0.], dtype=jnp.float32)
csi = 1/cable_length * (state[0:2]-state[6:8])
csi_dot = (state[0:2]-state[6:8])/cable_length
ciao = csi @ jnp.transpose(csi) * state[0:2]
print("ciao",ciao)
print("csi",csi)

input_max = jnp.array([1, 2.5, 2.5, 2])
input_min = jnp.array([0, -2.5, -2.5, -2])

mass = 0.027
mass_payload = 0.0025

gravity = 9.81
inertia = jnp.array([2.3951e-5, 2.3951e-5, 3.2347e-5], dtype=jnp.float32)
inertia_mat = jnp.diag(inertia)

spatial_inertia_mat = jnp.diag(jnp.concatenate([mass*jnp.ones(3, dtype=jnp.float32), inertia]))
print("spatial_inertia_mat",spatial_inertia_mat)


x = jnp.arange(0, 5, 0.1) 
input_sequence = jnp.sin(x)

for i in range(50):
    total_force = jnp.array([0., input_sequence[i]]) - mass*jnp.array([0., gravity],dtype=jnp.float32)
    acc_L = ((jnp.dot(csi, total_force) - mass*cable_length*jnp.dot(csi_dot,csi_dot) * csi)/(mass+mass_payload) - jnp.array([0,gravity]))
print("acc_L",acc_L)
total_force = jnp.array([0., 1]) - mass*jnp.array([0., gravity],dtype=jnp.float32)
print("total_force", total_force)

csi_ddot = csi * total_force

#/(mass*cable_length)  -  jnp.dot(csi_dot,jnp.transpose(csi_dot)) * csi
print("csi_ddot", csi_ddot)

resultat = jnp.cross(jnp.array([0,0,2]), jnp.array([0,0,3]))
print("resultat",resultat)
result2 = jnp.cross(jnp.array([0,0,1]), resultat)
print("result2",result2)



def func_taut(self,state,inputs,dt:float,csi):
        # the whole dynamic has to change, also the angular velocity
            v_kp1 = state[self.nq:] + dt * self.dynamics(state, inputs)[self.nq:] 

            # But in this case the velocity of the payload and the drone change differently
            v_kp1_parallel_drone = (mass * csi @ jnp.transpose(csi) * v_kp1[0:3] + mass_payload * csi @ jnp.transpose(csi) * v_kp1[3:6])/ (mass + mass_payload) 
            v_kp1_parallel_payload =  v_kp1_parallel_drone

            # TASK:  PROJECT VECTOR V_KP1_PARALLEL_DRONE ON VELOCITY DRONE OR I USE THE INVERSE OF THE EQUATION IN THE PAPER

            ############# PERPENDICULAR DIRECTION TO THE CABLE AND ROBOT/PAYLOAD VELOCITY ##################
            v_orthogoal_drone = jnp.cross(csi,state[6:9])
            # Normalize the vector
            v_orthogoal_drone = v_orthogoal_drone/jnp.linalg.norm(v_orthogoal_drone)
            v_perpendicular_drone = jnp.cross(v_orthogoal_drone,csi)

            v_orthogoal_payload = jnp.cross(csi,state[9:12])
            # Normalize the vector
            v_orthogoal_payload = v_orthogoal_payload/jnp.linalg.norm(v_orthogoal_payload)
            v_perpendicular_payload = jnp.cross(v_orthogoal_payload,csi)


            # finding norm of the vector v 
            v_norm = jnp.sqrt(sum(state[self.nq:self.nq+3]**2))     
            v_norm_payload = jnp.sqrt(sum(state[self.nq+3:self.nq+6]**2)) 
            # Apply the formula as mentioned above 
            # for projecting a vector onto another vector 
            # find dot product using np.dot() 
            
            proj_of_v_on_v_orth = (jnp.dot(state[self.nq:self.nq+3], v_perpendicular_drone)/v_norm**2)*state[self.nq:self.nq+3]
            proj_of_v_payload_on_v_payload_orth = (jnp.dot(state[self.nq+3:self.nq+6], v_perpendicular_payload)/v_norm_payload **2)*state[self.nq+3:self.nq+6]
            
            v = proj_of_v_on_v_orth + v_kp1_parallel_drone
            v_payload = proj_of_v_payload_on_v_payload_orth + v_kp1_parallel_payload 
            # Not sure about it  because the collision direction could have components along x,y of the drone if the drone an dthe payload are not vertically aligned#
            v_kp1 = v_kp1 + v_kp1.at[0:3].set(v) 

            v_kp1 = v_kp1 + v_kp1.at[3:6].set(v_payload) 

            return jnp.concatenate([
                    state[:self.nq] + dt * self.dynamics(jnp.concatenate([state[:self.nq], v_kp1]), inputs)[:self.nq],
                    v_kp1])


