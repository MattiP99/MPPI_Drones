import jax
import jax.numpy as jnp
import numpy as np


mass = 2.7
mass_payload = 0.25
cable_length = 0.5
gravity = 9.8

inputs = (mass + mass_payload) * gravity
state = jnp.array([0, 0. , 5, 0.,0., 4.5, 0, 0.,1 ,0,0.,-0.5], dtype=jnp.float32)

csi = (state[3:6] - state[0:3])/cable_length
csi_dot = (state[9:12] - state[6:9])/cable_length
#/cable_length

v_rel_along_cable = np.dot(csi_dot, csi) * (csi)
print("v_rel_along_cable",v_rel_along_cable)
# Compute total mass
total_mass = mass + mass_payload
        
# Compute new velocities conserving momentum
v_d_new = state[6:9] + (2 * mass_payload / total_mass) * v_rel_along_cable
v_b_new = state[9:12] - (2 * mass / total_mass) * v_rel_along_cable
print("v_d_new",v_d_new)
print("v_b_new",v_b_new)


v_kp1_parallel_drone2 = (1/(mass + mass_payload))*((mass *  ((csi * jnp.transpose(csi)) @ state[6:9])) + mass_payload * ((csi * jnp.transpose(csi)) @ state[9:12])) 
#v_kp1_parallel_drone = (1/(mass + mass_payload))*((mass *  (jnp.dot(csi, csi_dot)) ) + mass_payload * ((jnp.dot(csi, csi_dot)))) 
#print("v_kp1_parallel_drone",v_kp1_parallel_drone)

#state = state.at[9:12].set(state[9:12] -  v_kp1_parallel_drone * csi)

v_kp1_parallel_drone2 = v_kp1_parallel_drone2 * (-csi/jnp.linalg.norm(csi)) 
print("v_kp1_parallel_drone2",v_kp1_parallel_drone2)
#print("v_kp1_parallel_drone",v_kp1_parallel_drone)

print("v_kp1_parallel_payload",state[9:12])

#
v_orthogoal_drone = jnp.cross(csi,state[6:9])
print("v_orthogoal_drone",v_orthogoal_drone)
           
# Normalize the vector
v_perpendicular_drone = jnp.cross(csi , v_orthogoal_drone)


print("v_perpendicular_drone", v_perpendicular_drone)

# "csi * jnp.transpose(csi) * state[6:9]
print("v new", 1/(mass+mass_payload)* ( mass* (csi @ jnp.transpose(csi)) * state[6:9] + mass_payload * (csi @ jnp.transpose(csi)) * state[9:12]))
print("csi * jnp.transpose(csi) * state[9:12]", csi * jnp.transpose(csi) * state[9:12])
print("csi @ jnp.transpose(csi) @ state[6:9]", mass* (csi @ jnp.transpose(csi)) * state[6:9])
print("csi @ jnp.transpose(csi) @ state[9:12]", mass_payload * (csi @ jnp.transpose(csi)) * state[9:12])


print("csi",csi)
print("csi_dot",csi_dot)
print("csi csiT",csi * jnp.transpose(csi))
print("csi csi",csi * csi)
print("csi @ csiT",csi @ jnp.transpose(csi))
print("csiT @ csi",jnp.transpose(csi) @ csi)
print("csiT @ csi new", csi.reshape((3,1)) @ csi.reshape((1,3)))


acc_L = (1/(mass+mass_payload)) * ((jnp.dot(csi, jnp.array([0.,0.,inputs])) - mass*cable_length*jnp.dot(csi_dot, csi_dot)) * csi) - jnp.array([0,0,gravity])
csi_ddot = (1/(mass*cable_length))*jnp.cross(csi,jnp.cross(csi,jnp.array([0.,0.,inputs])))  -  jnp.dot(csi_dot,csi_dot) * csi
acc = acc_L -  cable_length * csi_ddot
cable_direction_projmat = csi.reshape((3,1)) @ csi.reshape((1,3))
print("acc_L",acc_L)
print("csi_ddot",csi_ddot)
print("acc",acc)
print("cable_direction_projmat",cable_direction_projmat)

