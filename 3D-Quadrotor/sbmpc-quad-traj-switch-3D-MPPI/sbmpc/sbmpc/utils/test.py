import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/mpiras/MPPI/sbmpc-quad-traj-switch-3D-MPPI/sbmpc')

from sbmpc.utils.geometry import quat2rotm, quat_product, quat_inverse, euler_to_quaternion, rotation_matrix_around_x ,skew


mass= 2.5
mass_payload = 0.25
gravity = 9.81
cable_length = 0.5
state = jnp.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
csi = (state[3:6] - state[0:3])/cable_length
csi_dot = (state[9:12] - state[6:9])/cable_length
csi_omega = jnp.cross(csi, csi_dot)
inputs = jnp.array([0,0,0,0])

inertia = jnp.array([2.3951e-3, 2.3951e-3, 3.2347e-3], dtype=jnp.float32)
inertia_mat = jnp.diag(inertia)

spatial_inertia_mat = jnp.diag(jnp.concatenate([mass*jnp.ones(3, dtype=jnp.float32), inertia]))
spatial_inertia_mat_inv = jnp.linalg.inv(spatial_inertia_mat)

F = jnp.array([0,0,mass_payload*gravity]) + jnp.array([0,0,inputs[0]])
quad_force_vector = F 
quad_centrifugal_f = mass * cable_length * (csi_omega @ csi_omega)
tension_vector = mass_payload * (-jnp.transpose(csi).reshape(1,3) @ quad_force_vector + quad_centrifugal_f) * csi.reshape(3,1) / (mass+mass_payload)
# Solving for Load Acceleration
acc_L = - jnp.transpose(tension_vector) / mass_payload - jnp.array([0.,0.,gravity])
# Solving for Quadrotor Acceleration
acc = (quad_force_vector + jnp.transpose(tension_vector)) / mass - jnp.array([0.,0.,gravity])

print("acc_L",acc_L)
print("acc",acc)
print("tension_vector",tension_vector)
print("quad_centrifugal_f",quad_centrifugal_f)
print("F",F)
print("csi_omega",csi_omega)

quat = jnp.array([1,0,0,0])
orientation_mat = quat2rotm(quat)

#quad_force_vector = F * quat2rotm(quat) @ e3  
total_force = jnp.array([0., 0., inputs[0]])# - mass*gravity*orientation_mat[2, :]  # transpose + 3rd col = 3rd row
total_force_world = orientation_mat @ total_force

print("total_force_world",total_force_world)

csi_omega = jnp.cross(csi, csi_dot)
    

quat = state[6:10]
ang_vel = state[16:19]
ang_vel_quat = jnp.array([0., state[16], state[17], state[18]])
print("quad_force_vector taut",total_force_world)
quad_centrifugal_f = mass * cable_length * (csi_omega @ csi_omega)

print("quad_centrifugal_f taut",quad_centrifugal_f)

tension_vector = mass_payload * (-csi.reshape(1,3) @ total_force_world + quad_centrifugal_f) * csi.reshape(3,1) / (mass+mass_payload)
print("tension_vectortaut",tension_vector)
# Solving for Load Acceleration
    
    
acc_L =  -tension_vector.reshape(3,) / mass_payload - jnp.array([0.,0.,gravity])
print("tension_vector taut TRANSPOST",tension_vector)
print("acc_L taut",acc_L)
# Solving for Quadrotor Acceleration
acc = (total_force_world + tension_vector.reshape(3,)) / mass - jnp.array([0.,0.,gravity])
print("acc taut",acc)

total_torque = inputs[1:4] - skew(ang_vel) @ inertia_mat @ ang_vel
#acc = quad_force_vector/mass - jnp.array([0.,0.,gravity])
acc_rot = spatial_inertia_mat_inv[3:6,3:6] @ total_torque
print("total_torque taut",total_torque)
print("acc_rot taut",acc_rot)

cable_direction_projmat = csi.reshape((3,1)) @ csi.reshape((1,3))
vDrone_proj = cable_direction_projmat @ state[10:13]
vPayload_proj = cable_direction_projmat @ state[13:16]

v_kp1_parallel_drone = (mass * vDrone_proj + mass_payload * vPayload_proj)/(mass_payload + mass)
print("v_kp1_parallel_drone taut",v_kp1_parallel_drone)

ciao = orientation_mat @ acc[:3]
print("ciao taut",ciao)

state_dot = jnp.concatenate([state[10:13],
                            #0.5 * quat_product(quat, ang_vel_quat),
                            state[13:16],
                            0.5 * quat_product(quat, ang_vel_quat),
                            #orientation_mat @ acc[:3],
                            orientation_mat @ acc[:3],
                            acc_L,
                            acc_rot])
print("state_dot taut",state_dot)

# Solving for Quadrotor Acceleration
total_torque_slack = inputs[1:4] - skew(ang_vel) @ inertia_mat @ ang_vel
#acc = quad_force_vector/mass - jnp.array([0.,0.,gravity])
acc_slack = spatial_inertia_mat_inv @ jnp.concatenate([total_force, total_torque_slack]) 
state_dot_slack =  jnp.concatenate([state[10:13],
                            #0.5 * quat_product(quat, ang_vel_quat),
                            state[13:16],
                            0.5 * quat_product(quat, ang_vel_quat),
                            #orientation_mat @ acc[:3],
                            orientation_mat @ acc_slack[:3],
                            acc_L,
                            acc_slack[3:6]])

print("state_dot slack",state_dot_slack)
ang_err = quat_product(quat_inverse(state[6:10]), state[6:10])[1:4]
print("ang_err ",ang_err)