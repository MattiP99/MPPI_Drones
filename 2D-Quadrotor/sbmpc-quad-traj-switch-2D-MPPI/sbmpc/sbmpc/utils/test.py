import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

mass= 2.5
mass_payload = 0.25
gravity = 9.81
cable_length = 0.5
state = jnp.array([0,0,0,0,0,0,0,0,0,0,0,0])
csi = (state[3:6] - state[0:3])/cable_length
csi_dot = (state[9:12] - state[6:9])/cable_length
csi_omega = jnp.cross(csi, csi_dot)
inputs = jnp.array([0,0,0])
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
