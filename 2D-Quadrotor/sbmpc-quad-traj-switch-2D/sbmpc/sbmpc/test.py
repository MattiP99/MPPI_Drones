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

csi = jnp.array([0,0,-1])
mass = 2.7
csi_dot = jnp.array([0,0,0.000001])
csi_ddot = jnp.cross(csi,jnp.cross(csi,jnp.array([0,0,44]))) / mass * cable_length - (csi_dot @ csi_dot) * csi
print("csi_ddot",csi_ddot)
lilli = (jnp.diag(jnp.array([1,1,1])) - ((csi).reshape(3,1) @ (csi).reshape(1,3)))
print("lilli",lilli)
