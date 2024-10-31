import time, os

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt
import sys




state_traj = jnp.zeros((1000,3))
h = 3
y0=h
x0=0
vx0 = 0
vy0 = 0
c = 0.8
t = 0
dt = 0.01
bt =0
g = 9.8
i = 0
while t<10:
    x = x0 + vx0*bt
    y = y0 + vy0*bt - 0.5*g*bt**2
    vx = vx0
    vy = vy0 - g*bt
    position = np.array([x,y,0])
    #print("position",position)
    if (position[1]<= 0):
        vy = -np.sqrt(c)*vy
        vx = np.sqrt(c)*vx
        vy0 = vy
        vx0 = vx
        x0 = 0
        y0 = 0
        bt = 0
    state_traj =  state_traj.at[i,:].set(position)
    t = t+dt
    bt = bt+dt
    i = i+1    
    
plt.plot(state_traj[:, 1])
plt.legend(["y_L"])
plt.grid()
plt.show()


# That works properly
# What to change in the code of the drone + payload system?
#   - Having a bouncing time ?
#   - Changing the initial velocity of the payload after the bounce
#   - Having a factor that represents the energy lost after the bounce?