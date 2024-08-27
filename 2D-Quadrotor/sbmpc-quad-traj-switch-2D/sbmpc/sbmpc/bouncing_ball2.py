import numpy as np
import matplotlib.pyplot as plt

# Parameters
g = 9.81  # acceleration due to gravity (m/s^2)
L = 5.0   # cable length (meters)
dt = 0.01 # time step (seconds)

# Initial conditions
r_d = np.array([0.0, 10.0]) # drone position (x_d, y_d)
v_d = np.array([1.0, 0.0])  # drone velocity (v_dx, v_dy)
r_b = np.array([0.0, 5.0])  # ball position (x_b, y_b)
v_b = np.array([0.0, 0.0])  # ball velocity (v_bx, v_by)

def equations_of_motion(t, state):
    r_d, v_d, r_b, v_b = state[:2], state[2:4], state[4:6], state[6:8]

    # Acceleration due to gravity on the ball
    a_b = np.array([0, -g])
    
    # No acceleration on the drone in this example (can be modified)
    a_d = np.array([0, 0])
    
    # Equations of motion
    dr_d_dt = v_d
    dv_d_dt = a_d
    dr_b_dt = v_b
    dv_b_dt = a_b

    return np.concatenate([dr_d_dt, dv_d_dt, dr_b_dt, dv_b_dt])

def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + 0.5*dt, y + 0.5*dt*k1)
    k3 = f(t + 0.5*dt, y + 0.5*dt*k2)
    k4 = f(t + dt, y + dt*k3)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def handle_collision(r_d, v_d, r_b, v_b, L):
    # Compute the distance between the drone and the ball
    dist = np.linalg.norm(r_d - r_b)
    
    if dist > L:
        # Inelastic collision: adjust velocities
        direction = (r_b - r_d) / dist
        relative_velocity = v_b - v_d
        print("v_b_before",v_b)
        # Reflect the ball velocity in the direction of the cable
        #v_rel_along_cable = np.dot(relative_velocity, direction)
        #v_b = v_b - 2 * v_rel_along_cable * direction

        v_rel_along_cable = np.dot(relative_velocity, direction) * direction
        
        # Adjust the ball's velocity by removing the component along the cable direction
        v_b = v_b +  v_rel_along_cable
        
        # Adjust drone's velocity to conserve momentum (if needed)
        v_d = v_d -  v_rel_along_cable

        print("v_b_after",v_b)
    return v_d, v_b

# Simulation loop
t = 0.0
state = np.concatenate([r_d, v_d, r_b, v_b])
trajectory_drone = []
trajectory_ball = []

for i in range(1000):
    # Store trajectory for plotting
    trajectory_drone.append(state[:2])
    trajectory_ball.append(state[4:6])

    # Perform a Runge-Kutta step
    state = rk4_step(equations_of_motion, t, state, dt)
    
    # Extract the updated state
    r_d, v_d, r_b, v_b = state[:2], state[2:4], state[4:6], state[6:8]
    
    # Check for collision and handle it
    v_d, v_b = handle_collision(r_d, v_d, r_b, v_b, L)
    # Update the state with the new velocities after collision
    state[2:4] = v_d
    state[6:8] = v_b
    
    t += dt

# Convert trajectories to numpy arrays for easier plotting
trajectory_drone = np.array(trajectory_drone)
trajectory_ball = np.array(trajectory_ball)

# Plotting the results
plt.figure(figsize=(10, 6))
#plt.plot(trajectory_drone[:, 0], trajectory_drone[:, 1], label='Drone')
#plt.plot(trajectory_ball[:, 0], trajectory_ball[:, 1], label='Ball (Payload)')

plt.plot(trajectory_drone[:, 0], trajectory_drone[:, 1], label='Drone')
plt.legend()
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Bouncing Ball Attached to a Drone')
plt.grid(True)
plt.show()


plt.plot(trajectory_ball[:, 1], label='Ball (Payload)')
plt.legend()
#plt.xlabel('X Position (m)')
#plt.ylabel('Y Position (m)')
plt.title('Bouncing Ball Attached to a Drone')
plt.grid(True)
plt.show()
