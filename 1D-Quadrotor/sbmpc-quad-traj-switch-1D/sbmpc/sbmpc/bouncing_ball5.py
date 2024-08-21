import numpy as np
import matplotlib.pyplot as plt

# Parameters
g = 9.81  # acceleration due to gravity (m/s^2)
L = 1.0   # cable length (meters)
dt = 0.01 # time step (seconds)
m_d = 2.5 # mass of the drone (kg)
m_b = 0.1 # mass of the ball (kg)
amplitude = 0.5  # amplitude of the vertical oscillation
frequency = 0.5  # frequency of the vertical oscillation

# Initial conditions
r_d = np.array([0.0, 10.0]) # drone position (x_d, y_d)
v_d = np.array([1.0, 0.0])  # drone velocity (v_dx, v_dy)
r_b = np.array([0.0, 9.0])  # ball position (x_b, y_b)
v_b = np.array([0.0, 0.0])  # ball velocity (v_bx, v_by)

# Input force from the drone's rotors to hover and fly horizontally with vertical oscillation
def input_force(t):
    F_vertical = m_d * g + amplitude * np.sin(2 * np.pi * frequency * t)  # Vertical force with oscillation
    F_horizontal = 1.0    # Constant horizontal force
    return np.array([F_horizontal, 1.05*F_vertical])


def equations_of_motion(t, state):
    r_d, v_d, r_b, v_b = state[:2], state[2:4], state[4:6], state[6:8]

    # Compute the distance between the drone and the ball
    dist = np.linalg.norm(r_d - r_b)
    
    # Get the force from the drone's rotors
    F_d = input_force(t)
    
    if dist > L:
        # Cable is taut
        direction = (r_b - r_d) / dist
        T = np.dot(F_d, direction)
        a_d = (F_d - T * direction) / m_d #+ np.array([0, -g])
        a_b = (T * direction - np.array([0, m_b * g])) / m_b
    else:
        # Cable is slack
        a_d = F_d / m_d #+ np.array([0, -g]) # Only external force acts on the drone
        a_b = np.array([0, -g])  # Only gravity acts on the ball
    
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

def handle_collision(r_d, v_d, r_b, v_b, L, m_d, m_b):
    # Compute the distance between the drone and the ball
    dist = np.linalg.norm(r_d - r_b)
    
    if dist > L:
        # Normalize the direction vector
        direction = (r_b - r_d) / dist
        relative_velocity = v_b - v_d
        
        # Compute the component of the relative velocity along the cable direction
        v_rel_along_cable = np.dot(relative_velocity, direction) * direction
        
        # Compute total mass
        total_mass = m_d + m_b
        
        # Compute new velocities conserving momentum
        v_d_new = v_d + (2 * m_b / total_mass) * v_rel_along_cable
        v_b_new = v_b - (2 * m_d / total_mass) * v_rel_along_cable
        
        # Adjust the position to ensure the ball is exactly at the cable length
        r_b = r_d + L * direction
        
        return v_d_new, v_b_new, r_b
    
    return v_d, v_b, r_b

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
    v_d, v_b, r_b = handle_collision(r_d, v_d, r_b, v_b, L, m_d, m_b)
    
    # Update the state with the new velocities and positions after collision
    state[2:4] = v_d
    state[4:6] = r_b
    state[6:8] = v_b
    
    t += dt

# Convert trajectories to numpy arrays for easier plotting
trajectory_drone = np.array(trajectory_drone)
trajectory_ball = np.array(trajectory_ball)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(trajectory_drone[:, 0], trajectory_drone[:, 1], label='Drone')
plt.plot(trajectory_ball[:, 0], trajectory_ball[:, 1], label='Ball (Payload)')
plt.legend()
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Bouncing Ball Attached to a Drone with Vertical Oscillation')
plt.grid(True)
plt.show()

plt.plot( trajectory_ball[:, 0], label='Ball (Payload)')
plt.legend()
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Bouncing Ball Attached to a Drone with Vertical Oscillation')
plt.grid(True)
plt.show()

