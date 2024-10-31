import jax.numpy as jnp
import sys
sys.path.append('/home/mpiras/MPPI/sbmpc-quad-traj-switch-2D-MPPI-TEST/sbmpc')
import sbmpc.utils.geometry as geom
from scipy.signal import chirp

initial_pos = jnp.array([0,0,5])
final_pos = jnp.array([0,4,9])
num_points = 526
dt = 30
mass = 2.7
mass_payload = 0.25
gravity = 9.81

# Function to compute the position along the trajectory
def poly5_trajectory(coeffs, t):
    # Evaluate the polynomial for a given t
    t_vec = jnp.array([t**5, t**4, t**3, t**2, t, 1])
    return jnp.dot(coeffs, t_vec)

# Function to compute the velocity along the trajectory
def poly5_velocity(coeffs, t):
    # Evaluate the derivative of the polynomial for a given t
    t_vec = jnp.array([5*t**4, 4*t**3, 3*t**2, 2*t, 1, 0])
    return jnp.dot(coeffs, t_vec)

# Function to solve for polynomial coefficients
def solve_poly5(start, end):
    # Boundary conditions
    # t = 0 and t = 1
    A = jnp.array([
        [0, 0, 0, 0, 0, 1],  # P(0) = P0
        [1, 1, 1, 1, 1, 1],  # P(1) = P1
        [0, 0, 0, 0, 1, 0],  # P'(0) = 0
        [5, 4, 3, 2, 1, 0],  # P'(1) = 0
        [0, 0, 0, 2, 0, 0],  # P''(0) = 0
        [20, 12, 6, 2, 0, 0] # P''(1) = 0
    ])

    B = jnp.array([start, end, 0, 0, 0, 0])  # Conditions vector

    # Solve for coefficients
    coeffs = jnp.linalg.solve(A, B)
    return coeffs


def poly5(t0,tf,q0,qf,dq0,dqf,ddq0,ddqf):
        '''Compute coefficents for fifth order polynomial'''
        T = tf - t0
        a0 = q0
        a1 = dq0
        a2 = 0.5*ddq0
        a3 = (1/(2*T**3))*(20*(qf-q0)-(8*dqf+12*dq0)*T+(ddqf-3*ddq0)*T**2)
        a4 = (1/(2*T**4))*(30*(q0-qf)+(14*dqf+16*dq0)*T-(2*ddqf-3*ddq0)*T**2)
        a5 = (1/(2*T**5))*(12*(qf-q0)-6*(dqf+dq0)*T+(ddqf-ddq0)*T**2)
        return a0,a1,a2,a3,a4,a5

# Define a function for time scaling (non-linear time scaling)
def time_scaling(t):
    # Cubic time scaling to increase velocity in the middle
    return t**3

#Linear Trajectory
def compute_linear_trajectory():
        reference = jnp.zeros((526, 16),dtype=jnp.float32)
        traj_length = jnp.linalg.norm(final_pos - initial_pos)
        time_vec = jnp.linspace(0, 1, num=num_points, dtype=jnp.float32) # time vector
        a0,a1,a2,a3,a4,a5 = poly5(t0=0, tf=1, q0=initial_pos, qf=final_pos, dq0=jnp.array([0,0,0]), dqf=jnp.array([0,0,0]), ddq0=jnp.array([0,0,0]), ddqf=jnp.array([0,0,0]))
        # s: path variable
        # Solve for the coefficients for each coordinate
        
        coeffs_x = solve_poly5(initial_pos[0], final_pos[0])
        coeffs_y = solve_poly5(initial_pos[1], final_pos[1])
        coeffs_z = solve_poly5(initial_pos[2], final_pos[2])
        
        # Generate a trajectory
        t_values = jnp.linspace(0, 1, num_points)
        i = 0
        for time in t_values:
            t= time_scaling(time)
            reference = reference.at[i,:].set([poly5_trajectory(coeffs_x, t),
                        poly5_trajectory(coeffs_y, t),
                        poly5_trajectory(coeffs_z, t),
                        poly5_trajectory(coeffs_x, t),
                        poly5_trajectory(coeffs_y, t),
                        poly5_trajectory(coeffs_z, t) - 0.5,
                        0,
                        poly5_velocity(coeffs_x, t),
                        poly5_velocity(coeffs_y, t),
                        poly5_velocity(coeffs_z, t),
                        poly5_velocity(coeffs_x, t),
                        poly5_velocity(coeffs_y, t),
                        poly5_velocity(coeffs_z, t),
                        0,
                        (mass+mass_payload)*gravity/2,
                        (mass+mass_payload)*gravity/2
                        ])
            i+=1
        
        """
        for i in range(num_points):
            # Drone position reference
            s_t = a0 + a1*time_vec[i] + a2*time_vec[i]**2 + a3*time_vec[i]**3 + a4*time_vec[i]**4 + a5*time_vec[i]**5 # s(t)
            ds_t = a1 + 2*a2*time_vec[i] + 3*a3*time_vec[i]**2 + 4*a4*time_vec[i]**3 + 5*a5*time_vec[i]**4 # ds(t)
            # x
            x_t = final_pos[0]/traj_length * s_t[0]  # x(t)
            dx_t =  ds_t[0] # dx(t)
            # y
            y_t = final_pos[1]/traj_length * s_t[1]  # y(t)
            dy_t =  ds_t[1] # dy(t)

            z_t = final_pos[2]/traj_length * s_t[2] # y(t)
            dz_t = ds_t[2] # dy(t)
            reference = reference.at[i,0].set(x_t)
            reference = reference.at[i,1].set(y_t)
            reference = reference.at[i,2].set(z_t)
            # Payload position reference
            reference = reference.at[i,3].set(x_t)
            reference = reference.at[i,4].set(y_t)
            reference = reference.at[i,5].set(z_t-0.5)
            # Drone velocity reference
            
            reference = reference.at[i,7].set(dx_t)
            reference = reference.at[i,8].set(dy_t)
            reference = reference.at[i,9].set(dz_t)
            reference = reference.at[i,10].set(dx_t)
            reference = reference.at[i,11].set(dy_t)
            reference = reference.at[i,12].set(dz_t)#

        # Input reference
            reference = reference.at[i,-2:].set([(mass+mass_payload)*gravity/2 , (mass+mass_payload)*gravity/2]) 
        """
        return reference

reference = compute_linear_trajectory()
print(reference[260:270,7:10])


"""
# Define start and end points
P0 = jnp.array([0, 0, 5])  # Start point in 3D
P1 = jnp.array([0, 4, 9])  # End point in 3D

# Solve for the coefficients for each coordinate
coeffs_x = solve_poly5(P0[0], P1[0])
coeffs_y = solve_poly5(P0[1], P1[1])
coeffs_z = solve_poly5(P0[2], P1[2])

# Generate a trajectory
t_values = jnp.linspace(0, 1, num_points)
trajectory = jnp.array([[poly5_trajectory(coeffs_x, t),
                        poly5_trajectory(coeffs_y, t),
                        poly5_trajectory(coeffs_z, t),
                        poly5_trajectory(coeffs_x, t),
                        poly5_trajectory(coeffs_y, t),
                        poly5_trajectory(coeffs_z, t) - 0.5,
                        0,
                        poly5_velocity(coeffs_x, t),
                        poly5_velocity(coeffs_y, t),
                        poly5_velocity(coeffs_z, t),
                        poly5_velocity(coeffs_x, t),
                        poly5_velocity(coeffs_y, t),
                        poly5_velocity(coeffs_z, t),
                        0
                        ] for t in t_values])

# Print trajectory
print(trajectory)
"""

