import numpy as np
import matplotlib.pyplot as plt

# Example of a poly5 function to generate coefficients of a 5th-order polynomial
def poly5(t0, tf, q0, qf, dq0, dqf, ddq0, ddqf):
    # Set up the system of equations to solve for the coefficients
    A = np.array([[1, t0, t0**2, t0**3, t0**4, t0**5],
                  [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
                  [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],
                  [1, tf, tf**2, tf**3, tf**4, tf**5],
                  [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
                  [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]])
    
    b = np.array([q0, dq0, ddq0, qf, dqf, ddqf])
    
    # Solve the linear system for the coefficients
    coeffs = np.linalg.solve(A, b)
    return coeffs

# Parameters for trajectory
t0 = 0
t1 = 10  # End time
traj_length = 5  # Length of trajectory (final position)
dq0 = 0  # Initial velocity
dqf = 0  # Final velocity
ddq0 = 0  # Initial acceleration
ddqf = 0  # Final acceleration

# Generate time vector
time_vec = np.linspace(t0, t1, 100)  # 100 time steps

# Get the polynomial coefficients
a0, a1, a2, a3, a4, a5 = poly5(t0, t1, q0=0, qf=traj_length, dq0=dq0, dqf=dqf, ddq0=ddq0, ddqf=ddqf)

# Calculate position s(t) and velocity ds(t)
s_t = a0 + a1*time_vec + a2*time_vec**2 + a3*time_vec**3 + a4*time_vec**4 + a5*time_vec**5  # s(t)
ds_t = a1 + 2*a2*time_vec + 3*a3*time_vec**2 + 4*a4*time_vec**3 + 5*a5*time_vec**4  # ds(t)

# Plot the results
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(time_vec, s_t, label="Position (s_t)")
plt.title('5th-order Polynomial Trajectory')
plt.ylabel('Position (s)')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time_vec, ds_t, label="Velocity (ds_t)", color='red')
plt.ylabel('Velocity (ds/dt)')
plt.xlabel('Time (s)')
plt.grid()

plt.show()

