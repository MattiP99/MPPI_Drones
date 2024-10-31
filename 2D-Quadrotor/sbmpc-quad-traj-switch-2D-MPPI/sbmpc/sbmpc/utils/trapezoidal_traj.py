import jax.numpy as jnp
import sys
sys.path.append('/home/mpiras/MPPI/sbmpcquad-traj-switch-2D-MPPI-TEST/sbmpc')
import sbmpc.utils.geometry as geom
from scipy.signal import chirp

mass = 2.7
mass_payload = 0.25
cable_length = 0.5

gravity = 9.81
dimension = 16

# Function to solve for polynomial coefficients
def solve_poly5(start, end):
    # Boundary conditions
    A = jnp.array([
        [0, 0, 0, 0, 0, 1],  # P(0) = P0
        [1, 1, 1, 1, 1, 1],  # P(1) = P1
        [0, 0, 0, 0, 1, 0],  # P'(0) = V0
        [5, 4, 3, 2, 1, 0],  # P'(1) = V1
        [0, 0, 0, 2, 0, 0],  # P''(0) = 0
        [20, 12, 6, 2, 0, 0] # P''(1) = 0
    ])

    B = jnp.array([start, end, 0, 0, 0, 0])  # Positions and velocities

    # Manually calculate the coefficients using matrix inversion
    A_inv = jnp.linalg.inv(A)  # Inverse of the matrix A
    coeffs = jnp.dot(A_inv, B)  # Multiply inverse matrix by boundary conditions vector B

    return coeffs

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

# Define a function for time scaling (non-linear time scaling)
def time_scaling(t):
    # Cubic time scaling to increase velocity in the middle
    return t**3

class Trapeizoidal_Trajectory:
    def __init__(self, initial_pos, final_pos,dt, num_points):
        self.initial_pos = initial_pos
        self.final_pos = final_pos
        self.num_points = num_points
        self.dt=dt
        self.reference = jnp.zeros((self.num_points, dimension ),dtype=jnp.float32)


    
    def poly5(self,t0,tf,q0,qf,dq0,dqf,ddq0,ddqf):
        '''Compute coefficents for fifth order polynomial'''
        T = tf - t0
        a0 = q0
        a1 = dq0
        a2 = 0.5*ddq0
        a3 = (1/(2*T**3))*(20*(qf-q0)-(8*dqf+12*dq0)*T+(ddqf-3*ddq0)*T**2)
        a4 = (1/(2*T**4))*(30*(q0-qf)+(14*dqf+16*dq0)*T-(2*ddqf-3*ddq0)*T**2)
        a5 = (1/(2*T**5))*(12*(qf-q0)-6*(dqf+dq0)*T+(ddqf-ddq0)*T**2)
        return a0,a1,a2,a3,a4,a5
    
    

    



    #Linear Trajectory
    def compute_linear_trajectory2(self):
        
        #traj_length = jnp.linalg.norm(final_pos - initial_pos)
        #time_vec = jnp.linspace(0, dt, num=num_points, dtype=jnp.float32) # time vector
        #a0,a1,a2,a3,a4,a5 = poly5(t0=0, tf=dt, q0=initial_pos, qf=final_pos, dq0=jnp.array([0,0,0]), dqf=jnp.array([0,0,0]), ddq0=jnp.array([0,0,0]), ddqf=jnp.array([0,0,0]))
        # s: path variable
        # Solve for the coefficients for each coordinate
        coeffs_x = solve_poly5(self.initial_pos[0], self.final_pos[0])
        coeffs_y = solve_poly5(self.initial_pos[1], self.final_pos[1])
        coeffs_z = solve_poly5(self.initial_pos[2], self.final_pos[2])
        
        # Generate a trajectory
        t_values = jnp.linspace(0, 1, self.num_points)
        i = 0
        for time in t_values:
            t= time_scaling(time)
            self.reference = self.reference.at[i,:].set([poly5_trajectory(coeffs_x, t),
                        poly5_trajectory(coeffs_y, t),
                        poly5_trajectory(coeffs_z, t),
                        poly5_trajectory(coeffs_x, t),
                        poly5_trajectory(coeffs_y, t),
                        poly5_trajectory(coeffs_z, t),
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
        return self.reference
    


    #Linear Trajectory
    def compute_linear_trajectory3(self):
        
        traj_length = jnp.linalg.norm(self.final_pos)
        time_vec = jnp.linspace(0, 1, num=self.num_points, dtype=jnp.float32) # time vector
        a0x,a1x,a2x,a3x,a4x,a5x = self.poly5(t0=0, tf=1, q0=self.initial_pos[0], qf=self.final_pos[0], dq0=0, dqf=0, ddq0=0, ddqf=0)
        a0y,a1y,a2y,a3y,a4y,a5y = self.poly5(t0=0, tf=1, q0=self.initial_pos[1], qf=self.final_pos[1], dq0=0, dqf=0, ddq0=0, ddqf=0)
        a0z,a1z,a2z,a3z,a4z,a5z = self.poly5(t0=0, tf=1, q0=self.initial_pos[2], qf=self.final_pos[2], dq0=0, dqf=0, ddq0=0, ddqf=0)
        time_in_seconds = time_vec * self.dt

        # s: path variable
        s_tx = a0x + a1x*time_in_seconds/self.dt + a2x*(time_in_seconds/self.dt)**2 + a3x*(time_in_seconds/self.dt)**3 + a4x*(time_in_seconds/self.dt)**4 + a5x*(time_in_seconds/self.dt)**5 # s(t)
        dsx_t = a1x + 2*a2x*(time_in_seconds/self.dt) + 3*a3x*(time_in_seconds/self.dt)**2 + 4*a4x*(time_in_seconds/self.dt)**3 + 5*a5x*(time_in_seconds/self.dt)**4 # ds(t)

        s_ty = a0y + a1y*(time_in_seconds/self.dt) + a2y*(time_in_seconds/self.dt)**2 + a3y*(time_in_seconds/self.dt)**3 + a4y*(time_in_seconds/self.dt)**4 + a5y*(time_in_seconds/self.dt)**5 # s(t)
        dsy_t = a1y + 2*a2y*(time_in_seconds/self.dt) + 3*a3y*(time_in_seconds/self.dt)**2 + 4*a4y*(time_in_seconds/self.dt)**3 + 5*a5y*(time_in_seconds/self.dt)**4 # ds(t)

        s_tz = a0z + a1z*(time_in_seconds/self.dt) + a2z*(time_in_seconds/self.dt)**2 + a3z*(time_in_seconds/self.dt)**3 + a4z*(time_in_seconds/self.dt)**4 + a5z*(time_in_seconds/self.dt)**5 # s(t)
        dsz_t = a1z + 2*a2z*(time_in_seconds/self.dt) + 3*a3z*(time_in_seconds/self.dt)**2 + 4*a4z*(time_in_seconds/self.dt)**3 + 5*a5z*(time_in_seconds/self.dt)**4 # ds(t)

        """
        # s: path variable
        s_tx = a0x + a1x*time_vec + a2x*time_vec**2 + a3x*time_vec**3 + a4x*time_vec**4 + a5x*time_vec**5 # s(t)
        dsx_t = a1x + 2*a2x*time_vec + 3*a3x*time_vec**2 + 4*a4x*time_vec**3 + 5*a5x*time_vec**4 # ds(t)

        s_ty = a0y + a1y*time_vec + a2y*time_vec**2 + a3y*time_vec**3 + a4y*time_vec**4 + a5y*time_vec**5 # s(t)
        dsy_t = a1y + 2*a2y*time_vec + 3*a3y*time_vec**2 + 4*a4y*time_vec**3 + 5*a5y*time_vec**4 # ds(t)

        s_tz = a0z + a1z*time_vec + a2z*time_vec**2 + a3z*time_vec**3 + a4z*time_vec**4 + a5z*time_vec**5 # s(t)
        dsz_t = a1z + 2*a2z*time_vec + 3*a3z*time_vec**2 + 4*a4z*time_vec**3 + 5*a5z*time_vec**4 # ds(t)
        """
        # x
        x_t = s_tx # x(t)
        dx_t = dsx_t/self.dt # dx(t)
        # y
        y_t = s_ty # y(t)
        dy_t = dsy_t/self.dt # dy(t)

        z_t = s_tz # y(t)
        dz_t = dsz_t/self.dt # dy(t)
        
        for i in range(self.num_points):
            


            # Drone position reference
            self.reference = self.reference.at[i,0].set(x_t[i])
            self.reference = self.reference.at[i,1].set(y_t[i])
            self.reference = self.reference.at[i,2].set(z_t[i])

            # Payload position reference
            self.reference = self.reference.at[i,3].set(x_t[i])
            self.reference = self.reference.at[i,4].set(y_t[i])
            self.reference = self.reference.at[i,5].set(z_t[i])

            # Drone velocity reference
            
            self.reference = self.reference.at[i,7].set(dx_t[i])
            self.reference = self.reference.at[i,8].set(dy_t[i])
            self.reference = self.reference.at[i,9].set(dz_t[i])

            self.reference = self.reference.at[i,10].set(dx_t[i])
            self.reference = self.reference.at[i,11].set(dy_t[i])
            self.reference = self.reference.at[i,12].set(dz_t[i])


            # Input reference
            self.reference = self.reference.at[i,-2:].set([(mass+mass_payload)*gravity/2 , (mass+mass_payload)*gravity/2]) 
        
        return self.reference
    

    #Linear Trajectory
    def compute_linear_trajectory(self):
        
        traj_length = jnp.linalg.norm(self.final_pos)
        time_vec = jnp.linspace(0, self.dt, num=self.num_points, dtype=jnp.float32) # time vector
        a0,a1,a2,a3,a4,a5 = self.poly5(t0=0, tf=1, q0=self.initial_pos, qf=self.final_pos, dq0=0, dqf=0, ddq0=0, ddqf=0)
        
        # s: path variable
        s_t = a0 + a1*time_vec + a2*time_vec**2 + a3*time_vec**3 + a4*time_vec**4 + a5*time_vec**5 # s(t)
        ds_t = a1 + 2*a2*time_vec + 3*a3*time_vec**2 + 4*a4*time_vec**3 + 5*a5*time_vec**4 # ds(t)
        # x
        x_t = self.initial_pos[0] + self.final_pos[0]/traj_length * s_t # x(t)
        dx_t = self.final_pos[0]/traj_length * ds_t # dx(t)
        # y
        y_t = self.initial_pos[1] + self.final_pos[1]/traj_length * s_t # y(t)
        dy_t = self.final_pos[1]/traj_length * ds_t # dy(t)

        z_t = self.initial_pos[2] + self.final_pos[2]/traj_length * s_t # y(t)
        dz_t = self.final_pos[2]/traj_length * ds_t # dy(t)

        for i in range(self.num_points):
            # Drone position reference
            self.reference = self.reference.at[i,0].set(x_t[i])
            self.reference = self.reference.at[i,1].set(y_t[i])
            self.reference = self.reference.at[i,2].set(z_t[i])

            # Payload position reference
            self.reference = self.reference.at[i,3].set(x_t[i])
            self.reference = self.reference.at[i,4].set(y_t[i])
            self.reference = self.reference.at[i,5].set(z_t[i]-0.5)

            # Drone velocity reference
            
            self.reference = self.reference.at[i,7].set(dx_t[i])
            self.reference = self.reference.at[i,8].set(dy_t[i])
            self.reference = self.reference.at[i,9].set(dz_t[i])

            self.reference = self.reference.at[i,10].set(dx_t[i])
            self.reference = self.reference.at[i,11].set(dy_t[i])
            self.reference = self.reference.at[i,12].set(dz_t[i])


            # Input reference
            self.reference = self.reference.at[i,-2:].set([(mass+mass_payload)*gravity/2 , (mass+mass_payload)*gravity/2]) 
        
        return self.reference
    
    def compute_sinusoidal_trajectory(self):
        traj_length = jnp.linalg.norm(self.final_pos)
        #time_vec = jnp.linspace(0, self.dt, num=self.num_points, dtype=jnp.float32) # time vector
        time_vec = jnp.arange(0, self.dt, self.dt/self.num_points)
        A = 100.0      # Amplitude (meters)
        f = 0.25      # Frequency (Hz)
        phi = 0.0    # Phase shift (radians)
        
        
        z_t =  A * jnp.sin(2 * jnp.pi * f * time_vec + phi)
        
        dz_t = A * 2 * jnp.pi * f * jnp.cos(2 * jnp.pi * f * time_vec + phi)
        for i in range(self.num_points):
            self.reference = self.reference.at[i,2].set(z_t[i])

            # Payload reference
            #self.reference = self.reference.at[i,3].set(?)
            #self.reference = self.reference.at[i,4].set(?)
            #self.reference = self.reference.at[i,5].set(?)
            self.reference = self.reference.at[i,8].set(dz_t[i])

            self.reference = self.reference.at[i,-1].set((mass+mass_payload)*gravity) 
        
        return self.reference
    
    def compute_chirp_trajectory(self):
        traj_length = jnp.linalg.norm(self.final_pos)
        #time_vec = jnp.linspace(0, self.dt, num=self.num_points, dtype=jnp.float32) # time vector
        time_vec = jnp.arange(0, self.dt, self.dt/self.num_points)
        A = 50.0      # Amplitude (meters)
       
        phi = 0.0    # Phase shift (radians)
        f0 = 0.01     # Initial frequency (Hz)
        f1 = 0.5     # Final frequency (Hz)
        
        
        z_t =  A * chirp(time_vec, f0=f0, f1=f1, t1=self.dt, method='quadratic')
        
        dz_t = jnp.gradient(z_t, self.dt/self.num_points)

        for i in range(self.num_points):
            self.reference = self.reference.at[i,2].set(z_t[i])

            # Payload reference
            #self.reference = self.reference.at[i,3].set(?)
            #self.reference = self.reference.at[i,4].set(?)
            #self.reference = self.reference.at[i,5].set(?)
            self.reference = self.reference.at[i,8].set(dz_t[i])

            self.reference = self.reference.at[i,-1].set((mass+mass_payload)*gravity) 
        
        return self.reference
    
    def compute_chirp_input(self):
        traj_length = jnp.linalg.norm(self.final_pos)
        #time_vec = jnp.linspace(0, self.dt, num=self.num_points, dtype=jnp.float32) # time vector
        time_vec = jnp.arange(0, self.dt, self.dt/self.num_points)
        A = 10.0      # Amplitude (meters)
       
        phi = 0.0    # Phase shift (radians)
        f0 = 0.001     # Initial frequency (Hz)
        f1 = 0.1    # Final frequency (Hz)
        
        
        input_t =  A * chirp(time_vec, f0=f0, f1=f1, t1=self.dt, method='linear')
        
        #dz_t = jnp.gradient(z_t, self.dt/self.num_points)
        for i in range(self.num_points):
            self.reference = self.reference.at[i,2].set(self.final_pos[2])

            # Payload reference
            #self.reference = self.reference.at[i,3].set(?)
            #self.reference = self.reference.at[i,4].set(?)
            #self.reference = self.reference.at[i,5].set(?)
            self.reference = self.reference.at[i,8].set(0)

            self.reference = self.reference.at[i,-1].set(input_t[i]) 
        
        return self.reference
    
    def compute_acceleration_stop_trajectory(self):
        # Parameters
        a = 2.0          # Acceleration (m/s^2)
        t_stop = self.dt * (1/2)     # Time at which the drone stops (seconds)

        # Time array    
        t = jnp.arange(0, self.dt,self.dt/self.num_points)

        #    Define position and velocity arrays
        z_t = jnp.zeros_like(t)
        dz_t = jnp.zeros_like(t)
        v_0 = 0.0
        # Compute the position and velocity before t_stop

        for i in range(len(t)):
            if i > 0:
                v_0 = self.reference[i-1,8]
            if t[i] <= t_stop:
                # During acceleration
                self.reference = self.reference.at[i,8].set(v_0 + a * t[i])  # Velocity: v = v_0 + at
                self.reference = self.reference.at[i,2].set(v_0 * t[i] + 0.5 * a * t[i]**2) # Position: z = z_0 + v_0*t + 0.5*a*t^2
                self.reference = self.reference.at[i,-1].set((mass+mass_payload)*gravity) 
            
            else:
                # After stop
                self.reference = self.reference.at[i,8].set(0)  # Velocity is 0 after stopping
                self.reference = self.reference.at[i,2].set(z_t[i-1]) # Position remains constant after stopping
                self.reference = self.reference.at[i,-1].set(0) 
        
        return self.reference
    
    def compute_circular_trajectory(self):
            #final_point = jnp.array([0.5,0.5,0.5]) # desired final coordinates xy (assuming starting point=(x=0,y=0))
            traj_length = jnp.linalg.norm(self.final_pos)
            radius = 2.0 # [m]
            z0_center = 7 # [m]
            y0_center = 0 # [m]
            angle_max = 2*jnp.pi
            time_vec = jnp.linspace(0, 10, num=self.num_points, dtype=jnp.float32) # time vector
            # s: path variable
            
            a0,a1,a2,a3,a4,a5 = self.poly5(t0=0, tf=2, q0=0, qf=angle_max, dq0=0, dqf=0, ddq0=0, ddqf=0)
            s_t = a0 + a1*time_vec + a2*time_vec**2 + a3*time_vec**3 + a4*time_vec**4 + a5*time_vec**5 # s(t)
            ds_t = a1 + 2*a2*time_vec + 3*a3*time_vec**2 + 4*a4*time_vec**3 + 5*a5*time_vec**4 # ds(t)
            theta_t = a0 + a1*time_vec + a2*time_vec**2 + a3*time_vec**3 + a4*time_vec**4 + a5*time_vec**5 # theta(t)
            dtheta_t = a1 + 2*a2*time_vec + 3*a3*time_vec**2 + 4*a4*time_vec**3 + 5*a5*time_vec**4 # dtheta(t)
            x_t = jnp.zeros((self.num_points,)) # x(t)
            dx_t = jnp.zeros((self.num_points,)) # x(t)
            y_t = y0_center + radius*jnp.sin(theta_t) # y(t)
            dy_t = radius*dtheta_t*jnp.cos(theta_t) # dy(t)
            z_t = z0_center + radius*jnp.cos(theta_t) # y(t)
            dz_t = -radius*dtheta_t*jnp.sin(theta_t) # dz(t)
            #w_dot = angle_max/self.Tsim_s # constant angular velocity
            # Assign variables to state
            for i in range(self.num_points):
                self.reference = self.reference.at[i,0].set(x_t[i])
                self.reference = self.reference.at[i,1].set(y_t[i])
                self.reference = self.reference.at[i,2].set(z_t[i])
                self.reference = self.reference.at[i,3].set(x_t[i])
                self.reference = self.reference.at[i,4].set(y_t[i])
                self.reference = self.reference.at[i,5].set(z_t[i])
                self.reference = self.reference.at[i,7].set(dx_t[i])
                self.reference = self.reference.at[i,8].set(dy_t[i])
                self.reference = self.reference.at[i,9].set(dz_t[i])
                self.reference = self.reference.at[i,10].set(dx_t[i])
                self.reference = self.reference.at[i,11].set(dy_t[i])
                self.reference = self.reference.at[i,12].set(dz_t[i])
                 # Attitude (yaw)
                
                self.reference = self.reference.at[i,-2:].set([(mass+mass_payload)*gravity/2, (mass+mass_payload)*gravity/2]) 
            return self.reference
    
    # Square trajectory
    def compute_square_trajectory(self):
        # SEGMENT 1
            # init_point = 0
            final_point = jnp.array([0,0,5.5]) # desired final coordinates xy (assuming starting point=(x=0,y=0))
            traj_length = jnp.linalg.norm(final_point)
            t1 = self.dt/5
            num = round(self.num_points/5)
            time_vec = jnp.linspace(0, t1, num=num, dtype=jnp.float32) # time vector
            a0,a1,a2,a3,a4,a5 = self.poly5(t0=0, tf=t1, q0=0, qf=traj_length, dq0=0, dqf=0, ddq0=0, ddqf=0)
            # s: path variable
            s_t = a0 + a1*time_vec + a2*time_vec**2 + a3*time_vec**3 + a4*time_vec**4 + a5*time_vec**5 # s(t)
            ds_t = a1 + 2*a2*time_vec + 3*a3*time_vec**2 + 4*a4*time_vec**3 + 5*a5*time_vec**4 # ds(t)
            # x
            x_t = final_point[0]/traj_length * s_t # x(t)
            dx_t = final_point[0]/traj_length * ds_t # dx(t)
            # y
            y_t = final_point[1]/traj_length * s_t # y(t)
            dy_t = final_point[1]/traj_length * ds_t # dy(t)

            z_t = final_point[2]/traj_length * s_t # z(t)
            dz_t = final_point[2]/traj_length * ds_t # dz(t)
            
            # Assign variables to state
            # Assign variables to state
            for i in range(num):
                self.reference = self.reference.at[i,0].set(0)
                self.reference = self.reference.at[i,1].set(y_t[i])
                self.reference = self.reference.at[i,2].set(z_t[i])
                self.reference = self.reference.at[i,7].set(0)
                self.reference = self.reference.at[i,8].set(dy_t[i])
                self.reference = self.reference.at[i,9].set(dz_t[i])

                self.reference = self.reference.at[i,-2:].set([(mass+mass_payload)*gravity/2, (mass+mass_payload)*gravity/2]) 

            # SEGMENT 2
            init_point = final_point
            final_point = jnp.array([0,1.0,6.5]) # desired final coordinates xy
            traj_length = jnp.linalg.norm(final_point-init_point)
            time_vec = jnp.linspace(0, t1, num=num, dtype=jnp.float32) # time vector
            a0,a1,a2,a3,a4,a5 = self.poly5(t0=0, tf=t1, q0=0, qf=traj_length, dq0=0, dqf=0, ddq0=0, ddqf=0)
            # s: path variable
            s_t = a0 + a1*time_vec + a2*time_vec**2 + a3*time_vec**3 + a4*time_vec**4 + a5*time_vec**5 # s(t)
            ds_t = a1 + 2*a2*time_vec + 3*a3*time_vec**2 + 4*a4*time_vec**3 + 5*a5*time_vec**4 # ds(t)
            # x
            x_t = init_point[0] + (final_point[0]-init_point[0])/traj_length * s_t # x(t)
            dx_t = (final_point[0]-init_point[0])/traj_length * ds_t # dx(t)
            # y
            y_t = init_point[1] + (final_point[1]-init_point[1])/traj_length * s_t # y(t)
            dy_t = (final_point[1]-init_point[1])/traj_length * ds_t # dy(t)

            z_t = init_point[2] + (final_point[2]-init_point[2])/traj_length * s_t # y(t)
            dz_t = (final_point[2]-init_point[2])/traj_length * ds_t # dy(t)

            for i in range(num):
                self.reference = self.reference.at[i+num,0].set(0)
                self.reference = self.reference.at[i+num,1].set(y_t[i])
                self.reference = self.reference.at[i+num,2].set(z_t[i])
                self.reference = self.reference.at[i+num,7].set(0)
                self.reference = self.reference.at[i+num,8].set(dy_t[i])
                self.reference = self.reference.at[i+num,9].set(dz_t[i])

                self.reference = self.reference.at[i+num,-2:].set([(mass+mass_payload)*gravity/2, (mass+mass_payload)*gravity/2]) 

            # SEGMENT 3
            init_point = final_point
            final_point = jnp.array([0.0,0.0,7.5]) # desired final coordinates xy
            traj_length = jnp.linalg.norm(final_point-init_point)
            time_vec = jnp.linspace(0, t1, num=num, dtype=jnp.float32) # time vector
            a0,a1,a2,a3,a4,a5 = self.poly5(t0=0, tf=t1, q0=0, qf=traj_length, dq0=0, dqf=0, ddq0=0, ddqf=0)
            # s: path variable
            s_t = a0 + a1*time_vec + a2*time_vec**2 + a3*time_vec**3 + a4*time_vec**4 + a5*time_vec**5 # s(t)
            ds_t = a1 + 2*a2*time_vec + 3*a3*time_vec**2 + 4*a4*time_vec**3 + 5*a5*time_vec**4 # ds(t)
            # x
            x_t = init_point[0] + (final_point[0]-init_point[0])/traj_length * s_t # x(t)
            dx_t = (final_point[0]-init_point[0])/traj_length * ds_t # dx(t)
            # y
            y_t = init_point[1] + (final_point[1]-init_point[1])/traj_length * s_t # y(t)
            dy_t = (final_point[1]-init_point[1])/traj_length * ds_t # dy(t)
            # Assign variables to state
            for i in range(num):
                self.reference = self.reference.at[i+2*num,0].set(0)
                self.reference = self.reference.at[i+2*num,1].set(y_t[i])
                self.reference = self.reference.at[i+2*num,2].set(z_t[i])
                self.reference = self.reference.at[i+2*num,3].set(1)
                self.reference = self.reference.at[i+2*num,7].set(0)
                self.reference = self.reference.at[i+2*num,8].set(dy_t[i])
                self.reference = self.reference.at[i+2*num,9].set(dz_t[i])

                self.reference = self.reference.at[i+2*num,-2:].set([(mass+mass_payload)*gravity/2, (mass+mass_payload)*gravity/2]) 
            # SEGMENT 4
            init_point = final_point
            final_point = jnp.array([0,-1.0,6.5]) # desired final coordinates xy
            traj_length = jnp.linalg.norm(final_point-init_point)
            time_vec = jnp.linspace(0, t1, num=num, dtype=jnp.float32) # time vector
            a0,a1,a2,a3,a4,a5 = self.poly5(t0=0, tf=t1, q0=0, qf=traj_length, dq0=0, dqf=0, ddq0=0, ddqf=0)
            # s: path variable
            s_t = a0 + a1*time_vec + a2*time_vec**2 + a3*time_vec**3 + a4*time_vec**4 + a5*time_vec**5 # s(t)
            ds_t = a1 + 2*a2*time_vec + 3*a3*time_vec**2 + 4*a4*time_vec**3 + 5*a5*time_vec**4 # ds(t)
            # x
            x_t = init_point[0] + (final_point[0]-init_point[0])/traj_length * s_t # x(t)
            dx_t = (final_point[0]-init_point[0])/traj_length * ds_t # dx(t)
            # y
            y_t = init_point[1] + (final_point[1]-init_point[1])/traj_length * s_t # y(t)
            dy_t = (final_point[1]-init_point[1])/traj_length * ds_t # dy(t)

            z_t = init_point[2] + (final_point[2]-init_point[2])/traj_length * s_t # y(t)
            dz_t = (final_point[2]-init_point[2])/traj_length * ds_t # dy(t)

            # Assign variables to state
            for i in range(num):
                self.reference = self.reference.at[i+3*num,0].set(0)
                self.reference = self.reference.at[i+3*num,1].set(y_t[i])
                self.reference = self.reference.at[i+3*num,2].set(z_t[i])
                self.reference = self.reference.at[i+3*num,7].set(0)
                self.reference = self.reference.at[i+3*num,8].set(dy_t[i])
                self.reference = self.reference.at[i+3*num,9].set(dz_t[i])

                self.reference = self.reference.at[i+3*num,-2:].set([(mass+mass_payload)*gravity/2, (mass+mass_payload)*gravity/2]) 
            
            # SEGMENT 5
            init_point = final_point
            final_point = jnp.array([0.0,0.0,5.5]) # desired final coordinates xy
            traj_length = jnp.linalg.norm(final_point-init_point)
            time_vec = jnp.linspace(0, t1, num=num, dtype=jnp.float32) # time vector
            a0,a1,a2,a3,a4,a5 = self.poly5(t0=0, tf=t1, q0=0, qf=traj_length, dq0=0, dqf=0, ddq0=0, ddqf=0)
            # s: path variable
            s_t = a0 + a1*time_vec + a2*time_vec**2 + a3*time_vec**3 + a4*time_vec**4 + a5*time_vec**5 # s(t)
            ds_t = a1 + 2*a2*time_vec + 3*a3*time_vec**2 + 4*a4*time_vec**3 + 5*a5*time_vec**4 # ds(t)
            # x
            x_t = init_point[0] + (final_point[0]-init_point[0])/traj_length * s_t # x(t)
            dx_t = (final_point[0]-init_point[0])/traj_length * ds_t # dx(t)
            # y
            y_t = init_point[1] + (final_point[1]-init_point[1])/traj_length * s_t # y(t)
            dy_t = (final_point[1]-init_point[1])/traj_length * ds_t # dy(t)

            z_t = init_point[2] + (final_point[2]-init_point[2])/traj_length * s_t # y(t)
            dz_t = (final_point[2]-init_point[2])/traj_length * ds_t # dy(t)

            # Assign variables to state
            for i in range(num):
                self.reference = self.reference.at[i+4*num+1,0].set(0)
                self.reference = self.reference.at[i+4*num+1,1].set(y_t[i])
                self.reference = self.reference.at[i+4*num+1,2].set(z_t[i])
                self.reference = self.reference.at[i+4*num+1,7].set(0)
                self.reference = self.reference.at[i+4*num+1,8].set(dy_t[i])
                self.reference = self.reference.at[i+4*num+1,9].set(dz_t[i])

                self.reference = self.reference.at[i+4*num+1,-2:].set([(mass+mass_payload)*gravity/2, (mass+mass_payload)*gravity/2]) 
            return self.reference     
   #Linear Trajectory
    def compute_square_trajectory3(self):
        
        traj_length = jnp.linalg.norm(self.final_pos)
        first_point_start = jnp.array([0,0,5.5])
        first_point_end = jnp.array([0,1.0,7.5])
        time_vec = jnp.linspace(0, 1, num=int(self.num_points/4 - 0.5), dtype=jnp.float32) # time vector
        a0x,a1x,a2x,a3x,a4x,a5x = self.poly5(t0=0, tf=1, q0=first_point_start[0], qf=first_point_end[0], dq0=0, dqf=0, ddq0=0, ddqf=0)
        a0y,a1y,a2y,a3y,a4y,a5y = self.poly5(t0=0, tf=1, q0=first_point_start[1], qf=first_point_end[1], dq0=0, dqf=0, ddq0=0, ddqf=0)
        a0z,a1z,a2z,a3z,a4z,a5z = self.poly5(t0=0, tf=1, q0=first_point_start[2], qf=first_point_end[2], dq0=0, dqf=0, ddq0=0, ddqf=0)
        time_in_seconds = time_vec * self.dt

        # s: path variable
        s_tx = a0x + a1x*time_in_seconds/self.dt + a2x*(time_in_seconds/self.dt)**2 + a3x*(time_in_seconds/self.dt)**3 + a4x*(time_in_seconds/self.dt)**4 + a5x*(time_in_seconds/self.dt)**5 # s(t)
        dsx_t = a1x + 2*a2x*(time_in_seconds/self.dt) + 3*a3x*(time_in_seconds/self.dt)**2 + 4*a4x*(time_in_seconds/self.dt)**3 + 5*a5x*(time_in_seconds/self.dt)**4 # ds(t)

        s_ty = a0y + a1y*(time_in_seconds/self.dt) + a2y*(time_in_seconds/self.dt)**2 + a3y*(time_in_seconds/self.dt)**3 + a4y*(time_in_seconds/self.dt)**4 + a5y*(time_in_seconds/self.dt)**5 # s(t)
        dsy_t = a1y + 2*a2y*(time_in_seconds/self.dt) + 3*a3y*(time_in_seconds/self.dt)**2 + 4*a4y*(time_in_seconds/self.dt)**3 + 5*a5y*(time_in_seconds/self.dt)**4 # ds(t)

        s_tz = a0z + a1z*(time_in_seconds/self.dt) + a2z*(time_in_seconds/self.dt)**2 + a3z*(time_in_seconds/self.dt)**3 + a4z*(time_in_seconds/self.dt)**4 + a5z*(time_in_seconds/self.dt)**5 # s(t)
        dsz_t = a1z + 2*a2z*(time_in_seconds/self.dt) + 3*a3z*(time_in_seconds/self.dt)**2 + 4*a4z*(time_in_seconds/self.dt)**3 + 5*a5z*(time_in_seconds/self.dt)**4 # ds(t)

        """
        # s: path variable
        s_tx = a0x + a1x*time_vec + a2x*time_vec**2 + a3x*time_vec**3 + a4x*time_vec**4 + a5x*time_vec**5 # s(t)
        dsx_t = a1x + 2*a2x*time_vec + 3*a3x*time_vec**2 + 4*a4x*time_vec**3 + 5*a5x*time_vec**4 # ds(t)

        s_ty = a0y + a1y*time_vec + a2y*time_vec**2 + a3y*time_vec**3 + a4y*time_vec**4 + a5y*time_vec**5 # s(t)
        dsy_t = a1y + 2*a2y*time_vec + 3*a3y*time_vec**2 + 4*a4y*time_vec**3 + 5*a5y*time_vec**4 # ds(t)

        s_tz = a0z + a1z*time_vec + a2z*time_vec**2 + a3z*time_vec**3 + a4z*time_vec**4 + a5z*time_vec**5 # s(t)
        dsz_t = a1z + 2*a2z*time_vec + 3*a3z*time_vec**2 + 4*a4z*time_vec**3 + 5*a5z*time_vec**4 # ds(t)
        """
        # x
        x_t = s_tx # x(t)
        dx_t = dsx_t/self.dt # dx(t)
        # y
        y_t = s_ty # y(t)
        dy_t = dsy_t/self.dt # dy(t)

        z_t = s_tz # y(t)
        dz_t = dsz_t/self.dt # dy(t)
        
        for i in range(int(self.num_points/4 - 0.5)):
            


            # Drone position reference
            self.reference = self.reference.at[i,0].set(x_t[i])
            self.reference = self.reference.at[i,1].set(y_t[i])
            self.reference = self.reference.at[i,2].set(z_t[i])

            # Payload position reference
            self.reference = self.reference.at[i,3].set(x_t[i])
            self.reference = self.reference.at[i,4].set(y_t[i])
            self.reference = self.reference.at[i,5].set(z_t[i])

            # Drone velocity reference
            
            self.reference = self.reference.at[i,7].set(dx_t[i])
            self.reference = self.reference.at[i,8].set(dy_t[i])
            self.reference = self.reference.at[i,9].set(dz_t[i])

            self.reference = self.reference.at[i,10].set(dx_t[i])
            self.reference = self.reference.at[i,11].set(dy_t[i])
            self.reference = self.reference.at[i,12].set(dz_t[i])


            # Input reference
            self.reference = self.reference.at[i,-2:].set([(mass+mass_payload)*gravity/2 , (mass+mass_payload)*gravity/2]) 
        
        # Second edge
        traj_length = jnp.linalg.norm(self.final_pos)
        second_point_start = jnp.array([0,1.0,7.5])
        second_point_end = jnp.array([0,0.0,9.5])
        time_vec = jnp.linspace(0, 1, num=int(self.num_points/4 + 0.5), dtype=jnp.float32) # time vector
        a0x,a1x,a2x,a3x,a4x,a5x = self.poly5(t0=0, tf=1, q0=second_point_start[0], qf=second_point_end[0], dq0=0, dqf=0, ddq0=0, ddqf=0)
        a0y,a1y,a2y,a3y,a4y,a5y = self.poly5(t0=0, tf=1, q0=second_point_start[1], qf=second_point_end[1], dq0=0, dqf=0, ddq0=0, ddqf=0)
        a0z,a1z,a2z,a3z,a4z,a5z = self.poly5(t0=0, tf=1, q0=second_point_start[2], qf=second_point_end[2], dq0=0, dqf=0, ddq0=0, ddqf=0)
        time_in_seconds = time_vec * self.dt

        # s: path variable
        s_tx = a0x + a1x*time_in_seconds/self.dt + a2x*(time_in_seconds/self.dt)**2 + a3x*(time_in_seconds/self.dt)**3 + a4x*(time_in_seconds/self.dt)**4 + a5x*(time_in_seconds/self.dt)**5 # s(t)
        dsx_t = a1x + 2*a2x*(time_in_seconds/self.dt) + 3*a3x*(time_in_seconds/self.dt)**2 + 4*a4x*(time_in_seconds/self.dt)**3 + 5*a5x*(time_in_seconds/self.dt)**4 # ds(t)

        s_ty = a0y + a1y*(time_in_seconds/self.dt) + a2y*(time_in_seconds/self.dt)**2 + a3y*(time_in_seconds/self.dt)**3 + a4y*(time_in_seconds/self.dt)**4 + a5y*(time_in_seconds/self.dt)**5 # s(t)
        dsy_t = a1y + 2*a2y*(time_in_seconds/self.dt) + 3*a3y*(time_in_seconds/self.dt)**2 + 4*a4y*(time_in_seconds/self.dt)**3 + 5*a5y*(time_in_seconds/self.dt)**4 # ds(t)

        s_tz = a0z + a1z*(time_in_seconds/self.dt) + a2z*(time_in_seconds/self.dt)**2 + a3z*(time_in_seconds/self.dt)**3 + a4z*(time_in_seconds/self.dt)**4 + a5z*(time_in_seconds/self.dt)**5 # s(t)
        dsz_t = a1z + 2*a2z*(time_in_seconds/self.dt) + 3*a3z*(time_in_seconds/self.dt)**2 + 4*a4z*(time_in_seconds/self.dt)**3 + 5*a5z*(time_in_seconds/self.dt)**4 # ds(t)

        """
        # s: path variable
        s_tx = a0x + a1x*time_vec + a2x*time_vec**2 + a3x*time_vec**3 + a4x*time_vec**4 + a5x*time_vec**5 # s(t)
        dsx_t = a1x + 2*a2x*time_vec + 3*a3x*time_vec**2 + 4*a4x*time_vec**3 + 5*a5x*time_vec**4 # ds(t)

        s_ty = a0y + a1y*time_vec + a2y*time_vec**2 + a3y*time_vec**3 + a4y*time_vec**4 + a5y*time_vec**5 # s(t)
        dsy_t = a1y + 2*a2y*time_vec + 3*a3y*time_vec**2 + 4*a4y*time_vec**3 + 5*a5y*time_vec**4 # ds(t)

        s_tz = a0z + a1z*time_vec + a2z*time_vec**2 + a3z*time_vec**3 + a4z*time_vec**4 + a5z*time_vec**5 # s(t)
        dsz_t = a1z + 2*a2z*time_vec + 3*a3z*time_vec**2 + 4*a4z*time_vec**3 + 5*a5z*time_vec**4 # ds(t)
        """
        # x
        x_t = s_tx # x(t)
        dx_t = dsx_t/self.dt # dx(t)
        # y
        y_t = s_ty # y(t)
        dy_t = dsy_t/self.dt # dy(t)

        z_t = s_tz # y(t)
        dz_t = dsz_t/self.dt # dy(t)
        
        for i in range(int(self.num_points/4 - 0.5), int(self.num_points/2) ):
            


            # Drone position reference
            self.reference = self.reference.at[i,0].set(x_t[i])
            self.reference = self.reference.at[i,1].set(y_t[i])
            self.reference = self.reference.at[i,2].set(z_t[i])

            # Payload position reference
            self.reference = self.reference.at[i,3].set(x_t[i])
            self.reference = self.reference.at[i,4].set(y_t[i])
            self.reference = self.reference.at[i,5].set(z_t[i])

            # Drone velocity reference
            
            self.reference = self.reference.at[i,7].set(dx_t[i])
            self.reference = self.reference.at[i,8].set(dy_t[i])
            self.reference = self.reference.at[i,9].set(dz_t[i])

            self.reference = self.reference.at[i,10].set(dx_t[i])
            self.reference = self.reference.at[i,11].set(dy_t[i])
            self.reference = self.reference.at[i,12].set(dz_t[i])


            # Input reference
            self.reference = self.reference.at[i,-2:].set([(mass+mass_payload)*gravity/2 , (mass+mass_payload)*gravity/2]) 

        # Third edge

        traj_length = jnp.linalg.norm(self.final_pos)
        third_point_start = jnp.array([0,0.0,9.5])
        third_point_end = jnp.array([0,-1.0,7.5])
        time_vec = jnp.linspace(0, 1, num=int(self.num_points/4 - 0.5), dtype=jnp.float32) # time vector
        a0x,a1x,a2x,a3x,a4x,a5x = self.poly5(t0=0, tf=1, q0=third_point_start[0], qf=third_point_end[0], dq0=0, dqf=0, ddq0=0, ddqf=0)
        a0y,a1y,a2y,a3y,a4y,a5y = self.poly5(t0=0, tf=1, q0=third_point_start[1], qf=third_point_end[1], dq0=0, dqf=0, ddq0=0, ddqf=0)
        a0z,a1z,a2z,a3z,a4z,a5z = self.poly5(t0=0, tf=1, q0=third_point_start[2], qf=third_point_end[2], dq0=0, dqf=0, ddq0=0, ddqf=0)
        time_in_seconds = time_vec * self.dt

        # s: path variable
        s_tx = a0x + a1x*time_in_seconds/self.dt + a2x*(time_in_seconds/self.dt)**2 + a3x*(time_in_seconds/self.dt)**3 + a4x*(time_in_seconds/self.dt)**4 + a5x*(time_in_seconds/self.dt)**5 # s(t)
        dsx_t = a1x + 2*a2x*(time_in_seconds/self.dt) + 3*a3x*(time_in_seconds/self.dt)**2 + 4*a4x*(time_in_seconds/self.dt)**3 + 5*a5x*(time_in_seconds/self.dt)**4 # ds(t)

        s_ty = a0y + a1y*(time_in_seconds/self.dt) + a2y*(time_in_seconds/self.dt)**2 + a3y*(time_in_seconds/self.dt)**3 + a4y*(time_in_seconds/self.dt)**4 + a5y*(time_in_seconds/self.dt)**5 # s(t)
        dsy_t = a1y + 2*a2y*(time_in_seconds/self.dt) + 3*a3y*(time_in_seconds/self.dt)**2 + 4*a4y*(time_in_seconds/self.dt)**3 + 5*a5y*(time_in_seconds/self.dt)**4 # ds(t)

        s_tz = a0z + a1z*(time_in_seconds/self.dt) + a2z*(time_in_seconds/self.dt)**2 + a3z*(time_in_seconds/self.dt)**3 + a4z*(time_in_seconds/self.dt)**4 + a5z*(time_in_seconds/self.dt)**5 # s(t)
        dsz_t = a1z + 2*a2z*(time_in_seconds/self.dt) + 3*a3z*(time_in_seconds/self.dt)**2 + 4*a4z*(time_in_seconds/self.dt)**3 + 5*a5z*(time_in_seconds/self.dt)**4 # ds(t)

        """
        # s: path variable
        s_tx = a0x + a1x*time_vec + a2x*time_vec**2 + a3x*time_vec**3 + a4x*time_vec**4 + a5x*time_vec**5 # s(t)
        dsx_t = a1x + 2*a2x*time_vec + 3*a3x*time_vec**2 + 4*a4x*time_vec**3 + 5*a5x*time_vec**4 # ds(t)

        s_ty = a0y + a1y*time_vec + a2y*time_vec**2 + a3y*time_vec**3 + a4y*time_vec**4 + a5y*time_vec**5 # s(t)
        dsy_t = a1y + 2*a2y*time_vec + 3*a3y*time_vec**2 + 4*a4y*time_vec**3 + 5*a5y*time_vec**4 # ds(t)

        s_tz = a0z + a1z*time_vec + a2z*time_vec**2 + a3z*time_vec**3 + a4z*time_vec**4 + a5z*time_vec**5 # s(t)
        dsz_t = a1z + 2*a2z*time_vec + 3*a3z*time_vec**2 + 4*a4z*time_vec**3 + 5*a5z*time_vec**4 # ds(t)
        """
        # x
        x_t = s_tx # x(t)
        dx_t = dsx_t/self.dt # dx(t)
        # y
        y_t = s_ty # y(t)
        dy_t = dsy_t/self.dt # dy(t)

        z_t = s_tz # y(t)
        dz_t = dsz_t/self.dt # dy(t)
        
        for i in range(int(self.num_points/2) ,int(3*self.num_points/4 - 0.5)):
            


            # Drone position reference
            self.reference = self.reference.at[i,0].set(x_t[i])
            self.reference = self.reference.at[i,1].set(y_t[i])
            self.reference = self.reference.at[i,2].set(z_t[i])

            # Payload position reference
            self.reference = self.reference.at[i,3].set(x_t[i])
            self.reference = self.reference.at[i,4].set(y_t[i])
            self.reference = self.reference.at[i,5].set(z_t[i])

            # Drone velocity reference
            
            self.reference = self.reference.at[i,7].set(dx_t[i])
            self.reference = self.reference.at[i,8].set(dy_t[i])
            self.reference = self.reference.at[i,9].set(dz_t[i])

            self.reference = self.reference.at[i,10].set(dx_t[i])
            self.reference = self.reference.at[i,11].set(dy_t[i])
            self.reference = self.reference.at[i,12].set(dz_t[i])


            # Input reference
            self.reference = self.reference.at[i,-2:].set([(mass+mass_payload)*gravity/2 , (mass+mass_payload)*gravity/2]) 


        #fourth edge
        traj_length = jnp.linalg.norm(self.final_pos)
        fourth_point_start = jnp.array([0,-1.0,7.5])
        fourth_point_end = jnp.array([0,0.0,5.5])
        time_vec = jnp.linspace(0, 1, num=int(self.num_points/4 + 0.5), dtype=jnp.float32) # time vector
        a0x,a1x,a2x,a3x,a4x,a5x = self.poly5(t0=0, tf=1, q0=fourth_point_start[0], qf=fourth_point_end[0], dq0=0, dqf=0, ddq0=0, ddqf=0)
        a0y,a1y,a2y,a3y,a4y,a5y = self.poly5(t0=0, tf=1, q0=fourth_point_start[1], qf=fourth_point_end[1], dq0=0, dqf=0, ddq0=0, ddqf=0)
        a0z,a1z,a2z,a3z,a4z,a5z = self.poly5(t0=0, tf=1, q0=fourth_point_start[2], qf=fourth_point_end[2], dq0=0, dqf=0, ddq0=0, ddqf=0)
        time_in_seconds = time_vec * self.dt

        # s: path variable
        s_tx = a0x + a1x*time_in_seconds/self.dt + a2x*(time_in_seconds/self.dt)**2 + a3x*(time_in_seconds/self.dt)**3 + a4x*(time_in_seconds/self.dt)**4 + a5x*(time_in_seconds/self.dt)**5 # s(t)
        dsx_t = a1x + 2*a2x*(time_in_seconds/self.dt) + 3*a3x*(time_in_seconds/self.dt)**2 + 4*a4x*(time_in_seconds/self.dt)**3 + 5*a5x*(time_in_seconds/self.dt)**4 # ds(t)

        s_ty = a0y + a1y*(time_in_seconds/self.dt) + a2y*(time_in_seconds/self.dt)**2 + a3y*(time_in_seconds/self.dt)**3 + a4y*(time_in_seconds/self.dt)**4 + a5y*(time_in_seconds/self.dt)**5 # s(t)
        dsy_t = a1y + 2*a2y*(time_in_seconds/self.dt) + 3*a3y*(time_in_seconds/self.dt)**2 + 4*a4y*(time_in_seconds/self.dt)**3 + 5*a5y*(time_in_seconds/self.dt)**4 # ds(t)

        s_tz = a0z + a1z*(time_in_seconds/self.dt) + a2z*(time_in_seconds/self.dt)**2 + a3z*(time_in_seconds/self.dt)**3 + a4z*(time_in_seconds/self.dt)**4 + a5z*(time_in_seconds/self.dt)**5 # s(t)
        dsz_t = a1z + 2*a2z*(time_in_seconds/self.dt) + 3*a3z*(time_in_seconds/self.dt)**2 + 4*a4z*(time_in_seconds/self.dt)**3 + 5*a5z*(time_in_seconds/self.dt)**4 # ds(t)

        """
        # s: path variable
        s_tx = a0x + a1x*time_vec + a2x*time_vec**2 + a3x*time_vec**3 + a4x*time_vec**4 + a5x*time_vec**5 # s(t)
        dsx_t = a1x + 2*a2x*time_vec + 3*a3x*time_vec**2 + 4*a4x*time_vec**3 + 5*a5x*time_vec**4 # ds(t)

        s_ty = a0y + a1y*time_vec + a2y*time_vec**2 + a3y*time_vec**3 + a4y*time_vec**4 + a5y*time_vec**5 # s(t)
        dsy_t = a1y + 2*a2y*time_vec + 3*a3y*time_vec**2 + 4*a4y*time_vec**3 + 5*a5y*time_vec**4 # ds(t)

        s_tz = a0z + a1z*time_vec + a2z*time_vec**2 + a3z*time_vec**3 + a4z*time_vec**4 + a5z*time_vec**5 # s(t)
        dsz_t = a1z + 2*a2z*time_vec + 3*a3z*time_vec**2 + 4*a4z*time_vec**3 + 5*a5z*time_vec**4 # ds(t)
        """
        # x
        x_t = s_tx # x(t)
        dx_t = dsx_t/self.dt # dx(t)
        # y
        y_t = s_ty # y(t)
        dy_t = dsy_t/self.dt # dy(t)

        z_t = s_tz # y(t)
        dz_t = dsz_t/self.dt # dy(t)
        
        for i in range(int(3*self.num_points/4 - 0.5), int(self.num_points)):
            


            # Drone position reference
            self.reference = self.reference.at[i,0].set(x_t[i])
            self.reference = self.reference.at[i,1].set(y_t[i])
            self.reference = self.reference.at[i,2].set(z_t[i])

            # Payload position reference
            self.reference = self.reference.at[i,3].set(x_t[i])
            self.reference = self.reference.at[i,4].set(y_t[i])
            self.reference = self.reference.at[i,5].set(z_t[i])

            # Drone velocity reference
            
            self.reference = self.reference.at[i,7].set(dx_t[i])
            self.reference = self.reference.at[i,8].set(dy_t[i])
            self.reference = self.reference.at[i,9].set(dz_t[i])

            self.reference = self.reference.at[i,10].set(dx_t[i])
            self.reference = self.reference.at[i,11].set(dy_t[i])
            self.reference = self.reference.at[i,12].set(dz_t[i])


            # Input reference
            self.reference = self.reference.at[i,-2:].set([(mass+mass_payload)*gravity/2 , (mass+mass_payload)*gravity/2]) 

        return self.reference

    
    

 #sinusoidal  Trajectory
    def compute_sinusoidal_trajectory3(self):
        

        # Define parameters for the sinusoidal trajectories
        A_drone = 2.5      # Amplitude of the drone's z-axis motion
        A_payload = 1.0    # Amplitude of the payload's z-axis motion
        n_oscillations = 20 # Number of full oscillations to perform

        # Time duration for the entire trajectory
        T_total = self.dt  # Total time in seconds

        # Compute the required frequency to fit the oscillations within the time
        f_drone = n_oscillations / T_total  # Frequency for the drone's motion (in Hz)
        f_payload = n_oscillations / T_total  # Frequency for the payload (same number of oscillations)

        # Phase offsets (can adjust if needed)
        phi_drone = 0.0
        phi_payload = 0  # 90 degree phase shift for payload

        # Create the time array (sampling rate: num_points points over the total time)
        t = jnp.linspace(0, T_total, self.num_points, dtype=jnp.float32)

        # Calculate the drone's sinusoidal position (z-axis)
        z_drone = 0.5 + self.initial_pos[2] + A_drone * jnp.sin(2 * jnp.pi * f_drone * t + phi_drone)

        # Calculate the payload's sinusoidal position (z-axis)
        z_payload = self.initial_pos[2] + A_payload * jnp.sin(2 * jnp.pi * f_payload * t + phi_payload)

        # Compute the corresponding velocities (derivatives of position)
        v_drone = A_drone * 2 * jnp.pi * f_drone * jnp.cos(2 * jnp.pi * f_drone * t + phi_drone)
        v_payload = A_payload * 2 * jnp.pi * f_payload * jnp.cos(2 * jnp.pi * f_payload * t + phi_payload)

        
        
        
        # x
        x_t = jnp.zeros(self.num_points) # x(t)
        dx_t = jnp.zeros(self.num_points) # dx(t)
        # y
        y_t = jnp.zeros(self.num_points) # y(t)
        dy_t = jnp.zeros(self.num_points) # dy(t)

        #z_t = s_tz # y(t)
        #dz_t = dsz_t/self.dt # dy(t)
        
        for i in range(self.num_points):
            


            # Drone position reference
            self.reference = self.reference.at[i,0].set(x_t[i])
            self.reference = self.reference.at[i,1].set(y_t[i])
            self.reference = self.reference.at[i,2].set(z_drone[i])

            # Payload position reference
            self.reference = self.reference.at[i,3].set(x_t[i])
            self.reference = self.reference.at[i,4].set(y_t[i])
            self.reference = self.reference.at[i,5].set(z_payload[i])

            # Drone velocity reference
            
            self.reference = self.reference.at[i,7].set(dx_t[i])
            self.reference = self.reference.at[i,8].set(dy_t[i])
            self.reference = self.reference.at[i,9].set(v_drone[i])

            self.reference = self.reference.at[i,10].set(dx_t[i])
            self.reference = self.reference.at[i,11].set(dy_t[i])
            self.reference = self.reference.at[i,12].set(v_payload[i])


            # Input reference
            self.reference = self.reference.at[i,-2:].set([(mass+mass_payload)*gravity/2 , (mass+mass_payload)*gravity/2]) 
        
        return self.reference
       
#sinusoidal  Trajectory
    def compute_sinusoidal_trajectory_up_lateral3(self):
        

        # Define parameters for the sinusoidal trajectories in z-axis
        A_drone_z = 2.0      # Amplitude of the drone's z-axis motion
        A_payload_z = 1.0    # Amplitude of the payload's z-axis motion
        n_oscillations_z = 20 # Number of full oscillations in the z-axis

        # Define parameters for the bell-shaped velocity in y-axis
        V_max_drone_y = 1.0  # Maximum velocity for drone in y-axis (m/s)
        V_max_payload_y = 1.0  # Maximum velocity for payload in y-axis (m/s)
        y0_drone = self.initial_pos[1]       # Initial position of the drone in y-axis
        y0_payload = self.initial_pos[1]    # Initial position of the payload in y-axis

        # Total time for the entire trajectory
        T_total = self.dt  # Total time in seconds

        # Compute the required frequency to fit the oscillations within the time
        f_drone_z = n_oscillations_z / T_total  # Frequency for the drone's z motion (in Hz)
        f_payload_z = n_oscillations_z / T_total  # Frequency for the payload's z motion (same number of oscillations)

        # Phase offsets (can adjust if needed)
        phi_drone_z = 0.0
        phi_payload_z = 0  # 90 degree phase shift for payload in z

        # Create the time array (sampling rate: 1000 points over the total time)
        t = jnp.linspace(0, T_total, 1000)

        # Calculate the drone's sinusoidal position (z-axis)
        z_drone = 0.5 + self.initial_pos[2] + A_drone_z * jnp.sin(2 * jnp.pi * f_drone_z * t + phi_drone_z)

        # Calculate the payload's sinusoidal position (z-axis)
        z_payload = self.initial_pos[2] + A_payload_z * jnp.sin(2 * jnp.pi * f_payload_z * t + phi_payload_z)

        # Calculate the bell-shaped velocity for the y-axis (cosine-shaped velocity profile)
        v_drone_y = V_max_drone_y * (1 - jnp.cos(jnp.pi * t / T_total))  # Drone's bell-shaped velocity in y-axis
        v_payload_y = V_max_payload_y * (1 - jnp.cos(jnp.pi * t / T_total))  # Payload's bell-shaped velocity in y-axis

        # Integrate the velocity to get the position in the y-axis
        y_drone = y0_drone + jnp.cumsum(v_drone_y) * (T_total / 1000)
        y_payload = y0_payload + jnp.cumsum(v_payload_y) * (T_total / 1000)

        # Compute the corresponding velocities for the z-axis
        v_drone_z = A_drone_z * 2 * jnp.pi * f_drone_z * jnp.cos(2 * jnp.pi * f_drone_z * t + phi_drone_z)
        v_payload_z = A_payload_z * 2 * jnp.pi * f_payload_z * jnp.cos(2 * jnp.pi * f_payload_z * t + phi_payload_z)


        
        
        
        # x
        x_t = jnp.zeros(self.num_points) # x(t)
        dx_t = jnp.zeros(self.num_points) # dx(t)
        # y
        y_t = jnp.zeros(self.num_points) # y(t)
        dy_t = jnp.zeros(self.num_points) # dy(t)

        #z_t = s_tz # y(t)
        #dz_t = dsz_t/self.dt # dy(t)
        
        for i in range(self.num_points):
            


            # Drone position reference
            self.reference = self.reference.at[i,0].set(x_t[i])
            self.reference = self.reference.at[i,1].set(y_drone[i])
            self.reference = self.reference.at[i,2].set(z_drone[i])

            # Payload position reference
            self.reference = self.reference.at[i,3].set(x_t[i])
            self.reference = self.reference.at[i,4].set(y_payload[i])
            self.reference = self.reference.at[i,5].set(z_payload[i])

            # Drone velocity reference
            
            self.reference = self.reference.at[i,7].set(dx_t[i])
            self.reference = self.reference.at[i,8].set(v_drone_y[i])
            self.reference = self.reference.at[i,9].set(v_drone_z[i])

            self.reference = self.reference.at[i,10].set(dx_t[i])
            self.reference = self.reference.at[i,11].set(v_payload_y[i])
            self.reference = self.reference.at[i,12].set(v_payload_z[i])


            # Input reference
            self.reference = self.reference.at[i,-2:].set([(mass+mass_payload)*gravity/2 , (mass+mass_payload)*gravity/2]) 
        
        return self.reference
                  
#Example usage
#initial_pos = jnp.array([0, 0, 0])
#final_pos = jnp.array([0.5, 0.5, 0.5])

#calculator = Trapeizoidal_Trajectory(initial_pos, final_pos, 2, 526)
#trajectory = calculator.compute_trajectory4()
#reference = jnp.zeros((526, 17),dtype=jnp.float32)
#for i in range(526):
#    reference = reference.at[i,0:3].set(trajectory[i,0:3])
#    reference = reference.at[i,3].set(1)
#    reference = reference.at[i,7:10].set(trajectory[i,3:6])
#    reference = reference.at[i,-4].set(9.8) 
   
#print('ref', reference[:10,0:8])
#print('ref', reference[:10,8:17])
#print('traj size', len(trajectory))
# Print the trajectory
#for pos in trajectory:
#    print(pos)
    