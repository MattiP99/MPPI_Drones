import jax.numpy as jnp
import sys
sys.path.append('/home/mpiras/MPPI/sbmpc2/sbmpc')
import sbmpc.utils.geometry as geom
from scipy.signal import chirp

mass = 2.7
mass_payload = 0.25
cable_length = 0.5

gravity = 9.81
dimension = 13

class Trapeizoidal_Trajectory:
    def __init__(self, initial_pos, final_pos,dt, num_points):
        self.initial_pos = initial_pos
        self.final_pos = final_pos
        self.num_points = num_points
        self.dt=dt
        self.reference = jnp.zeros((self.num_points,dimension ),dtype=jnp.float32)


    
    def poly3(self,t0,tf,q0,qf,dq0,dqf):
        '''Compute coefficents for third order polynomial'''
        a0 = q0
        a1 = dq0
        a2 = (-3*(q0-qf)-(2*dq0+dqf)*tf)/tf**2
        a3 = (2*(q0-qf)+(dq0+dqf)*tf)/tf**3
        return a0,a1,a2,a3
    
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
    def compute_linear_trajectory(self):
        
        traj_length = jnp.linalg.norm(self.final_pos)
        time_vec = jnp.linspace(0, self.dt, num=self.num_points, dtype=jnp.float32) # time vector
        a0,a1,a2,a3,a4,a5 = self.poly5(t0=0, tf=self.dt, q0=0, qf=traj_length, dq0=0, dqf=0, ddq0=0, ddqf=0)
        # s: path variable
        s_t = a0 + a1*time_vec + a2*time_vec**2 + a3*time_vec**3 + a4*time_vec**4 + a5*time_vec**5 # s(t)
        ds_t = a1 + 2*a2*time_vec + 3*a3*time_vec**2 + 4*a4*time_vec**3 + 5*a5*time_vec**4 # ds(t)
        # x
        x_t = self.final_pos[0]/traj_length * s_t # x(t)
        dx_t = self.final_pos[0]/traj_length * ds_t # dx(t)
        # y
        y_t = self.final_pos[1]/traj_length * s_t # y(t)
        dy_t = self.final_pos[1]/traj_length * ds_t # dy(t)

        z_t = self.final_pos[2]/traj_length * s_t # y(t)
        dz_t = self.final_pos[2]/traj_length * ds_t # dy(t)

        for i in range(self.num_points):
            # Drone position reference
            #self.reference = self.reference.at[i,0].set(x_t[i])
            #self.reference = self.reference.at[i,1].set(y_t[i])
            self.reference = self.reference.at[i,2].set(z_t[i])

            # Payload position reference
            #self.reference = self.reference.at[i,3].set(?)
            #self.reference = self.reference.at[i,4].set(?)
            #self.reference = self.reference.at[i,5].set(?)

            # Drone velocity reference
            #self.reference = self.reference.at[i,6].set(dx_t[i])
            #self.reference = self.reference.at[i,7].set(dy_t[i])
            self.reference = self.reference.at[i,8].set(dz_t[i])

            # Input reference
            self.reference = self.reference.at[i,-1].set((mass+mass_payload)*gravity) 
        
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
    

    
         
    # Different Segments Trajectory
    def compute_trajectory3(self):
            reference = jnp.zeros((self.num_points, 17),dtype=jnp.float32)
            traj_length = jnp.linalg.norm(self.final_pos)
            t1 = self.dt/2
            num = round(self.num_points/2)
            time_vec = jnp.linspace(0, t1, num=num, dtype=jnp.float32) # time vector
            a0,a1,a2,a3,a4,a5 = self.poly5(t0=0, tf=t1, q0=0, qf=traj_length, dq0=0, dqf=0, ddq0=0, ddqf=0)
            # s: path variable
            s_t = a0 + a1*time_vec + a2*time_vec**2 + a3*time_vec**3 + a4*time_vec**4 + a5*time_vec**5 # s(t)
            ds_t = a1 + 2*a2*time_vec + 3*a3*time_vec**2 + 4*a4*time_vec**3 + 5*a5*time_vec**4 # ds(t)
            # x
            x_t = self.final_pos[0]/traj_length * s_t # x(t)
            dx_t = self.final_pos[0]/traj_length * ds_t # dx(t)
            # y
            y_t = self.final_pos[1]/traj_length * s_t # y(t)
            dy_t = self.final_pos[1]/traj_length * ds_t # dy(t)

            z_t = self.final_pos[2]/traj_length * s_t # y(t)
            dz_t = self.final_pos[2]/traj_length * ds_t # dy(t)
            # Assign variables to state
            for i in range(num):
                self.reference = self.reference.at[i,0].set(x_t[i])
                self.reference = self.reference.at[i,1].set(y_t[i])
                self.reference = self.reference.at[i,2].set(z_t[i])
                self.reference = self.reference.at[i,3].set(1)
                self.reference = self.reference.at[i,7].set(dx_t[i])
                self.reference = self.reference.at[i,8].set(dy_t[i])
                self.reference = self.reference.at[i,9].set(dz_t[i])

                self.reference = self.reference.at[i,-4].set(9.8*0.027) 
            
            # SEGMENT 2
            init_pos2 = self.final_pos
            final_pos2 = jnp.array([1,-0.5,0.7]) # desired final coordinates xy
            traj_length = jnp.linalg.norm(final_pos2-init_pos2)
            time_vec = jnp.linspace(0, t1, num=num, dtype=jnp.float32) # time vector
            a0,a1,a2,a3,a4,a5 = self.poly5(t0=0, tf=t1, q0=0, qf=traj_length, dq0=0, dqf=0, ddq0=0, ddqf=0)
            # s: path variable
            s_t = a0 + a1*time_vec + a2*time_vec**2 + a3*time_vec**3 + a4*time_vec**4 + a5*time_vec**5 # s(t)
            ds_t = a1 + 2*a2*time_vec + 3*a3*time_vec**2 + 4*a4*time_vec**3 + 5*a5*time_vec**4 # ds(t)
            # x
            x_t = init_pos2[0] + (final_pos2[0]-init_pos2[0])/traj_length * s_t # x(t)
            dx_t = (final_pos2[0]-init_pos2[0])/traj_length * ds_t # dx(t)
            # y
            y_t = init_pos2[1] + (final_pos2[1]-init_pos2[1])/traj_length * s_t # y(t)
            dy_t = (final_pos2[1]-init_pos2[1])/traj_length * ds_t # dy(t)

            z_t = init_pos2[2] + (final_pos2[2]-init_pos2[2])/traj_length * s_t # y(t)
            dz_t = (final_pos2[2]-init_pos2[2])/traj_length * ds_t # dy(t)


            # Assign variables to state
            for i in range(num+1):
                self.reference = self.reference.at[i+num,0].set(x_t[i])
                self.reference = self.reference.at[i+num,1].set(y_t[i])
                self.reference = self.reference.at[i+num,2].set(z_t[i])
                self.reference = self.reference.at[i+num,3].set(1)
                self.reference = self.reference.at[i+num,7].set(dx_t[i])
                self.reference = self.reference.at[i+num,8].set(dy_t[i])
                self.reference = self.reference.at[i+num,9].set(dz_t[i])

                self.reference = self.reference.at[i+num,-4].set(9.8*0.027) 

            return reference

    # Square trajectory
    def compute_trajectory4(self):
        # SEGMENT 1
            # init_point = 0
            final_point = jnp.array([0.5,0.5,0.5]) # desired final coordinates xy (assuming starting point=(x=0,y=0))
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
                self.reference = self.reference.at[i,0].set(x_t[i])
                self.reference = self.reference.at[i,1].set(y_t[i])
                self.reference = self.reference.at[i,2].set(z_t[i])
                self.reference = self.reference.at[i,3].set(1)
                self.reference = self.reference.at[i,7].set(dx_t[i])
                self.reference = self.reference.at[i,8].set(dy_t[i])
                self.reference = self.reference.at[i,9].set(dz_t[i])

                self.reference = self.reference.at[i,-4].set(9.8*0.027) 

            # SEGMENT 2
            init_point = final_point
            final_point = jnp.array([0.5,-0.5,1.0]) # desired final coordinates xy
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
                self.reference = self.reference.at[i+num,0].set(x_t[i])
                self.reference = self.reference.at[i+num,1].set(y_t[i])
                self.reference = self.reference.at[i+num,2].set(z_t[i])
                self.reference = self.reference.at[i+num,3].set(1)
                self.reference = self.reference.at[i+num,7].set(dx_t[i])
                self.reference = self.reference.at[i+num,8].set(dy_t[i])
                self.reference = self.reference.at[i+num,9].set(dz_t[i])

                self.reference = self.reference.at[i+num,-4].set(9.8*0.027) 

            # SEGMENT 3
            init_point = final_point
            final_point = jnp.array([-0.5,-0.5,0.5]) # desired final coordinates xy
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
                self.reference = self.reference.at[i+2*num,0].set(x_t[i])
                self.reference = self.reference.at[i+2*num,1].set(y_t[i])
                self.reference = self.reference.at[i+2*num,2].set(z_t[i])
                self.reference = self.reference.at[i+2*num,3].set(1)
                self.reference = self.reference.at[i+2*num,7].set(dx_t[i])
                self.reference = self.reference.at[i+2*num,8].set(dy_t[i])
                self.reference = self.reference.at[i+2*num,9].set(dz_t[i])

                self.reference = self.reference.at[i+2*num,-4].set(9.8*0.027) 
            # SEGMENT 4
            init_point = final_point
            final_point = jnp.array([-0.5,0.5,1.0]) # desired final coordinates xy
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
                self.reference = self.reference.at[i+3*num,0].set(x_t[i])
                self.reference = self.reference.at[i+3*num,1].set(y_t[i])
                self.reference = self.reference.at[i+3*num,2].set(z_t[i])
                self.reference = self.reference.at[i+3*num,3].set(1)
                self.reference = self.reference.at[i+3*num,7].set(dx_t[i])
                self.reference = self.reference.at[i+3*num,8].set(dy_t[i])
                self.reference = self.reference.at[i+3*num,9].set(dz_t[i])

                self.reference = self.reference.at[i+3*num,-4].set(9.8*0.027) 
            
            # SEGMENT 5
            init_point = final_point
            final_point = jnp.array([0.0,0.0,0.5]) # desired final coordinates xy
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
                self.reference = self.reference.at[i+4*num+1,0].set(x_t[i])
                self.reference = self.reference.at[i+4*num+1,1].set(y_t[i])
                self.reference = self.reference.at[i+4*num+1,2].set(z_t[i])
                self.reference = self.reference.at[i+4*num+1,3].set(1)
                self.reference = self.reference.at[i+4*num+1,7].set(dx_t[i])
                self.reference = self.reference.at[i+4*num+1,8].set(dy_t[i])
                self.reference = self.reference.at[i+4*num+1,9].set(dz_t[i])

                self.reference = self.reference.at[i+4*num+1,-4].set(9.8*0.027) 
            return self.reference
    

    def compute_trajectory5(self):
            #final_point = jnp.array([0.5,0.5,0.5]) # desired final coordinates xy (assuming starting point=(x=0,y=0))
            traj_length = jnp.linalg.norm(self.final_pos)
            radius = 2.0 # [m]
            x0_center = -2 # [m]
            y0_center = 0 # [m]
            angle_max = 2*jnp.pi
            time_vec = jnp.linspace(0, 6, num=self.num_points, dtype=jnp.float32) # time vector
            # s: path variable
            
            a0,a1,a2,a3,a4,a5 = self.poly5(t0=0, tf=2, q0=0, qf=angle_max, dq0=0, dqf=0, ddq0=0, ddqf=0)
            s_t = a0 + a1*time_vec + a2*time_vec**2 + a3*time_vec**3 + a4*time_vec**4 + a5*time_vec**5 # s(t)
            ds_t = a1 + 2*a2*time_vec + 3*a3*time_vec**2 + 4*a4*time_vec**3 + 5*a5*time_vec**4 # ds(t)
            theta_t = a0 + a1*time_vec + a2*time_vec**2 + a3*time_vec**3 + a4*time_vec**4 + a5*time_vec**5 # theta(t)
            dtheta_t = a1 + 2*a2*time_vec + 3*a3*time_vec**2 + 4*a4*time_vec**3 + 5*a5*time_vec**4 # dtheta(t)
            x_t = x0_center + radius*jnp.cos(theta_t) # x(t)
            y_t = y0_center + radius*jnp.sin(theta_t) # y(t)
            dx_t = -radius*dtheta_t*jnp.sin(theta_t) # dx(t)
            dy_t = radius*dtheta_t*jnp.cos(theta_t) # dy(t)
            z_t = self.final_pos[2]/traj_length * s_t # z(t)
            dz_t = self.final_pos[2]/traj_length * ds_t # dz(t)
            #w_dot = angle_max/self.Tsim_s # constant angular velocity
            # Assign variables to state
            for i in range(self.num_points):
                self.reference = self.reference.at[i,0].set(x_t[i])
                self.reference = self.reference.at[i,1].set(y_t[i])
                self.reference = self.reference.at[i,2].set(z_t[i])
                self.reference = self.reference.at[i,3].set(1)
                self.reference = self.reference.at[i,7].set(dx_t[i])
                self.reference = self.reference.at[i,8].set(dy_t[i])
                self.reference = self.reference.at[i,9].set(dz_t[i])
                 # Attitude (yaw)
                quat = geom.euler_to_quaternion(0,0,theta_t[i]) # 
                self.reference = self.reference.at[i,3:7].set(quat)
                self.reference = self.reference.at[i,-4].set(9.8*0.027) 
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
    