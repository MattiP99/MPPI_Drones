import jax.numpy as jnp
import sys
sys.path.append('/home/mpiras/MPPI/sbmpc2/sbmpc')
import sbmpc.utils.geometry as geom

import jax.numpy as jnp
import sys
sys.path.append('/home/mpiras/MPPI/sbmpc2/sbmpc')
import sbmpc.utils.geometry as geom

mass = 0.027
gravity = 9.81

class Trapeizoidal_Trajectory:
    def __init__(self, initial_pos, final_pos, dt, num_points):
        self.initial_pos = initial_pos
        self.final_pos = final_pos
        self.num_points = num_points
        self.dt=dt
        self.reference = jnp.zeros((self.num_points, 17),dtype=jnp.float32)

    def compute_trajectory(self):
        # Calculate the maximum velocity and acceleration
        max_velocity = 0.3  # You can adjust this value
        max_acceleration = 1.0  # You can adjust this value

        # Calculate the time to increase or decrease the velocity
        time_to_change_velocity = max_velocity / max_acceleration

        # Calculate the distance between initial and final positions
        distance = jnp.sqrt(jnp.sum((self.final_pos - self.initial_pos)**2))

        # Calculate the time to reach the final position
        total_time = distance / max_velocity

        # Calculate the time interval between each point
        time_interval = total_time / (self.num_points - 1)

        # Initialize the trajectory array
        trajectory = []

        # Compute the trajectory using the trapezoidal method
        for i in range(self.num_points):
            t = i * time_interval

            if t < time_to_change_velocity:
                # Acceleration phase
                pos = self.initial_pos + 0.5 * max_acceleration * t**2
            elif t < (total_time - time_to_change_velocity):
                # Constant velocity phase
                pos = self.initial_pos + max_velocity * (t - 0.5 * time_to_change_velocity)
            else:
                # Deceleration phase
                pos = self.final_pos - 0.5 * max_acceleration * (total_time - t)**2

            trajectory.append(pos)

        # Adjust the trajectory to have 526 rows
        while len(trajectory) < 526:
            trajectory.append(self.final_pos)

        return jnp.array(trajectory)
    
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
    def compute_trajectory2(self):
        
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
            self.reference = self.reference.at[i,0].set(x_t[i])
            self.reference = self.reference.at[i,1].set(y_t[i])
            self.reference = self.reference.at[i,2].set(z_t[i])
            self.reference = self.reference.at[i,3].set(1)
            self.reference = self.reference.at[i,7].set(dx_t[i])
            self.reference = self.reference.at[i,8].set(dy_t[i])
            self.reference = self.reference.at[i,9].set(dz_t[i])

            self.reference = self.reference.at[i,-4:].set(jnp.array([mass*gravity, 0.,0.,0.]))
        
        return self.reference
    
    ################### DIFFERENT SEGMENTS TRAJECTORY ############################
    def compute_trajectory3(self):
            #reference = jnp.zeros((self.num_points, 19),dtype=jnp.float32)
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

                self.reference = self.reference.at[i,-4:].set(jnp.array([mass*gravity, 0.,0.,0.]))
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
                self.reference = self.reference.at[i+num,-4:].set(jnp.array([mass*gravity, 0.,0.,0.]))

            return self.reference

    ####################################  SQUARE TRAJECTORY ########################################
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

                self.reference = self.reference.at[i,-6:].set(jnp.array([mass*gravity, 0.,0.,0.]))
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
                self.reference = self.reference.at[i+num,-4:].set(jnp.array([mass*gravity, 0.,0.,0.]))

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
                self.reference = self.reference.at[i+2*num,-4:].set(jnp.array([mass*gravity, 0.,0.,0.]))

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
                self.reference = self.reference.at[i+3*num,-4:].set(jnp.array([mass*gravity, 0.,0.,0.]))

            
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
                self.reference = self.reference.at[i+4*num+1,-4:].set(jnp.array([mass*gravity, 0.,0.,0.]))

            return self.reference
    

    ### Circle trajectory
    
    def compute_trajectory5(self):
            
            # SEGMENT 1 take off
            final_point = jnp.array([0.0,0.0,0.5]) # desired final coordinates xy (assuming starting point=(x=0,y=0))
            traj_length = jnp.linalg.norm(final_point)
            t1 = 2
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

                self.reference = self.reference.at[i,-4:].set(jnp.array([mass*gravity, 0.,0.,0.]))

            #final_point = jnp.array([0.0,0.0,0.5]) # desired final coordinates xy (assuming starting point=(x=0,y=0))
            #traj_length = jnp.linalg.norm(final_point)
            radius = 2.0 # [m]
            x0_center = final_point[0] # [m]
            y0_center = final_point[1] # [m]
            angle_max = 2*jnp.pi
            time_vec = jnp.linspace(0, self.dt - self.dt/5, num=self.num_points, dtype=jnp.float32) # time vector
            # s: path variable
            
            a0,a1,a2,a3,a4,a5 = self.poly5(t0=0, tf=self.dt - self.dt/5, q0=0, qf=angle_max, dq0=0, dqf=0, ddq0=0, ddqf=0)
            #s_t = a0 + a1*time_vec + a2*time_vec**2 + a3*time_vec**3 + a4*time_vec**4 + a5*time_vec**5 # s(t)
            #ds_t = a1 + 2*a2*time_vec + 3*a3*time_vec**2 + 4*a4*time_vec**3 + 5*a5*time_vec**4 # ds(t)
            theta_t = a0 + a1*time_vec + a2*time_vec**2 + a3*time_vec**3 + a4*time_vec**4 + a5*time_vec**5 # theta(t)
            dtheta_t = a1 + 2*a2*time_vec + 3*a3*time_vec**2 + 4*a4*time_vec**3 + 5*a5*time_vec**4 # dtheta(t)
            x_t = x0_center + radius*jnp.cos(theta_t) # x(t)
            y_t = y0_center + radius*jnp.sin(theta_t) # y(t)
            dx_t = -radius*dtheta_t*jnp.sin(theta_t) # dx(t)
            dy_t = radius*dtheta_t*jnp.cos(theta_t) # dy(t)
            z_t = jnp.tile(final_point[2], (self.num_points))
            #self.final_pos[2]/traj_length * s_t # z(t)
            #dz_t = self.final_pos[2]/traj_length * ds_t # dz(t)
            #w_dot = angle_max/self.Tsim_s # constant angular velocity
            # Assign variables to state
            for i in range(self.num_points):
                self.reference = self.reference.at[i,0].set(x_t[i])
                self.reference = self.reference.at[i,1].set(y_t[i])
                self.reference = self.reference.at[i,2].set(z_t[i])
                self.reference = self.reference.at[i,3].set(1)
                self.reference = self.reference.at[i,7].set(dx_t[i])
                self.reference = self.reference.at[i,8].set(dy_t[i])
                #self.reference = self.reference.at[i,9].set(dz_t[i])
                 # Attitude (yaw)
                quat = geom.euler_to_quaternion(0,0,theta_t[i]) # 
                self.reference = self.reference.at[i,3:7].set(quat)
                self.reference = self.reference.at[i,-4:].set(jnp.array([mass*gravity, 0.,0.,0.]))
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