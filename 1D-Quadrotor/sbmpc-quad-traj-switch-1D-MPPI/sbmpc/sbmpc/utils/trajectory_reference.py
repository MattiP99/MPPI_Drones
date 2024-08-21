import numpy as np
import matplotlib.pyplot as plt
#from utils_operations import euler_to_quaternion
from scipy.signal import chirp
#from utils_extra import measure_time # decorator to measure execution time of a function
#from utils_operations import quaternion_to_euler_angle_vectorized2

#########################################################################################
class Trajectory:
    def __init__(self, nx, Tsim_s, FREQ, Tf, INIT_HEIGHT, N_horizon, traj_type='none', NP_DTYPE='float64', plot=False):
        '''Initialize class variables, empty trajectory, compute and check downsample factor'''
        # nx = 3 + 4 + 3 + 3
        self.nx = nx
        self.Tsim_s = Tsim_s
        self.FREQ = FREQ
        self.Tf = Tf
        self.INIT_HEIGHT = INIT_HEIGHT
        self.N_horizon = N_horizon
        self.traj_type = traj_type
        print("Initializing Trajectory of type: ", self.traj_type)
        self.plot = plot
        self.NP_DTYPE = NP_DTYPE
        #self.n_samples = round((Tsim_s+Tf) * FREQ) # total number of samples in the trajectory (including the prediction horizon)
        self.n_samples_real = round(Tsim_s * FREQ) # total number of samples in the trajectory
        self.pred_hor_traj_samples = round(self.Tf * self.FREQ) # number of samples in the prediction horizon
        self.n_samples_tot = self.n_samples_real # this number is updated by the function which modify ref_traj
        # Initialize empty trajectory
        self.ref_pos = np.zeros((self.n_samples_real,3), dtype=NP_DTYPE)
        self.ref_att = np.zeros((self.n_samples_real,4), dtype=NP_DTYPE)
        self.ref_vel = np.zeros((self.n_samples_real,3), dtype=NP_DTYPE)
        self.ref_w = np.zeros((self.n_samples_real,3), dtype=NP_DTYPE)
        self.ref_att[:,0] = 1.0 # quat representing null orientation = [1,0,0,0]
        self.ref_pos[:,2] = INIT_HEIGHT # [m] set z_ref to INIT_HEIGHT
        # Horizontally stack the references
        self.ref_traj = np.hstack((self.ref_pos, self.ref_att, self.ref_vel, self.ref_w))
        # Check that reference and prediction horizon frequencies are multiples
        horizon_freq = self.N_horizon/self.Tf
        if (self.FREQ % horizon_freq) != 0:
            raise ValueError("The frequency of the reference must be a multiple of the prediction horizon frequency.")
        else:
            self.downsample_factor = int(self.FREQ/horizon_freq)
            print("Downsampling reference trajectory with a factor:", self.downsample_factor)


    def compute_ref_trajectory(self):
        ''' Reference definition according to sim_parameters.yaml'''
        print("Creating reference signal with", self.n_samples_real, "samples")

        if self.traj_type == 'none':
            pass
            # Hovering at INIT_HEIGHT, with desired yaw
            #self.ref_att[:,:] = np.array([0,0,0,1]) # quat representing pi = [0,0,0,1]
            #self.ref_att[:,:] = np.array([0.9238795, 0, 0, 0.3826834]) # quat representing pi/4


        elif self.traj_type == 'step':
            self.ref_pos[round(self.n_samples_real/2):,0] = 1.0 # set x_ref to 1

        elif self.traj_type == 'sstep':
            self.ref_pos[:self.n_samples_real,0] = self.cosine_interpolation(t_min=2, t_max=3, y_min=0, y_max=1.0)

        elif self.traj_type == 'circle':
            radius = 2.0 # [m]
            x0_center = -2 # [m]
            y0_center = 0 # [m]
            angle_max = 2*np.pi
            time_vec = np.linspace(0, self.Tsim_s, num=self.n_samples_real, dtype=self.NP_DTYPE) # time vector
            a0,a1,a2,a3,a4,a5 = self.poly5(t0=0, tf=self.Tsim_s, q0=0, qf=angle_max, dq0=0, dqf=0, ddq0=0, ddqf=0)
            theta_t = a0 + a1*time_vec + a2*time_vec**2 + a3*time_vec**3 + a4*time_vec**4 + a5*time_vec**5 # theta(t)
            dtheta_t = a1 + 2*a2*time_vec + 3*a3*time_vec**2 + 4*a4*time_vec**3 + 5*a5*time_vec**4 # dtheta(t)
            x = x0_center + radius*np.cos(theta_t) # x(t)
            y = y0_center + radius*np.sin(theta_t) # y(t)
            x_dot = -radius*dtheta_t*np.sin(theta_t) # dx(t)
            y_dot = radius*dtheta_t*np.cos(theta_t) # dy(t)
            #w_dot = angle_max/self.Tsim_s # constant angular velocity
            # Assign variables to state
            self.ref_pos[:self.n_samples_real,0] = x
            self.ref_pos[:self.n_samples_real,1] = y
            self.ref_vel[:self.n_samples_real,0] = x_dot
            self.ref_vel[:self.n_samples_real,1] = y_dot
            #self.ref_w[:self.n_samples_real,2] = w_dot
            # Attitude (yaw)
            for i in range(self.n_samples_real):
                quat = euler_to_quaternion(0,0,theta_t[i]) # 
                self.ref_att[i,:] = quat

        elif self.traj_type == 'yaw':
            # yaw rotation 0 to angle_max (no coming back to zero yaw)
            angle_max = 2*np.pi
            yaw = np.linspace(0, angle_max, num=self.n_samples_real, dtype=self.NP_DTYPE) # generate reference yaws exluding the last sample
            w_dot = angle_max/self.Tsim_s # constant angular velocity
            self.ref_w[:self.n_samples_real,2] = w_dot
            for i in range(self.n_samples_real):
                # TODO: understand if this could be done more efficiently (also in the circle trajectory)
                quat = euler_to_quaternion(0,0,yaw[i])
                self.ref_att[i,:] = quat
                
        elif self.traj_type == 'yaw_backforth':
            # yaw rotation 0 to angle_max (no coming back to zero yaw)
            angle_max = np.pi/2
            num = round(self.n_samples_real/2)
            yaw = np.linspace(0, angle_max, num=num, dtype=self.NP_DTYPE) # generate reference yaws exluding the last sample
            #w_dot = angle_max/self.Tsim_s # constant angular velocity
            #self.ref_w[:self.n_samples_real,2] = w_dot
            # FIRST SEGMENT
            for i in range(num):
                quat = euler_to_quaternion(0,0,yaw[i])
                self.ref_att[i,:] = quat
            yaw = np.linspace(angle_max, 0, num=num, dtype=self.NP_DTYPE) # generate reference yaws exluding the last sample
            # SECOND SEGMENT
            for i in range(num):
                quat = euler_to_quaternion(0,0,yaw[i])
                self.ref_att[num+i,:] = quat

        elif self.traj_type == 'chirp':
            fmax = 0.8 # Hz # 0.8 ok with 20 seconds
            time_vec = np.linspace(0, int(self.Tsim_s/2), num=int(self.n_samples_real/2), dtype=self.NP_DTYPE) # exlude the last sample
            x_traj1 = -1.0 + chirp(time_vec, f0=0, t1=int(self.Tsim_s/2), f1=fmax, method='linear', phi=0)
            x_traj2 = -1.0 - chirp(time_vec, f0=fmax, t1=int(self.Tsim_s/2), f1=0, method='linear', phi=180)
            # compute derivatives (used numpy gradient for convenience)
            dx = time_vec[1]-time_vec[0]
            dx_traj1 = np.gradient(x_traj1, dx)
            dx_traj2 = np.gradient(x_traj2, dx)
            # assign values
            self.ref_pos[:round(self.n_samples_real/2),0] = x_traj1 # first segment x
            self.ref_pos[round(self.n_samples_real/2):self.n_samples_real,0] = x_traj2 # second segment x
            self.ref_vel[:round(self.n_samples_real/2),0] = dx_traj1 # first segment dx
            self.ref_vel[round(self.n_samples_real/2):self.n_samples_real,0] = dx_traj2 # second segment dx

        elif self.traj_type == 'line':
            final_point = np.array([4,4]) # desired final coordinates xy (assuming starting point=(0,0))
            traj_length = np.linalg.norm(final_point)
            time_vec = np.linspace(0, self.Tsim_s, num=self.n_samples_real, dtype=self.NP_DTYPE) # time vector
            a0,a1,a2,a3,a4,a5 = self.poly5(t0=0, tf=self.Tsim_s, q0=0, qf=traj_length, dq0=0, dqf=0, ddq0=0, ddqf=0)
            # s: path variable
            s_t = a0 + a1*time_vec + a2*time_vec**2 + a3*time_vec**3 + a4*time_vec**4 + a5*time_vec**5 # s(t)
            ds_t = a1 + 2*a2*time_vec + 3*a3*time_vec**2 + 4*a4*time_vec**3 + 5*a5*time_vec**4 # ds(t)
            # x
            x_t = final_point[0]/traj_length * s_t # x(t)
            dx_t = final_point[0]/traj_length * ds_t # dx(t)
            # y
            y_t = final_point[1]/traj_length * s_t # y(t)
            dy_t = final_point[1]/traj_length * ds_t # dy(t)
            # Assign variables to state
            self.ref_pos[:self.n_samples_real,0] = x_t
            self.ref_pos[:self.n_samples_real,1] = y_t
            self.ref_vel[:self.n_samples_real,0] = dx_t
            self.ref_vel[:self.n_samples_real,1] = dy_t

        elif self.traj_type == 'linereal':
            # SEGMENT 1
            final_point = np.array([0.5,0.5]) # desired final coordinates xy (assuming starting point=(x=0,y=0))
            traj_length = np.linalg.norm(final_point)
            t1 = self.Tsim_s/2
            num = round(self.n_samples_real/2)
            time_vec = np.linspace(0, t1, num=num, dtype=self.NP_DTYPE) # time vector
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
            # Assign variables to state
            self.ref_pos[:num,0] = x_t
            self.ref_pos[:num,1] = y_t
            self.ref_vel[:num,0] = dx_t
            self.ref_vel[:num,1] = dy_t
            # SEGMENT 2
            init_point = np.array([0.5,0.5])
            final_point = np.array([0,0]) # desired final coordinates xy
            traj_length = np.linalg.norm(final_point-init_point)
            time_vec = np.linspace(0, t1, num=num, dtype=self.NP_DTYPE) # time vector
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
            self.ref_pos[num:,0] = x_t
            self.ref_pos[num:,1] = y_t
            self.ref_vel[num:,0] = dx_t
            self.ref_vel[num:,1] = dy_t

        elif self.traj_type == 'square':
            # SEGMENT 1
            # init_point = 0
            final_point = np.array([0.5,0.5]) # desired final coordinates xy (assuming starting point=(x=0,y=0))
            traj_length = np.linalg.norm(final_point)
            t1 = self.Tsim_s/5
            num = round(self.n_samples_real/5)
            time_vec = np.linspace(0, t1, num=num, dtype=self.NP_DTYPE) # time vector
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
            # Assign variables to state
            self.ref_pos[:num,0] = x_t
            self.ref_pos[:num,1] = y_t
            self.ref_vel[:num,0] = dx_t
            self.ref_vel[:num,1] = dy_t
            # SEGMENT 2
            init_point = final_point
            final_point = np.array([0.5,-0.5]) # desired final coordinates xy
            traj_length = np.linalg.norm(final_point-init_point)
            time_vec = np.linspace(0, t1, num=num, dtype=self.NP_DTYPE) # time vector
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
            self.ref_pos[num:2*num,0] = x_t
            self.ref_pos[num:2*num,1] = y_t
            self.ref_vel[num:2*num,0] = dx_t
            self.ref_vel[num:2*num,1] = dy_t
            # SEGMENT 3
            init_point = final_point
            final_point = np.array([-0.5,-0.5]) # desired final coordinates xy
            traj_length = np.linalg.norm(final_point-init_point)
            time_vec = np.linspace(0, t1, num=num, dtype=self.NP_DTYPE) # time vector
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
            self.ref_pos[2*num:3*num,0] = x_t
            self.ref_pos[2*num:3*num,1] = y_t
            self.ref_vel[2*num:3*num,0] = dx_t
            self.ref_vel[2*num:3*num,1] = dy_t
            # SEGMENT 4
            init_point = final_point
            final_point = np.array([-0.5,0.5]) # desired final coordinates xy
            traj_length = np.linalg.norm(final_point-init_point)
            time_vec = np.linspace(0, t1, num=num, dtype=self.NP_DTYPE) # time vector
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
            self.ref_pos[3*num:4*num,0] = x_t
            self.ref_pos[3*num:4*num,1] = y_t
            self.ref_vel[3*num:4*num,0] = dx_t
            self.ref_vel[3*num:4*num,1] = dy_t
            # SEGMENT 5
            init_point = final_point
            final_point = np.array([0.0,0.0]) # desired final coordinates xy
            traj_length = np.linalg.norm(final_point-init_point)
            time_vec = np.linspace(0, t1, num=num, dtype=self.NP_DTYPE) # time vector
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
            self.ref_pos[4*num:,0] = x_t
            self.ref_pos[4*num:,1] = y_t
            self.ref_vel[4*num:,0] = dx_t
            self.ref_vel[4*num:,1] = dy_t

            

        elif self.traj_type == '3d':
            # qf should be computed as the arc length (integral of p dot)
            time_vec = np.linspace(0, self.Tsim_s, num=self.n_samples_real, dtype=self.NP_DTYPE) # time vector
            a0,a1,a2,a3,a4,a5 = self.poly5(t0=0, tf=self.Tsim_s, q0=0, qf=10, dq0=0, dqf=0, ddq0=0, ddqf=0)
            # s: path variable
            s_t = a0 + a1*time_vec + a2*time_vec**2 + a3*time_vec**3 + a4*time_vec**4 + a5*time_vec**5 # s(t)
            ds_t = a1 + 2*a2*time_vec + 3*a3*time_vec**2 + 4*a4*time_vec**3 + 5*a5*time_vec**4 # ds(t)
            # x
            x_t = s_t # y(t)
            dx_t = ds_t # dy(t)
            # y
            y_t = np.sin(1.5*s_t) # x(t)
            dy_t = 1.5*np.cos(s_t)*ds_t # dx(t)
            # z
            z_t = 1 + 0.4*s_t
            dz_t = 0.4*ds_t

            # Assign variables to state
            self.ref_pos[:self.n_samples_real,0] = x_t
            self.ref_pos[:self.n_samples_real,1] = y_t
            self.ref_pos[:self.n_samples_real,2] = z_t
            self.ref_vel[:self.n_samples_real,0] = dx_t
            self.ref_vel[:self.n_samples_real,1] = dy_t
            self.ref_vel[:self.n_samples_real,2] = dz_t
        
        elif self.traj_type == 'other':
            pass

        else:
            raise ValueError("Invalid reference type.")
        # stack all the subtrajectories to compose the full ref_traj
        self.ref_traj = np.hstack((self.ref_pos, self.ref_att, self.ref_vel, self.ref_w))


    def add_land_trajectory(self, BASE_HEIGHT, t_land=5.0):
        # Define the trajectory to land from the current position
        # The motion is in z axis only (line trajectory type)
        NN = round(t_land*self.FREQ) # (t_land*self.FREQ) is the number of samples in the land trajectory
        init_pos = self.ref_pos[-1,:] # last pos
        final_pos = np.array([init_pos[0], init_pos[1], BASE_HEIGHT]) # same xy coords, BASE_HEIGHT z
        traj_length = np.linalg.norm(final_pos-init_pos)
        time_vec = np.linspace(0, t_land, num=NN, dtype=self.NP_DTYPE) # time vector
        a0,a1,a2,a3,a4,a5 = self.poly5(t0=0, tf=t_land, q0=0, qf=traj_length, dq0=0, dqf=0, ddq0=0, ddqf=0)
        # s: path variable
        s_t = a0 + a1*time_vec + a2*time_vec**2 + a3*time_vec**3 + a4*time_vec**4 + a5*time_vec**5 # s(t)
        ds_t = a1 + 2*a2*time_vec + 3*a3*time_vec**2 + 4*a4*time_vec**3 + 5*a5*time_vec**4 # ds(t)
        # z
        z_t = self.INIT_HEIGHT + (final_pos[-1]-init_pos[-1])/traj_length * s_t # z(t)
        dz_t = (final_pos[-1]-init_pos[-1])/traj_length * ds_t # dz(t)
        # Assign variables to state
        land_traj_array = np.zeros((NN, self.nx))
        # Fill land_traj_array with the last values of references
        land_traj_array[:,0:2] = self.ref_pos[-1,:2] # pos x y
        land_traj_array[:,3:7] = self.ref_att[-1,:] # att
        land_traj_array[:,7:9] = self.ref_vel[-1,:2] # vel x y z
        # Define trajectory for z
        land_traj_array[:,2] = z_t
        land_traj_array[:,9] = dz_t
        self.ref_traj = np.vstack((self.ref_traj, land_traj_array)) # stack the land trajectory after the ref_traj
        self.n_samples_tot = self.ref_traj.shape[0] # update the total number of samples IMPORTANT
        print("Added landing samples, the trajectory has now shape", self.ref_traj.shape)
        self.Tsim_s += t_land # add landing time to the total traj time


    def define_last_pred_horizon(self):
        pred_hor_traj_array = np.zeros((self.pred_hor_traj_samples, self.nx))
        # Define the reference copying the last value
        pred_hor_traj_array[:,0:3] = self.ref_traj[-1,0:3] # pos x y z
        pred_hor_traj_array[:,3:7] = self.ref_traj[-1,3:7] # att
        pred_hor_traj_array[:,7:10] = self.ref_traj[-1,7:10] # vel x y z
        # maybe we should do the same for the angular velocity, but in some cases (e.g. yaw traj) this may not be good
        # Stack to increase ref_traj
        self.ref_traj = np.vstack((self.ref_traj, pred_hor_traj_array)) # stack the pred hor traj after the ref_traj
        self.n_samples_tot = self.ref_traj.shape[0] # update the total number of samples IMPORTANT
        print("Added last prediction horizon samples, the trajectory has now shape", self.ref_traj.shape)


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
    

    # TODO: implement this function in a more efficient (and Pythonic) way
    def cosine_interpolation(self,t_min,t_max,y_min,y_max,NP_DTYPE='float32'):
        time_vec = np.linspace(0, self.Tsim_s, num=self.n_samples_real, dtype=self.NP_DTYPE) # time vector
        interp = np.full_like(time_vec, 0, dtype=NP_DTYPE)
        for index, value in enumerate(time_vec):
            # print(index,value)
            if value < t_min:
                interp[index] = y_min
            elif value > t_max:
                interp[index] = y_max
            else:
                interp[index] = y_min + 0.5*(y_max-y_min)*(1-np.cos((value-t_min)*np.pi/(t_max-t_min)))
        return interp


    def downsample_future_traj(self, it):
        '''Downsample the reference trajectory to fit the prediction horizon samples, starting from the current iteration'''
        '''ONLINE version'''
        self.ref_traj_MPC = np.zeros((self.N_horizon+1,self.nx), dtype=self.NP_DTYPE)
        # consider ref_traj from current iteration
        ref_traj_it = self.ref_traj[it:round(it+self.FREQ*self.Tf)+1,:] # take ref_traj from current it to (current it + prediction horizon)
        # slice only first dimension (rows = time samples)
        self.ref_traj_MPC = ref_traj_it[::self.downsample_factor, :] # slicing notation start:stop:step
        print("ONLINE downsampling, shape of downsampled array:", self.ref_traj_MPC.shape)


    #@measure_time
    def downsample_offline_future_traj(self):
        '''Downsample the reference trajectory to fit the prediction horizon samples'''
        '''OFFLINE version'''
        arr_size = self.n_samples_tot - self.pred_hor_traj_samples # number of samples - pred horizon samples
        self.ref_traj_MPC_off = np.zeros((arr_size,self.N_horizon+1,self.nx), dtype=self.NP_DTYPE) # for each sample there is a N_horizon+1 x nx array
        # for each sample (excluding prediction horizon) 
        for it in range(arr_size):
            ref_traj_it = self.ref_traj[it:round(it+self.FREQ*self.Tf)+1,:] # take slice of reference traj to downsample
            self.ref_traj_MPC_off[it,:,:] = ref_traj_it[::self.downsample_factor, :]
        print("OFFLINE downsampling, shape of downsampled array:", self.ref_traj_MPC_off.shape)
        

   
