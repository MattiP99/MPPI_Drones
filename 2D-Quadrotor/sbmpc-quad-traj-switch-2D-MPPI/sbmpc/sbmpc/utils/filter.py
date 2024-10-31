import jax.numpy as jnp
import jax

from jax.scipy.signal import convolve
from math import factorial


class MovingAverage:
    def __init__(self, window_size: int = 3, step_size: int = 1):
        kernel = jnp.zeros(window_size + (window_size - 1) * (step_size - 1))
        for i in range(window_size):
            kernel = kernel.at[i*step_size].set(1)
        kernel = kernel.at[-1].set(1)
        self.kernel = kernel

        self.filter = jax.jit(self._filter)
        self.filter_batch = jax.vmap(self._filter, in_axes=(0, None, None))

    def _filter(self, signal, left_padding, right_padding):
        signal_ = jnp.concatenate((left_padding, signal, right_padding))
        filtered = (convolve(signal_, self.kernel, mode='same') /
                    convolve(jnp.ones_like(signal_), self.kernel, mode='same'))
        #jax.debug.print("{signal}",signal = filtered[left_padding.shape[0]:(left_padding.shape[0] + signal.shape[0])])
        return filtered[left_padding.shape[0]:(left_padding.shape[0] + signal.shape[0])]
    
    
    def savitzky_golay2(self, y,left_padding, right_padding):

        window_size = 5
        order = 2
        deriv=0
        rate=1
        """
        try:
            window_size = jnp.abs(int(window_size))
            order = jnp.abs(int(order))
        except ValueError:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        """
        
        #window_size = jnp.abs(int(window_size))
        #order = jnp.abs(int(order))
        order_range = range(order+1)
        half_window = (window_size - 1) // 2
        # precompute coefficients
        #b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        b = jnp.array([[k**i for i in order_range] for k in range(-half_window, half_window+1)])

        #b = self.create_matrix_with_while_loop(order_range, half_window)
        #jax.debug.print("{b}",b = b)

        #m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        m = jnp.linalg.pinv(b)[deriv] * rate**deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself

        firstvals = y[0] - jnp.abs( y[1:half_window+1][::-1] - y[0] )
        lastvals = y[-1] + jnp.abs(y[-half_window-1:-1][::-1] - y[-1])
        y_ = jnp.concatenate((firstvals, y, lastvals))
        signal_ = convolve( m[::-1], y_, mode='valid')
        
        return signal_
        
        #filtered = (fftconvolve(signal_, self.kernel_SavitskyGolay, mode='same')/fftconvolve(jnp.ones_like(signal_), self.kernel_SavitskyGolay, mode='same'))
        #return filtered[left_padding.shape[0]:(left_padding.shape[0] + signal.shape[0])]
        
        


