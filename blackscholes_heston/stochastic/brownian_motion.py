"""
Brownian Motion and Stochastic Process Generators.

This module provides functions for generating various types of stochastic
processes commonly used in financial modeling, including random walks,
Brownian motion, correlated Brownian motion, and geometric Brownian motion.
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_random_walk(n_steps, up_probability=0.5, seed=None):
    """
    Generate a random walk with specified number of steps and up probability.
    
    Parameters
    ----------
    n_steps : int
        Number of steps in the random walk
    up_probability : float, optional
        Probability of moving up (default is 0.5)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    list
        List of positions in the random walk
    """
    if seed is not None:
        np.random.seed(seed)
    
    walk = []
    position = 0
    
    for i in range(n_steps):
        walk.append(position)
        random_value = np.random.random()
        
        if random_value < up_probability:
            position -= 1
        else:
            position += 1
    
    return walk


def generate_brownian_motion(time_horizon, n_steps, seed=None):
    """
    Generate a Brownian motion path on [0, T] using N steps.
    
    Parameters
    ----------
    time_horizon : float
        Total time horizon T
    n_steps : int
        Number of time steps
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Array of Brownian motion values
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = time_horizon / n_steps
    increments = np.random.normal(0, np.sqrt(dt), n_steps)
    brownian_path = np.cumsum(increments)
    
    return brownian_path


def generate_correlated_brownian_motion(brownian_motion_1, correlation, 
                                       time_horizon, n_steps, seed=None):
    """
    Generate a correlated Brownian motion path.
    
    Parameters
    ----------
    brownian_motion_1 : array_like
        First Brownian motion path to correlate with
    correlation : float
        Correlation coefficient between -1 and 1
    time_horizon : float
        Total time horizon T
    n_steps : int
        Number of time steps
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    list
        Correlated Brownian motion path
    """
    if seed is not None:
        np.random.seed(seed)
    
    if not -1 <= correlation <= 1:
        raise ValueError("Correlation must be between -1 and 1")
    
    brownian_motion_2 = generate_brownian_motion(time_horizon, n_steps)
    correlated_path = []
    
    for i in range(n_steps):
        correlated_value = (correlation * brownian_motion_1[i] + 
                           np.sqrt(1 - correlation ** 2) * brownian_motion_2[i])
        correlated_path.append(correlated_value)
    
    return correlated_path


def generate_brownian_motion_with_drift(time_horizon, drift, volatility, 
                                       time_step, seed=None):
    """
    Generate a Brownian motion with drift path on [0, T].
    
    Parameters
    ----------
    time_horizon : float
        Total time horizon T
    drift : float
        Drift parameter (mu)
    volatility : float
        Volatility parameter (sigma)
    time_step : float
        Time step size (dt)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Array of Brownian motion with drift values
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_steps = int(round(time_horizon / time_step))
    time_grid = np.linspace(0, time_horizon, n_steps)
    
    # Generate increments with drift
    increments = np.random.normal(
        drift * time_step, 
        volatility * np.sqrt(time_step), 
        n_steps - 1
    )
    
    # Insert initial value of 0
    increments = np.insert(increments, 0, 0)
    brownian_path = np.cumsum(increments)
    
    return brownian_path


def generate_geometric_brownian_motion(initial_value, drift, volatility, 
                                     time_horizon, time_step, seed=None,
                                     plot=False):
    """
    Generate a geometric Brownian motion path on [0, T].
    
    This follows the stochastic differential equation:
    dS_t = mu * S_t * dt + sigma * S_t * dW_t
    
    Parameters
    ----------
    initial_value : float
        Initial value S_0
    drift : float
        Drift parameter (mu)
    volatility : float
        Volatility parameter (sigma)
    time_horizon : float
        Total time horizon T
    time_step : float
        Time step size (dt)
    seed : int, optional
        Random seed for reproducibility
    plot : bool, optional
        Whether to plot the path (default is False)
        
    Returns
    -------
    np.ndarray
        Array of geometric Brownian motion values
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_steps = int(round(time_horizon / time_step))
    time_grid = np.linspace(0, time_horizon, n_steps)
    
    # Generate underlying Brownian motion with drift
    log_path = generate_brownian_motion_with_drift(
        time_horizon, drift, volatility, time_step
    )
    
    # Apply exponential transformation
    geometric_path = initial_value * np.exp(log_path)
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(time_grid, geometric_path)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Geometric Brownian Motion')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return geometric_path


def plot_random_walk(n_steps, up_probability=0.5, seed=None):
    """
    Generate and plot a random walk.
    
    Parameters
    ----------
    n_steps : int
        Number of steps in the random walk
    up_probability : float, optional
        Probability of moving up (default is 0.5)
    seed : int, optional
        Random seed for reproducibility
    """
    walk = generate_random_walk(n_steps, up_probability, seed)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(walk)), walk)
    plt.xlim(0, n_steps)
    plt.xlabel("Steps: $n$")
    plt.ylabel("Random Walk: $S_n$")
    plt.title(f"Random Walk with {n_steps} steps")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_brownian_motion(time_horizon, n_steps, seed=None):
    """
    Generate and plot a Brownian motion path.
    
    Parameters
    ----------
    time_horizon : float
        Total time horizon T
    n_steps : int
        Number of time steps
    seed : int, optional
        Random seed for reproducibility
    """
    brownian_path = generate_brownian_motion(time_horizon, n_steps, seed)
    time_grid = np.linspace(0, time_horizon, n_steps)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_grid, brownian_path)
    plt.xlabel('Time')
    plt.ylabel('Brownian Motion')
    plt.title(f'Brownian Motion over [{0}, {time_horizon}]')
    plt.grid(True, alpha=0.3)
    plt.show()


def compare_processes(time_horizon=1, n_steps=1000, seed=42):
    """
    Compare different stochastic processes in a single plot.
    
    Parameters
    ----------
    time_horizon : float, optional
        Total time horizon (default is 1)
    n_steps : int, optional
        Number of time steps (default is 1000)
    seed : int, optional
        Random seed for reproducibility (default is 42)
    """
    time_grid = np.linspace(0, time_horizon, n_steps)
    
    # Generate different processes
    bm = generate_brownian_motion(time_horizon, n_steps, seed)
    bm_drift = generate_brownian_motion_with_drift(
        time_horizon, 0.05, 0.2, time_horizon/n_steps, seed
    )
    gbm = generate_geometric_brownian_motion(
        100, 0.05, 0.2, time_horizon, time_horizon/n_steps, seed
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Brownian Motion
    axes[0, 0].plot(time_grid, bm)
    axes[0, 0].set_title('Standard Brownian Motion')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Brownian Motion with Drift
    axes[0, 1].plot(time_grid, bm_drift)
    axes[0, 1].set_title('Brownian Motion with Drift')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Geometric Brownian Motion
    axes[1, 0].plot(time_grid, gbm)
    axes[1, 0].set_title('Geometric Brownian Motion')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Random Walk
    rw = generate_random_walk(100, seed=seed)
    axes[1, 1].plot(range(len(rw)), rw)
    axes[1, 1].set_title('Random Walk')
    axes[1, 1].set_xlabel('Steps')
    axes[1, 1].set_ylabel('Position')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function for testing and demonstration."""
    plt.style.use('ggplot')
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("Generating and comparing stochastic processes...")
    compare_processes()


if __name__ == "__main__":
    main()


# Legacy function aliases for backward compatibility
def RandomWalk(n, p):
    """Legacy alias for generate_random_walk."""
    return generate_random_walk(n, p)


def BrownianMotion(T, N):
    """Legacy alias for generate_brownian_motion."""
    return generate_brownian_motion(T, N)


def CorBrownianMotion(B1, rho, T, N):
    """Legacy alias for generate_correlated_brownian_motion."""
    return generate_correlated_brownian_motion(B1, rho, T, N)


def BrownianMotionDrift(T, mu, sigma, dt):
    """Legacy alias for generate_brownian_motion_with_drift."""
    return generate_brownian_motion_with_drift(T, mu, sigma, dt)


def GeometricBrownianMotion(S0, mu, sigma, T, dt):
    """Legacy alias for generate_geometric_brownian_motion."""
    return generate_geometric_brownian_motion(S0, mu, sigma, T, dt, plot=True)
