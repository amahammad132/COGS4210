import numpy as np

def approximate_pi(num_points):
    """
    Approximate the value of pi using Monte Carlo simulation.
    
    Parameters:
    num_points (int): The number of random points to generate
    
    Returns:
    float: The approximated value of pi
    """
    
    #random x and y coordinates generated into arrays
    x = np.random.uniform(-1, 1, num_points)
    y = np.random.uniform(-1, 1, num_points)
    
    distances_squared = x**2 + y**2
    
    # Count how many points fall inside the circle
    points_inside_circle = np.sum(distances_squared <= 1)
    
    
    pi_approximation = 4 * points_inside_circle / num_points
    
    return pi_approximation


def test():
    """
    Test the Monte Carlo approximation with different numbers of points.
    """
    print("Monte Carlo Approximation of pi")
    print("=" * 50)
    print(f"Actual value of pi: {np.pi:.10f}\n")
    
    # Test with increasing numbers of points
    test_cases = [100, 1000, 10000, 100000, 1000000]
    
    for n in test_cases:
        pi_approx = approximate_pi(n)
        error = abs(pi_approx - np.pi)
        error_percent = (error / np.pi) * 100
        
        print(f"Number of points: {n:>10,}")
        print(f"Approximated pi:   {pi_approx:.10f}")
        print(f"Error:            {error:.10f} ({error_percent:.4f}%)")
        print("-" * 50)


if __name__ == "__main__":
    test()