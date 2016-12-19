#!/bin/usr/env python3

import numpy as np
import matplotlib as mtplt
import numba as nb
import matplotlib.pyplot as plt
import nose.tools as n

"""
Implements 4 numerical methods for solving ODEs: Euler's, Heun's, and 2nd 
and 4th order Runge-Kutta.
"""
@nb.jit(["float64[:,:](int64, float64[:])"], nopython=True)
def init_solutions(n, u_0):
    """
    Initializes the solution array

    Parameters:
    -----------
        n: integer
            number of sampling points in mesh
        u_0: array of floats
            vector of initial conditions, the number of components should equal the number of ODEs in the system

    Returns:
    --------
        array, first column initialized to initial conditions
            this array will be filled with the integrate function with all the values for the solutions
    """
    u = np.zeros((n, len(u_0)))
    u[0,:] = u_0#Set first column equal to the vector of initial conditions
    return u


#@nb.jit(["float64[:,:](float[:,:],float64,float64,"], nopython=False)
def euler_step(u, delta_t, t, du):
    """
    Implementation of Euler's method for coupled pair of ODEs for a single step

    Parameters:
    -----------
        u: array-like
            vector of initial conditions
        delta_t: float
            time step size
        t: float
            current time
        du: lambda
            vector-valued function for differential equation
    Returns:
    --------
        tuple of floats
            vector of values for the function at the next time step 
    """
    return u + delta_t * du(u, t)

def leapfrog_step(u, u_k_minus1, delta_t, t, du):
    """
    Implementation of the Leapfrog Method approximation for systems of coupled ODEs. IMPORTANT: This method takes 2 initial conditions to solve. The implementation in this module will take only one initial condition and find the second using a single step of Euler's method.

    Parameters:
    -----------
        u: array-like
            vector of initial conditions
        u_k_minus1: array-like
            vector of second initial condition
        delta_t: float
            time step size
        t: float
            current time
        du: lambda
            vector-valued function for differential equation
    Returns:
    --------
        tuple of floats
            vector of values for the function at the next time step
    """
    return u_k_minus1 + 2 * delta_t * du(u, t)

def heun_step(u, delta_t, t, du):
    """
    Implementation of the Heun's Method (Trapezoid) approximation for systems of coupled ODEs

    Parameters:
    -----------
        u: array-like
            vector of initial conditions
        delta_t: float
            time step size
        t: float
            current time
        du: lambda
            vector-valued function for differential equation
    Returns:
    --------
        tuple of floats
            vector of values for the function at the next time step
    """
    u_tilde = u + delta_t * du(u, t)# One estimate using Euler's method, average slope will be used
    return u + delta_t / 2 * (du(u, t) + du(u_tilde, t + delta_t))

def rk2_step(u, delta_t, t, du):
    """
    Implementation of the Runge-Kutta 2nd order approximation for systems of coupled ODEs

    Parameters:
    -----------
        u: array-like
            vector of initial conditions
        delta_t: float
            time step size
        t: float
            current time
        du: lambda
            vector-valued function for differential equation
    Returns:
    --------
        tuple of floats
            vector of values for the function at the next time step
    """
    K1 = delta_t * du(u, t)
    K2 = delta_t * du(u + K1 / 2, t + delta_t / 2)# 2 intermediate approximations
    return u + K2

def rk4_step(u, delta_t, t, du):
    """
    Implementation of the Runge-Kutta 4th order approximation for solving a system of coupled ODEs

    Parameters:
    -----------
        u: array-like
            vector of initial conditions
        delta_t: float
            time step size
        t: float
            current time
        du: lambda
            vector-valued function for differential equation
    Returns:
    --------
        tuple of floats
            vector of values for the function at the next time step
    """
    K1 = delta_t * du(u, t)
    K2 = delta_t * du(u + K1 / 2, t + delta_t / 2)
    K3 = delta_t * du(u + K2 / 2, t + delta_t / 2)
    K4 = delta_t * du(u + K3, t + delta_t)# 4 intermediate approximations
    return u + (K1 + 2 * K2 + 2 * K3 + K4) / 6

#@nb.jit(nopython=True)
def integrate(u_0, a, b, delta_t, du, integrator):
    """
    Parameters:
    -----------
        u_0: array-like
            vector of initial conditions
        a: float
            intial time point
        b: float
            final time point
        detla_t: float
            time interval
        du: array-like
            vector of lambda functions for differential equations
        integrator: string
            numerical method for integration
        u_1: array-like
            vector of second initial condition for use in leapfrog only

    Returns:
        u: array-like
            array of meshes representing solutions
    """
    if integrator == 'rk4':
        step = rk4_step
    elif integrator == 'rk2':
        step = rk2_step
    elif integrator == 'heun':
        step = heun_step
    elif integrator == 'leapfrog':
        step = leapfrog_step
        ###Second Initial Condition!
    #        if i == 1:
    #            u[1] = step(u[0], delta_t, a + delta_t, du)
    #        else:
    #            u[i] = leapfrog_step(u[i - 1], u[i - 2], delta_t, t_current, du)
    elif integrator == 'euler':
        step = euler_step
    @nb.jit(nopython=True)
    def for_loop():
        n = int((b - a) / float(delta_t)) #Number of points in t-mesh
        u = init_solutions(n, u_0)
        for i in range(1, n):
            t_current = a + delta_t * i
            u[i] = step(u[i - 1], delta_t, t_current, du)  
        return u
    return for_loop()

def test_integrate_euler():
    u_prime = lambda u, t: np.array((u[1], -u[0]))
    test = integrate((1, 0), 0, 1, 0.001, u_prime, 'euler')
    tvals = np.linspace(0, 1, 1000)
    case = np.transpose(np.array((np.cos(tvals),-1 * np.sin(tvals))))
    print(test,case)
    assert np.allclose(test,case, rtol=1E-1)

def test_integrate_leapfrog(): 
    u_prime = lambda u, t: np.array((u[1], -u[0]))
    test = integrate((1, 0), 0, 0.1, 0.001, u_prime, 'leapfrog')
    tvals = np.linspace(0, 0.1, 100)
    case = np.transpose(np.array((np.cos(tvals),-1 * np.sin(tvals))))
    print(test,case)
    assert np.allclose(test,case, rtol=1E-1)

def test_integrate_heun(): 
    u_prime = lambda u, t: np.array((u[1], -u[0]))
    test = integrate((1, 0), 0, 0.1, 0.001, u_prime, 'heun')
    tvals = np.linspace(0, 0.1, 100)
    case = np.transpose(np.array((np.cos(tvals),-1 * np.sin(tvals))))
    print(test,case)
    assert np.allclose(test,case, rtol=1E-1)

def test_integrate_rk2(): 
    u_prime = lambda u, t: np.array((u[1], -u[0]))
    test = integrate((1, 0), 0, 0.1, 0.001, u_prime, 'rk2')
    tvals = np.linspace(0, 0.1, 100)
    case = np.transpose(np.array((np.cos(tvals),-1 * np.sin(tvals))))
    print(test,case)
    assert np.allclose(test,case, rtol=1E-1)

def test_integrate_rk4():
    u_prime = lambda u, t: np.array((u[1], -u[0]))
    test = integrate((1, 0), 0, 1, 0.001, u_prime, 'rk4')
    tvals = np.linspace(0, 1, 1000)
    case = np.transpose(np.array((np.cos(tvals),-1 * np.sin(tvals))))
    print(test,case)
    assert np.allclose(test,case, rtol=1E-1)

def plot(u, title, actual_solutions, tvals, solvable = True, scatter = False, animated = False, phi = 0):
    plot = plt.figure(1, figsize=(10, 8))
    if scatter:
        plt.scatter(u[0], u[1])
    else:
        plt.plot(u[0], u[1], 'r', label='u_1(t)')
    if solvable:
        plt.plot(actual_solutions[0], actual_solutions[1], 'r--', label='u_1actual(t)')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1, ncol=4, mode="expand", borderaxespad=0.)
    plt.title(title)
    if animated:
        filename = 'frame' + str(phi) + '.png'
    else:
        filename = title + '.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
    return plot

if __name__ == "__main__":
    import sys
    args = sys.argv
    if len(args) != 5:
        print("Didn't enter correct number of arguments. Need 4.")
    else:
        u_0 = float(args[1])
        a = float(args[2])
        b = float(args[3])
        delta_t = float(args[4])
        main(u_0, a, b, delta_t)
