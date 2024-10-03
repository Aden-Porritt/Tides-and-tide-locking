import pytide
import numpy as np
import scipy.integrate
import numba as nb

G = 6.67408 * 10**-11

@nb.njit()
def sigmoid(x):
    return (2 / (1 + np.exp(-1000000.0 * x))) - 1

@nb.njit()
def keplers_third_law_fast(a, mass1, mass2):
    return np.sqrt(6.67408 * 10**-11.0 * (mass1 + mass2) / a ** 3)

def keplers_third_law(a, body1, body2):
    return np.sqrt(G * (body1.mass + body2.mass) / a ** 3)

@nb.njit()
def calculated_a_gradient_fast(w, a, Q, k_2, r, mass1, mass2):
    return sigmoid(w - keplers_third_law_fast(a, mass1, mass2)) * (3 * k_2 / Q) * (mass2 / mass1) * (r / a) ** 5 * keplers_third_law_fast(a, mass1, mass2) * a

@nb.njit()
def calculated_w_gradient_fast(w, a, Q, k_2, alpha, r, mass1, mass2):
    return -sigmoid(w - keplers_third_law_fast(a, mass1, mass2)) * (3 * k_2 / (2 * alpha * Q)) * (mass2 ** 2 / (mass1 * (mass1 + mass2))) * (r / a) ** 3 * keplers_third_law_fast(a, mass1, mass2) ** 2

def calculated_a_gradient(w, a, body1, body2):
    return sigmoid(w - keplers_third_law(a, body1, body2)) * (3 * body1.k_2 / (body1.Q)) * (body2.mass / body1.mass) * (body1.radius / a) ** 5 * keplers_third_law(a, body1, body2) * a

def calculated_w_gradient(w, a, body1, body2):
    return -sigmoid(w - keplers_third_law(a, body1, body2)) * (3 * body1.k_2 / (2 * body1.alpha * body1.Q)) * (body2.mass ** 2 / (body1.mass * (body1.mass + body2.mass))) * (body1.radius / a) ** 3 * keplers_third_law(a, body1, body2) ** 2

def calculated_tidal_despin_gradient(t, y, body1, body2):
    w, a = y

    dw = calculated_w_gradient(w, a, body1, body2)

    da = calculated_a_gradient(w, a, body1, body2)

    return np.array([dw, da]) * 365.25 * 24 * 60 * 60

def calculated_tidal_despin_fast(start_a, body1, start_w=0.000174):
    return start_w * start_a ** 6 * body1.Q / (3 * G * body1.k_2 * body1.mass ** 2 * body1.radius ** 3)

@nb.njit()
def calculated_tidal_despin_e_helper(w, a, Q, k_2, alpha, r, mass1, mass2, step):
    start_w = w
    start_a = a
    time = 0
    while w - keplers_third_law_fast(a, mass1, mass2) > (start_w - keplers_third_law_fast(start_a, mass1, mass2)) / 500.0:
        if time > 10.0**100.0:
            break
        dw = calculated_w_gradient_fast(w, a, Q, k_2, alpha, r, mass1, mass2)
        da = calculated_a_gradient_fast(w, a, Q, k_2, r, mass1, mass2)

        dw = dw * 365.25 * 24 * 60 * 60
        da = da * 365.25 * 24 * 60 * 60

        w += dw * step
        a += da * step

        step *= 1.01

        time += step

    return time

def calculated_tidal_despin_e(start_a, start_w, body1, body2, step_size=1.0):
    return calculated_tidal_despin_e_helper(start_w, start_a, body1.Q, body1.k_2, body1.alpha, body1.radius, body1.mass, body2.mass, step_size)


def calculated_tidal_despin(start_a, start_w, body1, body2, steps=10):
    w = start_w
    a = start_a
    t_max = 10
    t_min = 0

    time_data = np.array([])
    w_data = np.array([])

    a_data = np.array([])

    while (w - keplers_third_law(a, body1, body2)) > (start_w - keplers_third_law(start_a, body1, body2)) / 500:
        initial_conditions = [w, a]
        sol = scipy.integrate.solve_ivp(calculated_tidal_despin_gradient, [t_min, t_max], initial_conditions, t_eval=np.linspace(t_min, t_max, steps), args=(body1, body2,), method='RK45')
        time_data = np.append(time_data, sol.t)
        w_data = np.append(w_data, sol.y[0])
        a_data = np.append(a_data, sol.y[1])

        t_min = time_data[-1]
        t_max = t_min * 1.01
        w = w_data[-1]
        a = a_data[-1]
    
    return time_data, w_data, a_data

    





