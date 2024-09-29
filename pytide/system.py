import pytide
import numpy as np
import scipy.integrate

def sigmoid(x):
    return (2 / (1 + np.exp(-1000000 * x))) - 1

def keplers_third_law(a, body1, body2):
    return np.sqrt(pytide.G * (body1.mass + body2.mass) / a ** 3)

def calculated_a_gradient(w, a, body1, body2):
    return sigmoid(w - keplers_third_law(a, body1, body2)) * (3 * body1.k_2 / (body1.Q)) * (body2.mass / body1.mass) * (body1.radius / a) ** 5 * keplers_third_law(a, body1, body2) * a

def calculated_w_gradient(w, a, body1, body2):
    return -sigmoid(w - keplers_third_law(a, body1, body2)) * (3 * body1.k_2 / (2 * body1.alpha * body1.Q)) * (body2.mass ** 2 / (body1.mass * (body1.mass + body2.mass))) * (body1.radius / a) ** 3 * keplers_third_law(a, body1, body2) ** 2

def calculated_tidal_despin_gradient(t, y, body1, body2):
    w, a = y

    dw = calculated_w_gradient(w, a, body1, body2)

    da = calculated_a_gradient(w, a, body1, body2)

    return np.array([dw, da]) * 365.25 * 24 * 60 * 60

def calculated_tidal_despin(start_a, start_w, body1, body2, steps):
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

    





