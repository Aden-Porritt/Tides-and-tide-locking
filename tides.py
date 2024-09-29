import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import scipy
import scipy.integrate

G = 6.67408 * 10**-11

def period_to_omega(period):
  return 2 * np.pi / (period * 24 * 60 * 60)

def calculated_tidal_despin_change(xi, y, system, body):
  w, a = y
  system.a = a
  if body == 's':
    system.omega_s = w
    dw = system.change_in_omega_s()
    da = system.change_in_a_s()
  elif body == 'p':
    system.omega_p = w
    dw = system.change_in_omega_p()
    da = system.change_in_a_p()
  return np.array([dw, da]) * 24 * 60 * 60 * 365.25

class Body:
  def __init__(self, radius, mass, k_2 = 0.1, Q = 100, alpha = 0.4):
    self.radius = radius
    self.mass = mass
    self.k_2 = k_2
    self.Q = Q
    self.alpha = alpha

  def copy(self):
    return Body()

class System:
  def __init__(self, host, satellite, start_a, start_omega_s=period_to_omega(10/24), start_omega_p=period_to_omega(10/24)):
    self.host = host
    self.satellite = satellite
    self.start_a = start_a # orbital distance m
    self.a = start_a
    self.start_n = self.keplers_third_law()
    self.start_omega_s = start_omega_s # satellites angular velocity
    self.start_omega_p = start_omega_p # planets angular velocity
    self.omega_s = start_omega_s
    self.omega_p = start_omega_p
    self.s_despin = 0.0
    self.p_despin = 0.0
    self.s_despin_data = [[], []]
    self.p_despin_data = [[], []]
    self.sol = [None, None]

  def reset(self):
    self.a = self.start_a
    self.omega_s = self.start_omega_s
    self.omega_p = self.start_omega_p

  def keplers_third_law(self):
    return np.sqrt(G * (self.satellite.mass + self.host.mass) / (self.a ** 3))

  def change_in_a_s(self):
    return np.sign(self.omega_s - self.keplers_third_law()) * (3 * self.satellite.k_2 / self.satellite.Q) * (self.host.mass / self.satellite.mass) * (self.satellite.radius / self.a) ** 5 * self.keplers_third_law() * self.a

  def change_in_a_p(self):
    return np.sign(self.omega_p - self.keplers_third_law()) * (3 * self.host.k_2 / self.host.Q) * (self.satellite.mass / self.host.mass) * (self.host.radius / self.a) ** 5 * self.keplers_third_law() * self.a

  def change_in_omega_s(self):
    return -np.sign(self.omega_s - self.keplers_third_law()) * (3 * self.satellite.k_2 / (2 * self.satellite.alpha * self.satellite.Q)) * (self.host.mass ** 2 / (self.satellite.mass * (self.host.mass + self.satellite.mass))) * (self.satellite.radius / self.a) ** 3 * self.keplers_third_law() ** 2

  def change_in_omega_p(self):
    return -np.sign(self.omega_p - self.keplers_third_law()) * (3 * self.host.k_2 / (2 * self.host.alpha * self.host.Q)) * (self.satellite.mass ** 2 / (self.host.mass * (self.host.mass + self.satellite.mass))) * (self.host.radius / self.a) ** 3 * self.keplers_third_law() ** 2

  def step_forward(self, time, body):
    if body == 's':
      self.omega_s += time * self.change_in_omega_s()
      self.a += time * self.change_in_a_s()
    elif body == 'p':
      self.omega_p += time * self.change_in_omega_p()
      self.a += time * self.change_in_a_p()
    
  def calculated_tidal_despin_scipy_helper(self, body, xi_max, steps):
    self.reset()
    self.reset()
    if body == 's':
      initial_conditions = np.array([self.start_omega_s, self.start_a])

    elif body == 'p':
      initial_conditions = np.array([self.start_omega_p, self.start_a])
    
    return scipy.integrate.solve_ivp(calculated_tidal_despin_change, [0, xi_max], initial_conditions, t_eval=np.linspace(0, xi_max,steps), args=(self, body,), method='RK45')

  def calculated_tidal_despin_scipy(self, body, xi_max=100, steps=10):
    for _ in range(1000):
      sol = self.calculated_tidal_despin_scipy_helper(body, xi_max, steps)

      if body == 's':
        if min(sol.y[0] - self.keplers_third_law()) < (self.start_omega_s - self.start_n) / 1000:
          break
    
      if body == 'p':
        if min(sol.y[0] - self.keplers_third_law()) < (self.start_omega_p - self.start_n) / 1000:
          break
      
      xi_max *= 1.5

    sol = self.calculated_tidal_despin_scipy_helper(body, xi_max, steps * 10)

    years_data = []
    w_data = []

    for i, w in enumerate(sol.y[0]):
      if w - self.keplers_third_law() < (self.start_omega_p - self.start_n) / 1000:
        break
      w_data.append(w)
      years_data.append(sol.t[i])
    
    if body == 's':
      self.sol[1] = sol
      self.s_despin_data[0] = years_data.copy()
      self.s_despin_data[1] = w_data.copy()

    elif body == 'p':
      self.sol[0] = sol
      self.p_despin_data[0] = years_data.copy()
      self.p_despin_data[1] = w_data.copy()
      
  def graph(self, body):
    if body == 's' or body == 'all':
      plt.plot(self.s_despin_data[0], self.s_despin_data[1], label="s")
    if body == 'p' or body == 'all':
      plt.plot(self.p_despin_data[0], self.p_despin_data[1], label="p")
    plt.legend()
    plt.show()

  def __str__(self):
    return f"s_despin: {self.s_despin:.2E}, p_despin: {self.p_despin:.2E}"
  
# radius, mass, k_2, Q, alpha

sun = Body(695700 * 10**3, 1.989 * 10**30)

# planets
mercury = Body(2440 * 10**3, 3.302 * 10**23, 0.1, 100, 0.33)
venus = Body(6051.8 * 10**3, 48.685 * 10**23, 0.25, 100, 0.33)
earth = Body(6371 * 10**3, 59.736 * 10**23, 0.299, 12, 0.3308)
mars = Body(3389.9 * 10**3, 6.4185 * 10**23, 0.14, 86, 0.366)
jupiter = Body(69911*10**3, 1898 * 10**24)
saturn = Body(58232*10**3, 568 * 10**24)
uranus = Body(25559*10**3, 86.8 * 10**24)
neptune = Body(24622*10**3, 102 * 10**24)

# earth
moon = Body(1737.53 * 10**3, 7.349 * 10**22, 0.03, 27)

# mars
phobos = Body(9.3 * 10**3, 1.08 * 10**16, 0.0000004, 100)
deimos = Body(7.8 * 10**3, 1.80 * 10**15)

# jupiter
io = Body(1821.3 * 10**3, 893.3 * 10**20, 0.03, 100)
europa = Body(1565 * 10**3, 479.7 * 10**20, 0.02, 100)

# saturn
hyperion = Body(185 * 10**3, 5.58 * 10**18, 0.0003, 100)

# uranus
miranda = Body(240 * 10**3, 0.659 * 10**20, 0.0009, 100)
ariel = Body(578 * 10**3, 13.53 * 10**20, 0.1, 100) # 

# neptune
triton = Body(1352.6 * 10**3, 214.7 * 10**20, 0.086, 100)

# pluto
pluto = Body(1137 * 10**3, 1.27 * 10**22, 0.06, 100)
charon = Body(586 * 10**3, 1.5 * 10**21, 0.006, 100)


# sun_mercury = System(sun, mercury, 47.8 * 10**9)
# sun_mercury.calculated_tidal_despin('s')
# print("sun_mercury")
# print(sun_mercury)

# sun_venus = System(sun, venus, 108.2 * 10**9)
# sun_venus.calculated_tidal_despin('s')
# print("sun_venus")
# print(sun_venus)

# sun_earth = System(sun, earth, 149.6 * 10**9)
# sun_earth.calculated_tidal_despin('s')
# print("sun_earth")
# print(sun_earth)

# sun_mars = System(sun, mars, 227.9 * 10**9)
# sun_mars.calculated_tidal_despin('s')
# print("sun_mars")
# print(sun_mars)

# earth_moon = System(earth, moon, 384 * 10**6)
# earth_moon.calculated_tidal_despin('all')
# print("earth_moon")
# print(earth_moon)

# mars_phobos = System(mars, phobos, 9.3 * 10**6)
# mars_phobos.calculated_tidal_despin('s')
# print("mars_phobos")
# print(mars_phobos)

# jupiter_io = System(jupiter, io, 421 * 10**6)
# jupiter_io.calculated_tidal_despin('s')
# print("jupiter_io")
# print(jupiter_io)

# jupiter_europa = System(jupiter, europa, 670 * 10**6)
# jupiter_europa.calculated_tidal_despin('s')
# print("jupiter_europa")
# print(jupiter_europa)

# saturn_hyperion = System(saturn, hyperion, 1471 * 10**6)
# saturn_hyperion.calculated_tidal_despin('s')
# print("saturn_hyperion")
# print(saturn_hyperion)

# uranus_miranda = System(uranus, miranda, 129 * 10**6)
# uranus_miranda.calculated_tidal_despin('s')
# print("uranus_miranda")
# print(uranus_miranda)

# uranus_ariel = System(uranus, ariel, 191 * 10**6)
# uranus_ariel.calculated_tidal_despin('s')
# print("uranus_ariel")
# print(uranus_ariel)

# neptune_triton = System(neptune, triton, 355 * 10**6)
# neptune_triton.calculated_tidal_despin('s')
# print("neptune_triton")
# print(neptune_triton)

# pluto_charon = System(pluto, charon, 19 * 10**6)
# pluto_charon.calculated_tidal_despin('all')
# print("pluto_charon")
# print(pluto_charon)

# mars_phobos.graph('all')
# mars_phobos.reset()
# print(mars_phobos.change_in_a_p())
# print(mars_phobos.change_in_a_s())

sun_earth = System(sun, earth, 149.6 * 10**9)
sun_earth.calculated_tidal_despin_scipy('s', 100, 10000)
print("sun_earth")
print(sun_earth)

sun_earth.graph('all')