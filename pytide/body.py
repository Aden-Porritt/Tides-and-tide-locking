
import pytide.system as system

class Body:
    def __init__(self, name, radius, mass, k_2 = 0.1, Q = 100, alpha = 0.4):
        self.name = name
        self.radius = radius
        self.mass = mass
        self.k_2 = k_2
        self.Q = Q
        self.alpha = alpha
        self.satellites = {}

    def add_satellite(self, body, orbit_radius):
        self.satellites[body.name] = [body, orbit_radius]

    def get_despin_time(self, satellite, a = 1, w_start=0.000174, despin_body='s', m='slow'):
        if type(satellite) == str:
            try:
                satellite, a = self.satellites[satellite]
            except:
                print("not in satellites")
                return None
        w_start = w_start + system.keplers_third_law(a, self, satellite)

        if despin_body == 's':
            body1 = satellite
            body2 = self
        else:
            body1 = self
            body2 = satellite

        if m == 'slow':
            sol = system.calculated_tidal_despin(a, w_start, body1, body2, 100)

        if m == 'fast':
            despin_time = system.calculated_tidal_despin_fast(a, body1)
            print(f'{body1.name}, {body2.name}\n{despin_time:.2E}')
            return None

        print(f'{body1.name}, {body2.name}\n{sol[0][-1]:.2E}')

        return sol

    def copy(self, name=''):
        return Body(name, self.radius, self.mass, self.k_2, self.Q, self.alpha)
    
    def __str__(self):
        return f'{self.name}, radius={self.radius:.2E}, mass={self.mass:.2E}'

# radius, mass, k_2, Q, alpha
sun = Body("sun", 695700 * 10**3, 1.989 * 10**30)

# planets
mercury = Body("mercury", 2440.0 * 10**3, 3.302 * 10**23, 0.1, 100.0, 0.33)
venus = Body("venus", 6051.8 * 10**3, 48.685 * 10**23, 0.25, 100.0, 0.33)
earth = Body("earth", 6371.0 * 10**3, 59.736 * 10**23, 0.299, 12.0, 0.3308)
mars = Body("mars", 3389.9 * 10**3, 6.4185 * 10**23, 0.14, 86.0, 0.366)
jupiter = Body("jupiter", 69911.0*10**3, 1898.0 * 10**24)
saturn = Body("saturn", 58232.0*10**3, 568.0 * 10**24)
uranus = Body("uranus", 25559.0*10**3, 86.8 * 10**24)
neptune = Body("neptune", 24622.0*10**3, 102.0 * 10**24)

sun.add_satellite(mercury, 47.8 * 10**9)
sun.add_satellite(venus, 108.2 * 10**9)
sun.add_satellite(earth, 149.6 * 10**9)
sun.add_satellite(mars, 227.9 * 10**9)
sun.add_satellite(jupiter, 778 * 10**9)
sun.add_satellite(saturn, 0)
sun.add_satellite(uranus, 0)
sun.add_satellite(neptune, 0)

# earth
moon = Body("moon", 1737.53 * 10**3, 7.349 * 10**22, 0.03, 27.0)

earth.add_satellite(moon, 384.0 * 10**6)

# mars
phobos = Body("phobos", 9.3 * 10**3, 1.08 * 10**16, 0.0000004, 100.0)
deimos = Body("deimos", 7.8 * 10**3, 1.80 * 10**15)

mars.add_satellite(phobos, 9.3 * 10**6)
mars.add_satellite(deimos, 0)

# jupiter
io = Body("io", 1821.3 * 10**3, 893.3 * 10**20, 0.03, 100.0)
europa = Body("europa", 1565.0 * 10**3, 479.7 * 10**20, 0.02, 100.0)

jupiter.add_satellite(io, 421.0 * 10**6)
jupiter.add_satellite(europa, 670.0 * 10**6)

# saturn
hyperion = Body("hyperion", 185.0 * 10**3, 5.58 * 10**18, 0.0003, 100.0)

saturn.add_satellite(hyperion, 1471.0 * 10**6)

# uranus
miranda = Body("miranda", 240.0 * 10**3, 0.659 * 10**20, 0.0009, 100.0)
ariel = Body("ariel", 578.0 * 10**3, 13.53 * 10**20, 0.1, 100.0)

uranus.add_satellite(miranda, 129.0 * 10**6)
uranus.add_satellite(ariel, 191.0 * 10**6)

# neptune
triton = Body("triton", 1352.6 * 10**3, 214.7 * 10**20, 0.086, 100.0)

neptune.add_satellite(triton, 355.0 * 10**6)

# pluto
pluto = Body("pluto", 1137.0 * 10**3, 1.27 * 10**22, 0.06, 100.0)
charon = Body("charon", 586.0 * 10**3, 1.5 * 10**21, 0.006, 100.0)

pluto.add_satellite(charon, 19.0 * 10**6)

