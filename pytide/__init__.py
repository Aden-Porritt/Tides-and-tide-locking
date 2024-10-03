
import numpy as np

from pytide.body import Body
import pytide.body as body
import pytide.system as system
from pytide.system import G

def period_to_omega(period):
    return 2 * np.pi / (period * 24 * 60 * 60)

print('hello')