from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    f_m: float = 0.025
    fstop: float = 2.0
    f_px: float = 1200.0
    sigma_max: float = 9.0

    def sigma(self, z, zf):
        z = np.asarray(z, np.float64)
        s = self.f_px * self.f_m / (2.0 * self.fstop * (zf - self.f_m)) * np.abs(1.0 - zf / z)
        return np.minimum(s, self.sigma_max)
