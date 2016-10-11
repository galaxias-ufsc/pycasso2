'''
Created on 21 de set de 2016

@author: andre
'''

from pycasso2.wcs import find_nearest_index
import numpy as np

__all__ = ['get_Lick_index', 'list_Lick_indices']


################################################################################
def get_Lick_index(l_obs, flux, dl, index_id):
    b = Lick_indices[index_id]
    return b.fromFlux(l_obs, flux, dl)
################################################################################


################################################################################
def list_Lick_indices():
    indices = Lick_indices.keys()
    indices.sort()
    return indices
################################################################################


################################################################################
class IndexEW(object):

    def __init__(self, wave1, wave2, wave1_b, wave2_b, wave1_r, wave2_r):
        self.wave1 = wave1
        self.wave2 = wave2
        self.wave1_b = wave1_b
        self.wave2_b = wave2_b
        self.wave1_r = wave1_r
        self.wave2_r = wave2_r
        # FIXME: Index G4300 has overlapping bands (!!!), disabled sanity check.
        #if not self._isValid():
        #    raise Exception('Invalid EW bands setup.')

    def _isValid(self):
        return (self.wave1_b < self.wave2_b) and \
               (self.wave2_b <= self.wave1) and \
               (self.wave1 < self.wave2) and \
               (self.wave2 <= self.wave1_r) and \
               (self.wave1_r <= self.wave2_r)

    def fromFlux(self, l_obs, flux, dl):
        values = [self.wave1, self.wave2, self.wave1_b,
                  self.wave2_b, self.wave1_r, self.wave2_r]
        l1, l2, l1_b, l2_b, l1_r, l2_r = find_nearest_index(l_obs, values)
        cont_b = flux[l1_b:l2_b].mean(axis=0)
        cont_r = flux[l1_r:l2_r].mean(axis=0)
        wave_c_b = (self.wave1_b + self.wave2_b) / 2
        wave_c_r = (self.wave1_r + self.wave2_r) / 2
        l_obs = l_obs[l1:l2, np.newaxis, np.newaxis]
        flux = flux[l1:l2]

        alpha = (cont_r - cont_b) / (wave_c_r - wave_c_b)
        cont1 = cont_b + alpha * (self.wave1 - wave_c_b)
        cont = cont1 + alpha * (l_obs - self.wave1)

        # FIXME: How to integrate masked arrays?
        return np.trapz((cont - flux) / cont, dx=dl, axis=0)
################################################################################


################################################################################
class IndexMag(object):

    def __init__(self, wave1, wave2, wave1_b, wave2_b, wave1_r, wave2_r):
        self.wave1 = wave1
        self.wave2 = wave2
        self.wave1_b = wave1_b
        self.wave2_b = wave2_b
        self.wave1_r = wave1_r
        self.wave2_r = wave2_r
        # FIXME: Index G4300 has overlapping bands (!!!), disabled sanity check.
        #if not self._isValid():
        #    raise Exception('Invalid EW bands setup.')

    def _isValid(self):
        return (self.wave1_b < self.wave2_b) and \
               (self.wave2_b <= self.wave1) and \
               (self.wave1 < self.wave2) and \
               (self.wave2 <= self.wave1_r) and \
               (self.wave1_r <= self.wave2_r)

    def fromFlux(self, l_obs, flux, dl):
        values = [self.wave1, self.wave2, self.wave1_b,
                  self.wave2_b, self.wave1_r, self.wave2_r]
        l1, l2, l1_b, l2_b, l1_r, l2_r = find_nearest_index(l_obs, values)
        cont_b = flux[l1_b:l2_b].mean(axis=0)
        cont_r = flux[l1_r:l2_r].mean(axis=0)
        wave_c_b = (self.wave1_b + self.wave2_b) / 2
        wave_c_r = (self.wave1_r + self.wave2_r) / 2
        l_obs = l_obs[l1:l2, np.newaxis, np.newaxis]
        flux = flux[l1:l2]

        alpha = (cont_r - cont_b) / (wave_c_r - wave_c_b)
        cont1 = cont_b + alpha * (self.wave1 - wave_c_b)
        cont = cont1 + alpha * (l_obs - self.wave1)

        # FIXME: How to integrate masked arrays?
        return -2.5 * np.log10(1.0 / (self.wave1 - self.wave2) * np.trapz(flux / cont, dx=dl, axis=0))
################################################################################


################################################################################
class IndexFluxRatio(object):

    def __init__(self, wave1_b, wave2_b, wave1_r, wave2_r):
        self.wave1_b = wave1_b
        self.wave2_b = wave2_b
        self.wave1_r = wave1_r
        self.wave2_r = wave2_r
        if not self._isValid():
            raise Exception('Invalid flux ratio bands setup.')

    def _isValid(self):
        return (self.wave1_b < self.wave2_b) and \
               (self.wave2_b <= self.wave1_r) and \
               (self.wave1_r <= self.wave2_r)

    def fromFlux(self, l_obs, flux, dl):
        values = [self.wave1_b, self.wave2_b, self.wave1_r, self.wave2_r]
        l1_b, l2_b, l1_r, l2_r = find_nearest_index(l_obs, values)
        flux_b = flux[l1_b:l2_b]
        flux_r = flux[l1_r:l2_r]

        # FIXME: How to integrate masked arrays?
        return np.trapz(flux_r, dx=dl, axis=0) / np.trapz(flux_b, dx=dl, axis=0)
################################################################################


################################################################################
class IndexMgFe(object):

    def fromFlux(self, l_obs, flux, dl):
        Mg_b = get_Lick_index(l_obs, flux, dl, 'Mg_b')
        Fe5270 = get_Lick_index(l_obs, flux, dl, 'Fe5270')
        Fe5335 = get_Lick_index(l_obs, flux, dl, 'Fe5335')
        return np.sqrt(Mg_b * (0.72 * Fe5270 + 0.28 * Fe5335))
################################################################################


################################################################################
Lick_indices = {'CN_1': IndexMag(4142.125, 4177.125, 4080.125, 4117.625, 4244.125, 4284.125),
                'CN_2': IndexMag(4142.125, 4177.125, 4083.875, 4096.375, 4244.125, 4284.125),
                'Ca4227': IndexEW(4222.250, 4234.750, 4211.000, 4219.750, 4241.000, 4251.000),
                'G4300': IndexEW(4281.375, 4316.375, 4266.375, 4282.625, 4318.875, 4335.125),
                'Fe4383': IndexEW(4369.125, 4420.375, 4359.125, 4370.375, 4442.875, 4455.375),
                'Ca4455': IndexEW(4452.125, 4474.625, 4445.875, 4454.625, 4477.125, 4492.125),
                'Fe4531': IndexEW(4514.250, 4559.250, 4504.250, 4514.250, 4560.500, 4579.250),
                'Fe4668': IndexEW(4633.999, 4720.250, 4611.500, 4630.250, 4742.750, 4756.500),
                'H_beta': IndexEW(4847.875, 4876.625, 4827.875, 4847.875, 4876.625, 4891.625),
                'Fe5015': IndexEW(4977.750, 5054.000, 4946.500, 4977.750, 5054.000, 5065.250),
                'Mg_1': IndexMag(5069.125, 5134.125, 4895.125, 4957.625, 5301.125, 5366.125),
                'Mg_2': IndexMag(5154.125, 5196.625, 4895.125, 4957.625, 5301.125, 5366.125),
                'Mg_b': IndexEW(5160.125, 5192.625, 5142.625, 5161.375, 5191.375, 5206.375),
                'Fe5270': IndexEW(5245.650, 5285.650, 5233.150, 5248.150, 5285.650, 5318.150),
                'Fe5335': IndexEW(5312.125, 5352.125, 5304.625, 5315.875, 5353.375, 5363.375),
                'Fe5406': IndexEW(5387.500, 5415.000, 5376.250, 5387.500, 5415.000, 5425.000),
                'Fe5709': IndexEW(5696.625, 5720.375, 5672.875, 5696.625, 5722.875, 5736.625),
                'Fe5782': IndexEW(5776.625, 5796.625, 5765.375, 5775.375, 5797.875, 5811.625),
                'Na_D': IndexEW(5876.875, 5909.375, 5860.625, 5875.625, 5922.125, 5948.125),
                'TiO_1': IndexMag(5936.625, 5994.125, 5816.625, 5849.125, 6038.625, 6103.625),
                'TiO_2': IndexMag(6189.625, 6272.125, 6066.625, 6141.625, 6372.625, 6415.125),
                'H_delta_A': IndexEW(4083.500, 4122.250, 4041.600, 4079.750, 4128.500, 4161.000),
                'H_gamma_A': IndexEW(4319.750, 4363.500, 4283.500, 4319.750, 4367.250, 4419.750),
                'H_delta_F': IndexEW(4090.999, 4112.250, 4057.250, 4088.500, 4114.750, 4137.250),
                'H_gamma_F': IndexEW(4331.250, 4352.250, 4283.500, 4319.750, 4354.750, 4384.750),
                # FIXME: Copied these two indices from Rafa, not sure why we have them.
                #'Na_D_mod': IndexEW(5876.875, 5909.375, 5860.625, 5875.625, 5922.125, 5948.125),
                #'TiO_2_Califa': IndexEW(6189.625, 6272.125, 6060.625, 6080.625, 6372.625, 6415.125),
                'D4000': IndexFluxRatio(3850.0, 3950.0, 4000.0, 4100.0),
                'MgFe': IndexMgFe(),
                }
################################################################################
