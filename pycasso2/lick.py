'''
Created on 21 de set de 2016

@author: andre
'''

from .resampling import find_nearest_index
import numpy as np

__all__ = ['get_Lick_index', 'list_Lick_indices']


##########################################################################
def get_Lick_index(index_id, l_obs, flux, error=None):
    if not index_id in Lick_indices:
        raise Exception('Unknown Lick index "%s".')
    b = Lick_indices[index_id]
    return b.fromFlux(l_obs, flux, error)
##########################################################################


##########################################################################
def list_Lick_indices():
    indices = list(Lick_indices.keys())
    indices.sort()
    return indices
##########################################################################


##########################################################################
class IndexEW(object):

    def __init__(self, wave1, wave2, wave1_b, wave2_b, wave1_r, wave2_r):
        self.wave1 = wave1
        self.wave2 = wave2
        self.wave1_b = wave1_b
        self.wave2_b = wave2_b
        self.wave1_r = wave1_r
        self.wave2_r = wave2_r
        # FIXME: Index G4300 has overlapping bands (!!!), disabled sanity check.
        # if not self._isValid():
        #    raise Exception('Invalid EW bands setup.')

    def _isValid(self):
        return (self.wave1_b < self.wave2_b) and \
               (self.wave2_b <= self.wave1) and \
               (self.wave1 < self.wave2) and \
               (self.wave2 <= self.wave1_r) and \
               (self.wave1_r <= self.wave2_r)

    def fromFlux(self, l_obs, flux, error=None):
        values = [self.wave1, self.wave2, self.wave1_b,
                  self.wave2_b, self.wave1_r, self.wave2_r]
        l1, l2, l1_b, l2_b, l1_r, l2_r = find_nearest_index(l_obs, values)
        cont_b = flux[l1_b:l2_b].mean(axis=0)
        cont_r = flux[l1_r:l2_r].mean(axis=0)
        wave_c_b = (self.wave1_b + self.wave2_b) / 2.0
        wave_c_r = (self.wave1_r + self.wave2_r) / 2.0
        if flux.ndim == 2:
            l_obs = l_obs[l1:l2, np.newaxis]
        elif flux.ndim == 3:
            l_obs = l_obs[l1:l2, np.newaxis, np.newaxis]
        else:
            raise Exception('flux must be 2 or 3-dimensional.')
        flux = flux[l1:l2]
        dl = l_obs[1] - l_obs[0]

        alpha = (cont_r - cont_b) / (wave_c_r - wave_c_b)
        cont1 = cont_b + alpha * (self.wave1 - wave_c_b)
        cont = cont1 + alpha * (l_obs - self.wave1)

        # FIXME: How to integrate masked arrays?
        ew = np.trapz((cont - flux) / cont, dx=dl, axis=0)
        if error is not None:
            error = error[l1:l2]
            ew_error = np.sqrt(np.trapz((error / cont)**2, dx=dl, axis=0))
        else:
            ew_error = None
        return ew, ew_error
##########################################################################


##########################################################################
class IndexMag(object):

    def __init__(self, wave1, wave2, wave1_b, wave2_b, wave1_r, wave2_r):
        self.wave1 = wave1
        self.wave2 = wave2
        self.wave1_b = wave1_b
        self.wave2_b = wave2_b
        self.wave1_r = wave1_r
        self.wave2_r = wave2_r
        # FIXME: Index G4300 has overlapping bands (!!!), disabled sanity check.
        # if not self._isValid():
        #    raise Exception('Invalid EW bands setup.')

    def _isValid(self):
        return (self.wave1_b < self.wave2_b) and \
               (self.wave2_b <= self.wave1) and \
               (self.wave1 < self.wave2) and \
               (self.wave2 <= self.wave1_r) and \
               (self.wave1_r <= self.wave2_r)

    def fromFlux(self, l_obs, flux, error=None):
        values = [self.wave1, self.wave2, self.wave1_b,
                  self.wave2_b, self.wave1_r, self.wave2_r]
        l1, l2, l1_b, l2_b, l1_r, l2_r = find_nearest_index(l_obs, values)
        cont_b = flux[l1_b:l2_b].mean(axis=0)
        cont_r = flux[l1_r:l2_r].mean(axis=0)
        wave_c_b = (self.wave1_b + self.wave2_b) / 2.0
        wave_c_r = (self.wave1_r + self.wave2_r) / 2.0
        if flux.ndim == 2:
            l_obs = l_obs[l1:l2, np.newaxis]
        elif flux.ndim == 3:
            l_obs = l_obs[l1:l2, np.newaxis, np.newaxis]
        else:
            raise Exception('flux must be 2 or 3-dimensional.')
        flux = flux[l1:l2]
        dl = l_obs[1] - l_obs[0]

        alpha = (cont_r - cont_b) / (wave_c_r - wave_c_b)
        cont1 = cont_b + alpha * (self.wave1 - wave_c_b)
        cont = cont1 + alpha * (l_obs - self.wave1)

        # FIXME: How to integrate masked arrays?
        int_flux_cont = np.trapz(flux / cont, dx=dl, axis=0)
        mag = -2.5 * np.log10(1.0 / (self.wave2 - self.wave1) * int_flux_cont)
        if error is not None:
            error = error[l1:l2]
            aux = (2.5 * 10**(-mag / 2.5))
            e1 = (((1.0 / (l2 - l1)**2) * int_flux_cont)**2) * (error[-1]**2)
            e2 = (((1.0 / (l2 - l1)**2) * int_flux_cont)**2) * (error[0]**2)
            e3 = (
                ((1.0 / (l2 - l1)) * np.trapz(error / cont, dx=dl, axis=0))**2)
            mag_error = aux * np.sqrt(e1 + e2 + e3)
        else:
            mag_error = None
        return mag, mag_error
##########################################################################


##########################################################################
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

    def fromFlux(self, l_obs, flux, error=None):
        values = [self.wave1_b, self.wave2_b, self.wave1_r, self.wave2_r]
        l1_b, l2_b, l1_r, l2_r = find_nearest_index(l_obs, values)
        flux_b = flux[l1_b:l2_b]
        flux_r = flux[l1_r:l2_r]
        dl = l_obs[1] - l_obs[0]

        # FIXME: How to integrate masked arrays?
        int_flux_r = np.trapz(flux_r, dx=dl, axis=0)
        int_flux_b = np.trapz(flux_b, dx=dl, axis=0)
        ratio = int_flux_r / int_flux_b
        if error is not None:
            error_b = error[l1_b:l2_b]
            error_r = error[l1_r:l2_r]
            int_error_b = np.trapz(error_b, dx=dl, axis=0)
            int_error_r = np.trapz(error_r, dx=dl, axis=0)
            e1 = int_error_r / int_flux_b
            e2 = (- int_flux_r) / ((int_flux_b)**2) * int_error_b
            ratio_error = np.sqrt(e1**2 + e2**2)
        else:
            ratio_error = None
        return ratio, ratio_error

##########################################################################


##########################################################################
class IndexMgFe(object):

    def fromFlux(self, l_obs, flux, error=None):
        Mg_b, Mg_b_error = get_Lick_index('Mg_b', l_obs, flux, error)
        Fe5270, Fe5270_error = get_Lick_index('Fe5270', l_obs, flux, error)
        Fe5335, Fe5335_error = get_Lick_index('Fe5335', l_obs, flux, error)
        MgFe = np.sqrt(Mg_b * (0.72 * Fe5270 + 0.28 * Fe5335))
        if error is not None:
            e1 = 1.0 / \
                (2.0 * np.sqrt(Mg_b_error * (0.72 * Fe5270 + 0.28 * Fe5335)))
            e2 = 1.0 / (2.0 * np.sqrt(Mg_b * 0.72 * Fe5270_error))
            e3 = 1.0 / (2.0 * np.sqrt(Mg_b * 0.28 * Fe5335_error))
            MgFe_error = np.sqrt(e1**2 + e2**2 + e3**2)
        else:
            MgFe_error = None
        return MgFe, MgFe_error
##########################################################################


##########################################################################
Lick_indices = {'CN_1': IndexMag(4142.125, 4177.125, 4080.125, 4117.625, 4244.125, 4284.125),
                'CN_2': IndexMag(4142.125, 4177.125, 4083.875, 4096.375, 4244.125, 4284.125),
                'Ca4227': IndexEW(4222.250, 4234.750, 4211.000, 4219.750, 4241.000, 4251.000),
                'G4300': IndexEW(4281.375, 4316.375, 4266.375, 4282.625, 4318.875, 4335.125),
                'Fe4383': IndexEW(4369.125, 4420.375, 4359.125, 4370.375, 4442.875, 4455.375),
                'Ca4455': IndexEW(4452.125, 4474.625, 4445.875, 4454.625, 4477.125, 4492.125),
                'Fe4531': IndexEW(4514.250, 4559.250, 4504.250, 4514.250, 4560.500, 4579.250),
                'Fe4668': IndexEW(4633.999, 4720.250, 4611.500, 4630.250, 4742.750, 4756.500),
                'H_beta': IndexEW(4847.875, 4876.625, 4827.875, 4847.875, 4876.625, 4891.625),
                'H_alpha': IndexEW(6542.820, 6582.820, 6480.0, 6530.0, 6600.0, 6670.0),
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
##########################################################################
