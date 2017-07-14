'''
Created on Aug 21, 2012

@author: Andre L. de Amorim
'''

from .tables import read_output_tables
from os import path, unlink
import sys
import copy
from numpy import random, iinfo
from string import Template
from bz2 import BZ2File
from shutil import copyfileobj
from astropy import log

__all__ = ['GridRun', 'GridFile']


###############################################################################
def output_ok(filename):
    if not path.exists(filename):
        return False
    try:
        read_output_tables(filename)
        return True
    except:
        log.warn('Error reading output file %s' % filename)
        return False
###############################################################################


###############################################################################
def delete_if_exists(filename):
    if not path.exists(filename):
        return
    unlink(filename)
###############################################################################


###############################################################################
def bz2_file(filename):
    '''
    Compresses a file, removing the original.

    Parameters
    ----------
    filename : string
        Path to file to be compressed.
    '''
    if filename.endswith('.bz2'):
        return
    filename_bz2 = filename + '.bz2'
    try:
        with open(filename, 'r') as fd:
            with BZ2File(filename_bz2, 'wb') as bzfd:
                copyfileobj(fd, bzfd)
        unlink(filename)
    except:
        log.error('Could not compress file %s' % filename)
        if path.exists(filename_bz2):
            unlink(filename_bz2)
###############################################################################


###############################################################################
def add_path_sep(the_path):
    if not the_path.endswith(path.sep):
        the_path += path.sep
    return the_path
###############################################################################


###############################################################################
def abs_to_rel(the_path, root):
    the_path = path.normpath(the_path)
    if the_path.startswith(path.sep):
        return path.relpath(the_path, root)
    else:
        return the_path
###############################################################################


###############################################################################
class GridRun(object):
    '''
    Store the STARLIGHT run configuration for a single spectrum.

    Parameters
    ----------
    in_file : string
        Input spectrum file path.

    etc_info_file : string
        ETC info file path.

    lum_distance_mpc : float
        Luminosity distance in Mpc.

    config_file : string
        Config file path.

    base_file : string
        Base file path.

    mask_file : string
        Mask file path.

    reddening : string
        Reddening law to fit. Either ``'CCM'`` or ``CAL``.

    v0_ini : float
        Initial velocity displacement.

    vd_ini : float
        Initial velocity dispersion.

    out_file : string
        Output file path.
    '''

    def __init__(self, in_file='', etc_info_file='', lum_distance_mpc=0.0, config_file='',
                 base_file='', mask_file='', reddening='CCM', v0_ini=0.0, vd_ini=0.0, out_file=''):
        self.inFile = in_file
        self.outFile = out_file
        self.configFile = config_file
        self.baseFile = base_file
        self.maskFile = mask_file
        self.reddening = reddening
        self.etcInfoFile = etc_info_file
        self.lumDistanceMpc = lum_distance_mpc
        self.v0_Ini = v0_ini
        self.vd_Ini = vd_ini

    @property
    def outFileCompressed(self):
        return self.outFile + '.bz2'

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        attrs = [self.inFile, self.etcInfoFile, self.lumDistanceMpc,
                 self.configFile, self.baseFile, self.maskFile, self.reddening,
                 self.v0_Ini, self.vd_Ini, self.outFile]
        return ' '.join(str(x) for x in attrs)
###############################################################################


###############################################################################
class GridFile(object):
    '''
    STARLIGHT grid file manager. This object's string representation
    is a valid grid file contents, and may be fed directly into STARLIGHT.

    Parameters
    ----------
    starlight_dir : string
        STARLIGHT path. This is the path that contains
        the mask file, the base file, etc. All other
        paths are made relative to this.

    grid_data : string
        The contents of a grid file. Use :meth:`fromFile` to
        load from a grid file.

    See also
    --------
    fromFile, write, checkOutput

    '''

    def __init__(self, starlight_dir, grid_data=None):
        self._starlightDir = path.abspath(starlight_dir)
        self.setLogDir('log')
        self.runs = []
        self.completed = []
        self.failed = []
        if grid_data is not None:
            self._load(grid_data)
            return
        self.name = None
        self.setBasesDir('.')    # [base_dir]
        self.setObsDir('.')      # [obs_dir]
        self.setOutDir('.')      # [out_dir]
        self.setMaskDir('.')     # [mask_dir]
        self.setEtcDir('.')      # [etc_dir]
        self.randPhone = 0       # [random phone number]
        # [llow_SN]   lower-lambda of S/N window <-- Not relevant when error-spectrum is provided
        self.lLow_SN = 5350.0
        # [lupp_SN]   upper-lambda of S/N window <-- Not relevant when error-spectrum is provided
        self.lUpp_SN = 5850.0
        self.lLow_Syn = 3650.0   # [Olsyn_ini] lower-lambda for fit
        self.lUpp_Syn = 6850.0   # [Olsyn_fin] upper-lambda for fit
        self.dLambda = 1.0       # [Odlsyn]    delta-lambda for fit
        self.fScale_Chi2 = 1.0   # [fscale_chi2] fudge-factor for chi2
        self.fitFix = 'FIT'      # [FIT/FXK] Fit or Fix kinematics
        self.errSpecAvail = 1  # [IsErrSpecAvailable]  1/0 = Yes/No
        self.flagSpecAvail = 1   # [IsFlagSpecAvailable] 1/0 = Yes/No
        # [IsPHOcOn] 1/0 = Yes/No  <=== !PHO! ATT: still needs coding + testing!
        self.isPhoEnabled = 0
        self.isQHREnabled = 0    # [IsQHRcOn] 1/0 = Yes/No  <=== !QHR!
        self.isFIREnabled = 0    # [IsFIRcOn] 1/0 = Yes/No  <=== !FIR!
        # [flux_unit] multiply spectrum in arq_obs by this value to obtain ergs/s/cm2/Angs
        self.fluxUnit = 1e-16

    @classmethod
    def fromFile(cls, starlight_dir, grid_file_name):
        '''
        Read the grid configuration from a file.

        Parameters
        ----------
        starlight_dir : string
            STARLIGHT path. This is the path that contains
            the mask file, the base file, etc. All other
            paths are made relative to this.

        grid_file_name : string
            The path to a grid file, relative to ``starlight_dir``.

        '''
        grid_data = open(grid_file_name).read()
        grid = cls(starlight_dir, grid_data)
        grid.name = path.basename(grid_file_name)
        return grid

    @property
    def starlightDir(self):
        return self._starlightDir

    def setBasesDir(self, bases_dir):
        self._basesDir = abs_to_rel(bases_dir, self._starlightDir)

    @property
    def basesDir(self):
        return self._basesDir

    @property
    def basesDirAbs(self):
        return path.join(self._starlightDir, self._basesDir)

    @property
    def basesDirSL(self):
        return add_path_sep(self._basesDir)

    def setObsDir(self, obs_dir):
        self._obsDir = abs_to_rel(obs_dir, self._starlightDir)

    @property
    def obsDir(self):
        return self._obsDir

    @property
    def obsDirAbs(self):
        return path.join(self._starlightDir, self._obsDir)

    @property
    def obsDirSL(self):
        return add_path_sep(self._obsDir)

    def setOutDir(self, out_dir):
        self._outDir = abs_to_rel(out_dir, self._starlightDir)

    @property
    def outDir(self):
        return self._outDir

    @property
    def outDirAbs(self):
        return path.join(self._starlightDir, self._outDir)

    @property
    def outDirSL(self):
        return add_path_sep(self._outDir)

    def setMaskDir(self, mask_dir):
        self._maskDir = abs_to_rel(mask_dir, self._starlightDir)

    @property
    def maskDir(self):
        return self._maskDir

    @property
    def maskDirAbs(self):
        return path.join(self._starlightDir, self._maskDir)

    @property
    def maskDirSL(self):
        return add_path_sep(self._maskDir)

    def setEtcDir(self, etc_dir):
        self._etcDir = abs_to_rel(etc_dir, self._starlightDir)

    @property
    def etcDir(self):
        return self._etcDir

    @property
    def etcDirAbs(self):
        return path.join(self._starlightDir, self._etcDir)

    @property
    def etcDirSL(self):
        return add_path_sep(self._etcDir)

    def setLogDir(self, log_dir):
        self._logDir = abs_to_rel(log_dir, self._starlightDir)

    @property
    def logDir(self):
        return self._logDir

    @property
    def logDirAbs(self):
        return path.join(self._starlightDir, self._logDir)

    def seed(self):
        self.randPhone = random.randint(iinfo('int32').max, dtype='int32')

    def _load(self, grid_data):
        l = grid_data.splitlines()
        self.setBasesDir(l[1].split()[0])
        self.setObsDir(l[2].split()[0])
        self.setMaskDir(l[3].split()[0])
        self.setEtcDir(l[4].split()[0])
        self.setOutDir(l[5].split()[0])
        self.randPhone = int(l[6].split()[0])
        self.lLow_SN = float(l[7].split()[0])
        self.lUpp_SN = float(l[8].split()[0])
        self.lLow_Syn = float(l[9].split()[0])
        self.lUpp_Syn = float(l[10].split()[0])
        self.dLambda = float(l[11].split()[0])
        self.fScale_Chi2 = float(l[12].split()[0])
        self.fitFix = l[13].split()[0]
        self.errSpecAvail = int(l[14].split()[0])
        self.flagSpecAvail = int(l[15].split()[0])
        self.isPhoEnabled = int(l[16].split()[0])
        self.isQHREnabled = int(l[17].split()[0])
        self.isFIREnabled = int(l[18].split()[0])
        self.fluxUnit = float(l[19].split()[0])
        for runs in l[20:]:
            rr = runs.split()
            self.runs.append(GridRun(rr[0], rr[1], float(rr[2]), rr[3], rr[4], rr[5],
                                     rr[6], float(rr[7]), float(rr[8]), rr[9]))

    def appendRun(self, in_file='', out_file='', config_file='', base_file='', mask_file='', reddening='CCM'):
        self.runs.append(
            GridRun(in_file, out_file, config_file, base_file, mask_file, reddening))

    def __str__(self):
        tplFile = path.join(path.dirname(__file__), 'gridfile.template')
        tpl = Template(open(tplFile).read())
        attrs = ['runsCount', 'basesDirSL', 'obsDirSL', 'maskDirSL', 'etcDirSL', 'outDirSL',
                 'randPhone', 'lLow_SN', 'lUpp_SN', 'lLow_Syn', 'lUpp_Syn', 'dLambda',
                 'fScale_Chi2', 'fitFix', 'errSpecAvail', 'flagSpecAvail',
                 'isPhoEnabled', 'isQHREnabled', 'isFIREnabled', 'fluxUnit',
                 ]

        mapping = {k: str(getattr(self, k)) for k in attrs}
        return tpl.substitute(mapping) + '\n'.join([str(run) for run in self.runs])

    def write(self, grid_file_name):
        open(grid_file_name, 'w').write(str(self))

    @property
    def runsCount(self):
        '''
        The number of remaining runs.
        '''
        return len(self.runs)

    def failRun(self, ix=0):
        '''
        Mark the indicated run as failed.

        Parameters
        ----------
        ix : integer
            The index of the run to be marked as failed.
            Defaults to the topmost one (``ix = 0``).

        See also
        --------
        failRuns
        '''
        bad_run = self.runs.pop(0)
        self.failed.append(bad_run)

    def failRuns(self):
        '''
        Marks all runs as failed.

        See also
        --------
        failRun, clearRuns
        '''
        for _ in range(len(self.runs)):
            self.failRun(0)

    def clearRuns(self):
        '''
        Clear all the run lists. 

        See also
        --------
        failRuns
        '''
        self.runs = []
        self.completed = []
        self.failed = []

    def checkOutput(self, compress=True):
        '''
        Check starlight output for a list of runs.
        Compresses the OK runs and deletes the bad ones.
        Updates the completed and remaining runs list.

        Parameters
        ----------
        compress : bool
            Compress the output files (bzip2). Default: ``True``.

        '''
        if not path.exists(self.outDirAbs):
            log.warn('Output dir does not exist! Skipping check.')
            return
        if len(self.runs) == 0:
            log.warn('Nothing to check.')
            return
        done = []
        not_done = []
        out_dir = self.outDirAbs
        for r in self.runs:
            filename = path.join(out_dir, r.outFile)
            filenamebz2 = path.join(out_dir, r.outFileCompressed)
            if output_ok(filenamebz2):
                done.append(r)
            elif output_ok(filename):
                if compress:
                    bz2_file(filename)
                done.append(r)
            else:
                not_done.append(r)
                delete_if_exists(filename)
                delete_if_exists(filenamebz2)
        self.completed.extend(done)
        self.runs = not_done

    def copy(self):
        return copy.deepcopy(self)


