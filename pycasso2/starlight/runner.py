'''
Created on Jun 17, 2013

@author: andre
'''

from astropy import log
from os import path
import subprocess
from Queue import Queue
from threading import Thread
from collections import deque
import time

__all__ = ['run_starlight', 'run_starlight_and_check' ,'StarlightRunner']

max_iterations = 10

###############################################################################
def run_starlight(exec_path, starlight_dir, grid_data, timeout, logfile=None):
    '''
    Run STARLIGHT passing the grid data as standard input.
    Set the executable path by changing
    ``pystarlight.util.starlight_runner.starlight_exec_path``.
    
    Parameters
    ----------
    exec_path : string
        Path to starlight executable.
        
    starlight_dir : string
        Working dir where to run STARLIGHT.
        
    grid_data : string
        String containing the contents of a grid file.
        
    timeout : float
        Maximum time to wait for completion, in seconds.

    Returns
    -------
    exit_status : int
        The exit status of the process. Zero means success.
    '''
    log.debug('Running %s' % exec_path)
    slProc = subprocess.Popen(exec_path, cwd=starlight_dir,
                              stdin=subprocess.PIPE, stdout=logfile, stderr=logfile)
    slProc.stdin.write(grid_data)
    slProc.stdin.close()
    t0 = time.time()
    while (time.time() - t0) < timeout:
        res = slProc.poll()
        if res is not None:
            return res
        time.sleep(1.0)
    log.warn('STARLIGHT took too long (%d seconds), sending SIGKILL.' % timeout)
    slProc.kill()
    return -999
###############################################################################


###############################################################################
def run_starlight_and_check(exec_path, grid, timeout, compress=True):
    '''
    Run STARLIGHT with a grid description, retrying until
    there's no error in output files.
    
    Parameters
    ----------
    exec_path : string
        Path to starlight executable.
        
    grid : :class:`pystarlight.util.gridfile.GridFile`
        A grid file descriptor.

    timeout : float
        Maximum time to wait for completion, in seconds.
    '''
    its = 0

    # First check, maybe some files were already run.
    grid.checkOutput(compress)
    if len(grid.completed) > 0:
        log.warn('Skipped %d completed runs from %s.' % (len(grid.completed), grid.name))
    
    t0 = time.time()
    while len(grid.runs) > 0 and its < max_iterations:
        grid_fname = path.join(grid.logDirAbs, '%s_%.4f' % (grid.name, time.time()))
        log_fname = grid_fname + '.log'
        grid.write(grid_fname)
        result = -1
        with open(log_fname, 'w') as logfile:
            log.debug('Starting starlight process for %s, %d to go...' % (grid.name, len(grid.runs)))
            result = run_starlight(exec_path, grid.starlightDir, str(grid), timeout, logfile)
        grid.checkOutput(compress)
        if result == -999 and len(grid.runs) > 0:
            log.error('%s hung last time, changing first run\'s random seed.' % grid.name)
            grid.runs[0].seed()
        elif result <> 0:
            log.error('%s failed with result %d, check your parameters!' % (grid.name, result))
        its += 1
    
    if its >= max_iterations:
        log.error('%s bailed without finishing the job, %d spectra left behind.' % (grid.name, len(grid.runs)))
        grid.failRuns()
        
    if len(grid.completed) > 0:
        log.debug('Took %.1f s to run %s (%d spectra).' % (time.time() - t0, grid.name, len(grid.completed)))
    return grid
###############################################################################


###############################################################################
class StarlightThread(Thread):
    '''
    Thread wrapper running STARLIGHT.
    
    Parameters
    ----------
    input_grids : :class:`Queue.Queue`
        A queue of the grid files to be processed.

    output_grids : :class:`collections.deque`
        A deque of grid files, updated when the processing is
        done, or failed.

    timeout : float
        Maximum time to wait for completion, in seconds.

    compress : bool
        Compress the output files (bzip2). Default: ``True``.
            
    '''
    def __init__(self, exec_path, input_grids, output_grids, timeout, compress):
        Thread.__init__(self)
        self._exec_path = exec_path
        self._inputGrids = input_grids
        self._outputGrids = output_grids
        self._timeout = timeout
        self._compress = compress
        
        
    def run(self):
        while True:
            grid = self._inputGrids.get()
            try:
                if grid.name is None:
                    grid.name = 'grid_%d' % self.ident
                _output_grid = run_starlight_and_check(self._exec_path, grid, self._timeout, compress=self._compress)
                self._outputGrids.append(_output_grid)
            except Exception as e:
                log.error(str(e))
                import traceback
                traceback.print_exc()
            self._inputGrids.task_done()
###############################################################################


###############################################################################
class StarlightRunner(object):
    '''
    Multiprocess STARLIGHT runner. Will run starlight on grid
    files appended into the queue using the :meth:`addGrid` method.
    
    Parameters
    ----------
    n_workers : int, optional
        Number of working processes to use. Defaults to ``1``.

    timeout : float, optional
        Maximum time to wait for completion, in seconds.
        Defaults to 20 minutes.

    compress : bool
        Compress the output files (bzip2). Default: ``True``.
            
    exec_path : string
        Path to starlight executable.
    '''

    def __init__(self, n_workers=1, timeout=1200.0, compress=True, exec_path='starlight'):
        self._inputGrids = Queue(maxsize=2*n_workers)
        self._outputGrids = deque()
        log.debug('starlight executable is %s' % exec_path)
        self._threads = [StarlightThread(exec_path, self._inputGrids, self._outputGrids, timeout, compress=compress) \
                         for _ in xrange(n_workers)]
        for t in self._threads:
            t.setDaemon(True)
            t.start()


    def addGrid(self, grid):
        '''
        Add a grid descriptor to the queue of processes.
        This will block until there's a free slot in the
        grid queue. 
        
        Parameters
        ----------
        grid : :class:`pystarlight.util.gridfile.GridFile`
            A grid file descriptor.
        '''
        self._inputGrids.put(grid, block=True)
        
        
    def wait(self):
        '''
        Wait the grid queue exhaustion.
        
        WARNING: This will block until all the remaining
        grid files are STARLIGHTed.
        '''
        self._inputGrids.join()
        
        
    def getCompletedOutputFiles(self):
        '''
        Get the completed output files so far.
        '''
        completed = []
        for g in self._outputGrids:
            completed.extend(path.join(g.outDirAbs, r.outFile) for r in g.completed)
        return completed


    def getFailedGrids(self):
        '''
        Get the failed completed grids, which
        still have unprocesses grid runs.
        '''
        return [g for g in self._outputGrids if len(g.failed) > 0]


    def getOutputGrids(self):
        '''
        Get the completed output grids so far.
        '''
        return [g for g in self._outputGrids]


    def getQueueSize(self):
        '''
        Get the number of grids in the queue (excluding the
        ones currently running).
        '''
        return self._inputGrids.qsize()
###############################################################################


