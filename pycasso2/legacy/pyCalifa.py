############################################################################
#                       General Tools for CALIFA                           #
#                                                                          #
# RGB@IAA ---> Last Change: 2018/01/16                                     #
############################################################################
#
#
#
################################ VERSION ###################################
VERSION = '0.5.6'                                                          #
############################################################################
#

from pycasso2.legacy.util import getDistance, getImageDistance, getMstarsForTime, getGenHalfRadius
from astropy.table import Table, Column, MaskedColumn, join, vstack, hstack
from pycasso2.legacy import fitsQ3DataCube
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.ndimage.filters import gaussian_filter1d
from itertools import count, izip, cycle, product
from collections import OrderedDict, namedtuple
from astropy.io.ascii import read as astread
import matplotlib.gridspec as gridspec
from subprocess import Popen, PIPE
import scipy.interpolate as sci
from astropy.wcs import WCS
import matplotlib as mpl
import astropy.stats
import scipy.stats
import numpy as np
import traceback
import colorsys
import datetime
import inspect
import pyfits
import types
import copy
import sys
import ast
import re
import os

# ---------------------------------------------------------------------------
class CST(object):
 AU = 1.4960e13     # 1 AU in cm
 PC = 3.08567758e18 # 1 pc in cm
 Lsun = 3.826e33    # Sun bolometric luminosity: ergs/s
 c = 2.99792458e10  # Speed of light in cm/s
 cA = 2.99792458e18 # Speed of light in Angstroms/s
 h = 6.6262e-27     # Planck constant in ergs/s (cgs)
 H0 = 70.0          # Hubble constant (km/s)/Mpc
# ---------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def checkList(var, return_None=True, include_array=False):
 if not (return_None and var is None):
  instances = (list, tuple, np.ndarray) if include_array else (list, tuple)
  if not isinstance(var, instances):
   var = [var]
 return var
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def includeExclude(include,exclude):
 include = checkList(include)
 exclude = checkList(exclude)
 if exclude is not None and include is not None:
  for item in exclude:
   if item in include:
    include.remove(item)
 return include
# ----------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def renameList(lnames,string,prefix=True,exclude=None,sep='_',get_dict=True):
 lnames = includeExclude(lnames,exclude)
 if prefix:
  dnames = ['%s%s%s' % (string,sep,item) for item in lnames]
 else:
  dnames = ['%s%s%s' % (item,sep,string) for item in lnames]
 if get_dict:
  dnames = {key: value for (key,value) in zip(lnames,dnames)}
 return dnames
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def getInteger(string,sep='_',default=None):
 ling = [int(s) for s in string.split(sep) if s.isdigit()]
 if len(ling) > 0:
  value = ling[-1]
 else:
  value = default
 return value
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def getIndexSubList(a, b=None):
 if b is None:
  idx = np.range(a.size)
 else:
  idx = np.atleast_1d(np.searchsorted(a, b))
 return idx
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def joinPath(name, dirname=None, force=True):
 if dirname is not None:
  orig_dirname = os.path.dirname(name)
  name = os.path.basename(name)
  if len(orig_dirname) > 0 and not force:
   dirname = orig_dirname  
  name = os.path.join(dirname, name)
 return name
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def format_exception(trace_exc, fmt=' >>> %s', linsert=None, lappend=None, sep='\n'):
 linsert = checkList(linsert)
 lappend = checkList(lappend)
 trace_exc = trace_exc.split(sep)
 if fmt is not None:
  trace_exc = [fmt % line for line in trace_exc]
 if linsert is not None:
  trace_exc = linsert + trace_exc
 if lappend is not None:
  trace_exc = trace_exc + lappend
 trace_exc = sep.join(trace_exc)
 return trace_exc
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def objectArray2List(value):
 if isinstance(value, np.ndarray):
  if value.dtype == np.dtype(object):
   value = [item.tolist() if isinstance(item, np.ndarray) else item for item in value]
  else:
   value = value.tolist()
 return value
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def getArrayIndices(value, index=None, axis=None, func=None, squeeze=False, args=False):
 ''' 
 Index can be a function, slice or integer. If integer, axis is needed 
 unless the 0 axis (left axis) is the one needed.
 The following multiindex selection is equivalent:
	index = lambda x: x[0,:,1]
 	index = np.index_exp[0,:,1]
        index = (0, slice(None, None, None), 1)
 '''
 if func is not None:
  if isinstance(func, str):
   func = eval(func)
  if args:
   value = func(*value)
  else:
   value = func(value)
 if index is None:
  return value
 if callable(index):
  value = index(value)
 else:
  if axis is None:
   value = value[index]
  else:
   value = np.take(value, index, axis=axis)
 if squeeze:
  value = value.squeeze()
 return value
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def dict2List(dobject, to_string=False):
 if not isinstance(dobject, dict):
  return objectArray2List(dobject)
 ndobj = {}
 for key in dobject:
  value = dobject[key]
  if isinstance(value, dict):
   value = {skey: objectArray2List(svalue) for (skey, svalue) in value.iteritems()}
  else:
   value = objectArray2List(value)
  ndobj[key] = value
 if to_string:
  ndobj = str(ndobj)
 return ndobj
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def invert_arrays_in_dictionary(dictionary, exclude=None):
 dictionary = copy.deepcopy(dictionary)
 exclude = [] if exclude is None else checkList(exclude)
 for key, value in dictionary.iteritems():
  if not key in exclude:
   if isinstance(value, (list, tuple, np.ndarray)):
    dictionary[key] = value[::-1]
 return dictionary
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def updateNestedDict(d, u, extend=False):
 import collections
 for k, v in u.items():
  if isinstance(v, collections.Mapping):
   r = updateNestedDict(d.get(k, {}), v, extend=extend)
   d[k] = r
  else:
   if extend and isinstance(u[k], list) and k in d and isinstance(d[k], list):
    d[k].extend(u[k])
   else:
    d[k] = u[k]
 return d
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def updateDictDefault(dictionary, default_dict=None, deep=True):
 if default_dict is not None and isinstance(default_dict, dict):
  if deep:
   default_dict = copy.deepcopy(default_dict)
  dictionary = updateNestedDict(default_dict, dictionary)
 return dictionary
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def delKeyDict(dictionary, keys, new=False):
 keys = checkList(keys)
 if new:
  dictionary = copy.deepcopy(dictionary)
 for key in keys:
  dictionary.pop(key, None)
 return dictionary
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def getListDefaultNames(defaults, names=None, exact=False, exclude=None, include=None, 
	defaults_include=None, ignore_case=False):
 defaults = checkList(defaults)
 names    = checkList(names)
 exclude  = checkList(exclude)
 include  = checkList(include)
 defaults_include = checkList(defaults_include)
 if exclude is not None:
  flags = re.IGNORECASE if ignore_case else 0
  exclude = re.compile('|'.join(exclude), flags=flags)
 if include is not None:
  flags = re.IGNORECASE if ignore_case else 0
  include = re.compile('|'.join(include), flags=flags)
 if names is None:
  new_names = defaults
 else:
  new_names = []
  for key in names:
   if exact:
    if key in defaults:
     new_names.append(key)
   else:
    if any(key in item for item in defaults):
     new_names.append(key)
 if exclude is not None:
  new_names = [item for item in new_names if exclude.search(item) is None]
 if include is not None:
  new_names = [item for item in new_names if include.search(item) is not None]
 if names is not None and defaults_include is not None:
  flags = re.IGNORECASE if ignore_case else 0
  defaults_include = re.compile('|'.join(defaults_include), flags=flags)
  new_names.extend([item for item in defaults if defaults_include.search(item) is not None])
 return new_names
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def sortedDict(dictionary, key=1):
 import operator
 sorted_dict = sorted(dictionary.items(), key=operator.itemgetter(key))
 return OrderedDict(sorted_dict)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def getCPUS(cpus=None, cpu2job=2, minus_jobs=1):
 if isinstance(cpus, int) and cpus < 2:
  cpus = None
 if (isinstance(cpus, bool) and cpus is True) or (isinstance(cpus, int) and cpus > 1):
  try:
   import multiprocess
  except:
   print ('>>> WARNING: You need to install the "multiprocess" package [NOT "multiprocessing"]!!!')
   return
  total_cpus = multiprocess.cpu_count()
  max_jobs = (total_cpus * cpu2job) - minus_jobs
  if max_jobs < total_cpus:
   max_jobs = total_cpus
 if isinstance(cpus, bool):
  if cpus:
   cpus = max_jobs
  else:
   cpus = None
 if cpus is not None:
  if cpus > max_jobs: 
   cpus = max_jobs
 return cpus
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def get_arguments(get_posargs=False):
 """Returns tuple containing dictionary of calling function's
    named arguments and a list of calling function's unnamed
    positional arguments.
 """
 posname, kwname, args = inspect.getargvalues(inspect.stack()[1][0])[-3:]
 posargs = args.pop(posname, [])
 args.update(args.pop(kwname, []))
 args.pop('self', [])
 if get_posargs:
  return args, posargs
 else:
  return args
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def getDictFunctionArgs(function, remove_self=True):
 afunc = inspect.getargspec(function)
 args = afunc.args
 if 'self' in args and remove_self:
  args.remove('self')
 dfunc = {key: value for (key, value) in zip(args, afunc.defaults)}
 return dfunc
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def getDuplicates(a):
 if not isinstance(a, np.ndarray):
  a = np.array(a)
 m = np.zeros_like(a, dtype=bool)
 m[np.unique(a, return_index=True)[1]] = True
 return a[~m]
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def array_mask1D_operation(prop, lmask, axis=-1, func=None):
 new_shape = (len(lmask),) + prop.shape[1:]

 if isinstance(prop, np.ma.MaskedArray):
  reduce_fill_value = np.ma.masked
  ap_prop = np.ma.masked_all(new_shape)
  ap_prop.fill_value = prop.fill_value
 else:
  ap_prop = np.empty(new_shape)

 for i, mask in enumerate(lmask):
  cprop = np.compress(mask, prop, axis=axis)
  if func is not None:
   cprop = func(cprop, axis=axis)
  ap_prop[i, ...] = cprop

 return ap_prop
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Write a file
def wlarray(name,arr,fmt='%s\n'):
 if isinstance(name,str):
  fil = open(name,'w')
 else:
  fil = name
 for item in arr:
  fil.write(fmt % item)
 fil.close()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def wspec(name,wave,spec,fmt='%6.1f %12.8e'):
 fmt += '\n'
 f = open(name,'w')
 for w,s in zip(wave,spec):
  f.write(fmt % (w,s))
 f.close()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def new_array(shape, dtype, fill_value=0.0):
    if fill_value == 0.0:
        a = np.zeros(shape, dtype)
    elif fill_value == 1.0:
        a = np.ones(shape, dtype)
    else:
        a = np.ones(shape, dtype) * fill_value
    return a
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def fillMaskedArray(x, fill_value=np.nan):
 if isinstance(x, np.ma.MaskedArray):
  x = x.copy()
  x.fill_value = fill_value
  x = x.filled()
 return x
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def replaceMaskMedian(value, big=None, zero=True, axis=0):
 nvalue = value[:]
 if zero:
  nvalue[~(nvalue != 0.0)] = np.nan
 if big is not None:
  med = np.nanmedian(nvalue,axis=axis)
  nvalue[~(nvalue < big*med[np.newaxis,:])] = np.nan
 med = np.nanmedian(nvalue,axis=axis)
 inds = np.where(np.isnan(nvalue))
 nvalue[inds] = np.take(med,inds[1])
 return nvalue
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def dict2object(dictionary,name='RGBclass'):
 return namedtuple(name, dictionary.keys())(**dictionary)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def appendKeyValueDict(idict,key,value,addNone=False):
 if value is not None or (value is None and addNone):
  if key not in idict:
   idict[key] = [value]
  else:
   idict[key].append(value)
 return idict
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def orderDictTable(odict,ikey='Name',addNone=False,order_keys2=[0,1],
	order_keys3=[0,2,1],replace_error=True,serror='e_'):
 ndict = OrderedDict()
 for key in odict:
  ndict = appendKeyValueDict(ndict,ikey,key)
  for subkey in odict[key]:
   if isinstance(odict[key][subkey],dict):
    for subkey2 in odict[key][subkey]:
     if isinstance(odict[key][subkey][subkey2],dict):
      for subkey3 in odict[key][subkey][subkey2]:
       lkeys = [subkey,subkey2,subkey3]
       lkeys = tuple([lkeys[i] for i in order_keys3])
       nkey = '%s_%s_%s' % lkeys
       if replace_error:
        if serror in nkey:
         nkey = '%s%s' % (serror,nkey.replace(serror,''))
       ndict = appendKeyValueDict(ndict,nkey,odict[key][subkey][subkey2][subkey3],addNone=addNone)
     else:
      lkeys = [subkey,subkey2]
      lkeys = tuple([lkeys[i] for i in order_keys2])
      nkey = '%s_%s' % lkeys
      if replace_error:
       if serror in nkey:
        nkey = '%s%s' % (serror,nkey.replace(serror,''))
      ndict = appendKeyValueDict(ndict,nkey,odict[key][subkey][subkey2],addNone=addNone)
   else:
    ndict = appendKeyValueDict(ndict,subkey,odict[key][subkey],addNone=addNone)
 return ndict
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def updateFitsHeader(nfits,lheader):
 fits = pyfits.open(nfits,'update')
 hd   = fits[0].header
 for key,vcom in lheader:
  if isinstance(vcom,(list,tuple)):
   val,com = vcom
   hd.set(key,val,com)
  else:
   hd.set(key,vcom)
 fits.flush()
 fits.close()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def getNamesDataSetsHDF5(filename):
 import h5py
 def get_group(group):
  dsets = []
  for name, obj in list(group.items()):
   if isinstance(obj, h5py.Dataset):
    dsets.append(name)
   elif isinstance(obj, h5py.Group):
    dsets.append(get_group(obj))
   # other objects such as links are ignored
  return dsets
 f = h5py.File(filename, 'r') if isinstance(filename, str) else filename
 dsets = get_group(f)
 if isinstance(filename, str):
  f.close()
 return dsets
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def readTableHDF5(nfile, tname, get_attr=True, to_list=True, masked=False, **kwargs):
 import h5py
 f = h5py.File(nfile, 'r')
 tnames = f.keys()
 if not tname in tnames:
  print ('>>> WARNING: Table "%s" NOT present in HDF5 file [%s]' % (tname, ' | '.join(tnames)))
  f.close()
  return
 data = f[tname]
 if isinstance(data, h5py.highlevel.Dataset):
  table = Table(np.array(data), masked=masked)
  if masked:
   table = maskTable(table, **kwargs)
 elif isinstance(data, h5py.highlevel.Group):
  dtable = OrderedDict()
  for key in data.keys():
   if (data[key].dtype == np.dtype(object)) and to_list:
    # Convert to list since HDF5 objects keeps dtype.metadata info and avoid
    # bad argument to internal function when using sorting in astropy Tables
    dtable[key] = np.array(data[key].value.tolist(), dtype=np.object) 
   else:
    dtable[key] = data[key]
  table = Table(dtable, masked=masked)
  if masked:
   table = maskTable(table, **kwargs)
 if get_attr:
  for key in data.attrs:
   value = data.attrs[key]
   try:
    value = value.strip().replace('array','').replace('"','').replace('\\n', ' ') # replace('([','[')
    value = ast.literal_eval(value)
   except:
    pass
   table._meta[key] = value
 f.close()
 return table
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def readHDF5(nfile, path, table=False):
 data = OrderedDict()
 import h5py
 f = h5py.File(nfile, 'r')
 try:
  fdata = f[path]
 except:
  print ('WARNING: Path "%s" NOT found in file "%s"' % (path, nfile))
  f.close()
  return
 if isinstance(fdata, h5py.highlevel.Group):
  for key in fdata.keys():
   data[key] = fdata[key].value
 elif isinstance(fdata, h5py.highlevel.Dataset):
  adata = np.array(fdata)
  for key in adata.dtype.names:
   data[key] = adata[key]
 f.close()
 if table:
  data = Table(data)
 return data
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def getDictAttributeHDF5(attrs):
 dattrs = OrderedDict()
 for key in attrs:
  value = attrs[key]
  try:
   value = value.strip().replace('array','').replace('"','').replace('\\n', ' ')
   value = ast.literal_eval(value)
  except:
   pass
  dattrs[key] = value
 return dattrs
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def getAttributeHDF5(nfile, path=None):
 import h5py
 f = h5py.File(nfile, 'r')
 tnames = f.keys()
 try:
  attrs = f.attrs if path is None else f[path].attrs 
 except:
  print ('>>> WARNING: PATH "%s" NOT present in HDF5 file [%s]' % (path, ' | '.join(tnames)))
  f.close()
  return 
 dattrs = getDictAttributeHDF5(attrs)
 f.close()
 return dattrs
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def getHeaderFile(nfile, key=None, subkey=None, path=None, key_path='MDATA', ext=1, extname=None):
 if nfile.endswith(('hdf5', 'h5')):
  if path is None:
   hdr = getAttributeHDF5(nfile, path=path)
   if not key_path in hdr:
    print ('WARNING: Guess PATH "%s" NOT present. Provide a path' % key_path)
    return
   path = hdr[key_path]
  hdr = getAttributeHDF5(nfile, path=path)
  name = path
  tfile = 'HDF5'

 elif nfile.endswith(('fits', 'fit')):
  hdr = pyfits.getheader(nfile, ext=ext, extname=extname)
  name = ext if extname is None else extname
  tfile = 'FITS'

 if key is not None:
  if not key in hdr:
   print ('>>> Key "%s" NOT in header of object "%s" [%s]' % (key, name, tfile))
   return
  try:
   hdr = ast.literal_eval(hdr[key])
  except:
   hdr = hdr[key]

 if key is not None and isinstance(hdr, dict) and subkey is not None:
  if not subkey in hdr:
   print ('>>> Subkey "%s" NOT in dictionary' % subkey)
   return 
  try:
   hdr = ast.literal_eval(hdr[subkey])
  except:
   hdr = hdr[subkey]
 return hdr
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def simpleArrayType(narray, size=True):
 if narray.dtype.kind == 'f':
  return np.float
 elif narray.dtype.kind == 'i':
  return np.int
 elif narray.dtype.kind == 'S':
  if size:
   flatten_narray = np.hstack(narray.flat)
   size = len(max(flatten_narray, key=len))
   return 'S%s' % size
  else:
   return np.str
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def maskArrayValues(array, float_mask=None, integer_mask=None, 
	string_mask=['NAN','--','-'], mask_invalid=True, **kwargs):
 float_mask   = checkList(float_mask)
 integer_mask = checkList(integer_mask)
 string_mask  = checkList(string_mask)
 if not isinstance(array, np.ma.MaskedArray):
  array = np.ma.array(array, mask=False)
 if np.issubdtype(array.data.dtype, np.floating) and float_mask is not None:
  if mask_invalid:
   array = np.ma.masked_invalid(array)
  for mvalue in float_mask:
   array = np.ma.masked_values(array, mvalue, **kwargs)
 if np.issubdtype(array.data.dtype, np.integer) and integer_mask is not None:
  for mvalue in integer_mask:
   array = np.ma.masked_equal(array, mvalue)
 if np.issubdtype(array.data.dtype, str) and string_mask is not None:
  for mvalue in string_mask:
   array.mask[array == mvalue] = True
 return array
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def maskArraysValues(arrays, **kwargs):
 if not isinstance(arrays, (list, tuple, dict)):
  arrays = maskArrayValues(arrays, **kwargs)
 elif isinstance(arrays, dict):
  for key in arrays:
   arrays[key] = maskArrayValues(arrays[key], **kwargs)
 else:
  for i, array in enumerate(arrays):
   arrays[i] = maskArrayValues(array, **kwargs)
 return arrays
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def getAxesLim(value, dl=0.05):
 if isinstance(value, (list, tuple)):
  lmin, lmax = value
 else:
  lmin = np.nanmin(value)
  lmax = np.nanmax(value)
 seg = lmax - lmin
 return lmin - seg*dl, lmax + seg*dl
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def set_grid_plot(nx=1, ny=1, dx=5, dy=5, nfig=None, force=False, figsize=None, fig=None, 
	grid=None, left=0.07, right=0.98, wspace=0.2, bottom=0.15, top=0.92, hspace=0.3):
 if nfig is not None and not force:
  nx = nfig // ny
  nx += nfig % ny
 figsize = (nx*dx, ny*dy) if figsize is None else figsize
 if fig is None:
  fig = mpl.pyplot.figure(figsize=figsize)
 if grid is None:
  gs = gridspec.GridSpec(ny, nx)
  gs.update(left=left, right=right, wspace=wspace, bottom=bottom, top=top, hspace=hspace)
 else:
  gs = gridspec.GridSpecFromSubplotSpec(ny, nx, subplot_spec=grid, wspace=wspace, hspace=hspace)
 return fig, gs
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def desaturate_color(color, prop, alpha=None, to_hex=True):
 """Decrease the saturation channel of a color by some percent.

 Parameters
 ----------
 color : matplotlib color
     hex, rgb-tuple, or html color name
 prop : float
     saturation channel of color will be multiplied by this value
 alpha: float
 to_hex: bool

 Returns
 -------
 new_color : rgb tuple or Hex, but in Hex alpha does not apply
     desaturated color code in RGB tuple representation

 """
 # Check inputs
 if not 0 <= prop <= 1:
  raise ValueError("prop must be between 0 and 1")

 # Get rgb tuple rep
 rgb = mpl.colors.colorConverter.to_rgb(color)

 # Convert to hls
 h, l, s = colorsys.rgb_to_hls(*rgb)

 # Desaturate the saturation channel
 s *= prop

 # Convert back to rgb
 new_color = colorsys.hls_to_rgb(h, l, s)

 if alpha is not None:
  new_color = mpl.colors.colorConverter.to_rgba(new_color, alpha)

 # To Hex
 if to_hex:
  new_color = mpl.colors.rgb2hex(new_color)

 return new_color
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def desaturate_colors(colors, prop, alpha=None):
 return [desaturate_color(color, prop, alpha=alpha) for color in checkList(colors)]
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def color_luminance(rgb, method=0, cdark='k', cbright='w', cutoff=130):
 rgb = copy.deepcopy(rgb)
 if len(rgb) == 4:
  rgb = rgb[:3]
 rgb = np.atleast_1d(rgb)

 if rgb.max() <= 1. and method != 4:
  rgb *= 255

 if rgb.max() > 1. and method == 4:
  rgb /= 255

 if method == 0:
   luminance = np.dot([0.2126, 0.7152, 0.0722], rgb)

 elif method == 1:
   luminance = np.dot([0.299, 0.587, 0.114], rgb)

 elif method == 2:
   luminance = np.sqrt(np.dot([0.299, 0.587, 0.114], rgb**2))

 elif method == 3:
   luminance = np.sqrt(np.dot([0.241, 0.691, 0.068], rgb**2))

 elif method == 4:
  rgb[rgb <= 0.03928] = rgb[rgb <= 0.03928] / 12.92
  rgb[rgb > 0.03928]  = ((rgb[rgb > 0.03928] + 0.055) / 1.055)**2.4
  luminance = np.dot([0.2126, 0.7152, 0.0722], rgb)

 return luminance
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def facebackcolor(rgb, method=0, cdark='k', cbright='w', cutoff=130):
 luminance = color_luminance(rgb, method=method)
 dcutoff = {0: cutoff, 1: cutoff, 2: cutoff, 3: cutoff, 4: np.sqrt(1.05 * 0.05) - 0.05}
 color = cdark if luminance > dcutoff[method] else cbright
 return color
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def complementRGB(rgb):
 def hilo(a, b, c):
  if c < b: b, c = c, b
  if b < a: a, b = b, a
  if c < b: b, c = c, b
  return a + c
 r, g, b = rgb
 k = hilo(r, g, b)
 return tuple(k - u for u in (r, g, b))
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def get_text_legend_labels(labels, colors='black', fs=10):

 class textObject(object):
  def __init__(self, text, color, fontsize=10):
   self.text = text
   self.color = color
   self.fontsize = fontsize

 class textObjectHandler(object):
  def legend_artist(self, legend, orig_handle, fontsize, handlebox):
   x0, y0 = handlebox.xdescent, handlebox.ydescent
   width, height = handlebox.width, handlebox.height
   patch = mpl.text.Text(x=0, y=0, text=orig_handle.text, color=orig_handle.color, verticalalignment=u'baseline',
                          horizontalalignment=u'left', multialignment=None, fontsize=orig_handle.fontsize,
                          fontproperties=None, rotation=0, linespacing=None,
                          rotation_mode=None)
   handlebox.add_artist(patch)
   return patch

 lobj = []
 colors = 'black' if colors is None else colors
 dhandler = {}
 if isinstance(colors, str):
  colors = [colors]*len(labels)
 for lb,cl in zip(labels,colors):
  obj = textObject(lb, cl, fontsize=fs)
  lobj.append(obj)
  dhandler[obj] = textObjectHandler()
 return lobj, dhandler
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def addTextLegend(col1, col2, ax=None, colors='black', fontsize=10, ncol=1, numpoints=1, artist=True, **kwargs):
 if ax is None:
  ax = plt.gca()
 col1 = col1 if isinstance(col1, dict) else checkList(col1)
 col2 = col2 if isinstance(col2, dict) else checkList(col2)
 extra_legend_labels, dhandler = get_text_legend_labels(col1, colors, fs=fontsize)
 legend = ax.legend(extra_legend_labels, col2, handler_map=dhandler, ncol=ncol, numpoints=numpoints, fontsize=fontsize, **kwargs)
 if artist:
  ax.add_artist(legend)
 return ax
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def rainbow_text(x, y, strings, colors, ax=None, transAxes=True, horizontal=True, **kw):
    """
    Take a list of ``strings`` and ``colors`` and place them next to each
    other, with text strings[i] being shown in colors[i].

    This example shows how to do both vertical and horizontal text, and will
    pass all keyword arguments to plt.text, so you can set the font size,
    family, etc.

    The text will get added to the ``ax`` axes, if provided, otherwise the
    currently active axes will be used.
    """
    from matplotlib import transforms
    if ax is None:
        ax = plt.gca()
    t = ax.transAxes if transAxes else ax.transData
    canvas = ax.figure.canvas

    # horizontal version
    if horizontal:
        for s, c in zip(strings, colors):
            text = ax.text(x, y, " " + str(s) + " ", color=c, transform=t, **kw)
            text.draw(canvas.get_renderer())
            ex = text.get_window_extent()
            t = transforms.offset_copy(text._transform, x=ex.width, units='dots')

    # vertical version
    else:
        for s, c in zip(strings, colors):
            text = ax.text(x, y, " " + str(s) + " ", color=c, transform=t,
                       rotation=90, va='bottom', ha='center', **kw)
            text.draw(canvas.get_renderer())
            ex = text.get_window_extent()
            t = transforms.offset_copy(text._transform, y=ex.height, units='dots')
    return ax
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def multicolor_label(x, y, list_of_strings, list_of_colors, ax=None, horizontal=True, anchorpad=0, 
	align='center', frameon=False, sep=5, pad=0, ha='center', va='center', 
	bpad=0, loc=None, **kw):
 """this function creates axes labels with multiple colors
 ax specifies the axes object where the labels should be drawn
 list_of_strings is a list of all of the text items
 list_if_colors is a corresponding list of colors for the strings
 axis='x', 'y', or 'both' and specifies which label(s) should be drawn"""
 from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

 if ax is None:
  ax = plt.gca()

 dloc = {'upper right': 1, 'upper left': 2, 'lower left': 3, 'lower right': 4, 'right': 5,
 	'center left': 6, 'center right': 7, 'lower center': 8, 'upper center': 9, 'center': 10}

 dva = {'top': 'upper', 'bottom': 'lower', 'center': ''}
 sloc = '%s %s' % (dva[va], ha)
 loc = dloc[sloc.strip()] if loc is None else loc

 if horizontal:
  rotation = 0
 else:
  rotation = 90
  list_of_strings = list_of_strings[::-1]
  list_of_colors = list_of_colors[::-1]

 boxes = [TextArea(text, textprops=dict(color=color, ha='left', rotation=rotation, **kw)) 
             for text,color in zip(list_of_strings,list_of_colors) ]

 if horizontal:
  box = HPacker(children=boxes, align=align, pad=pad, sep=sep)

 else:
  box = VPacker(children=boxes, align=align, pad=pad, sep=sep)

 anchored_xbox = AnchoredOffsetbox(loc=loc, child=box, pad=anchorpad, frameon=frameon, bbox_to_anchor=(x, y),
                                   bbox_transform=ax.transAxes, borderpad=bpad)
 ax.add_artist(anchored_xbox)
 return ax
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def addCustomLineLegend(labels, colors, lstyle='-', marker=None, ax=None, artist=True, 
	lw=1, dline={}, text_colors=None, dtext={}, **kwargs):
 from matplotlib.lines import Line2D
 if ax is None:
  ax = plt.gca()
 colors = checkList(colors)
 labels = checkList(labels)
 lstyle = checkList(lstyle)
 marker = checkList(marker)
 if len(lstyle) < len(labels):
  lstyle = [lstyle[0]] * len(labels)
 if len(colors) < len(labels):
  colors = [colors[0]] * len(labels)
 if marker is None:
  marker = [None] * len(labels)
 if len(marker) < len(labels):
  marker = [marker[0]] * len(labels)
 handles = []
 for cl, lb, lsty, mk in zip(colors, labels, lstyle, marker):
  handles.append(Line2D((0,1),(0,0), color=cl, marker=mk, linestyle=lsty, lw=lw, **dline))
 legend = ax.legend(handles, labels, **kwargs)
 if text_colors is not None:
  if isinstance(text_colors, bool):
   text_colors = colors
   for cl, text in zip(cycle(text_colors), legend.get_texts()):
    text.set_color(cl)
 if len(dtext) > 0:
  mpl.pyplot.setp(legend.get_texts(), **dtext)
 if artist:
  ax.add_artist(legend)
 return legend
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def addCustomPatchLegend(labels, colors, lstyle='-', ax=None, artist=True, 
	hatchs=None, lw=0.5, dpatch={}, text_colors=None, ecolors='k', 
	alphas=1, **kwargs):
 from matplotlib.patches import Patch
 if ax is None:
  ax = plt.gca()
 colors = checkList(colors)
 labels = checkList(labels)
 lstyle = checkList(lstyle)
 ecolor = checkList(ecolors)
 hatchs = checkList(hatchs)
 alphas = checkList(alphas)
 if hatchs is None:
  hatchs = [None] * len(labels)
 if alphas is None:
  alphas = [1] * len(labels)
 if ecolor is None:
  ecolor = colors
 if len(lstyle) < len(labels):
  lstyle = [lstyle[0]] * len(labels)
 if len(colors) < len(labels):
  colors = [colors[0]] * len(labels)
 if len(ecolors) < len(labels):
  ecolors = [ecolors[0]] * len(labels)
 if len(hatchs) < len(labels):
  hatchs = [hatchs[0]] * len(labels)
 if len(alphas) < len(labels):
  alphas = [alphas[0]] * len(labels)
 handles = []
 for cl, ec, lb, lsty, ht, alpha in zip(colors, ecolors, labels, lstyle, hatchs, alphas):
  handles.append(Patch(facecolor=cl, edgecolor=ec, linestyle=lsty, lw=lw, hatch=ht, alpha=alpha, **dpatch))
 legend = ax.legend(handles, labels, **kwargs)
 if text_colors is not None:
  if isinstance(text_colors, bool):
   text_colors = colors
   for cl, text in zip(cycle(text_colors), legend.get_texts()):
    text.set_color(cl)
 if artist:
  ax.add_artist(legend)
 return legend
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def mplText2latex(strings, single=True):
 strings = checkList(strings) if not isinstance(strings, np.ndarray) else strings
 strings = [r"%s" % st for st in strings]
 if len(strings) == 1 and single:
  return strings[0]
 else:
  return strings
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def get_axe_position(axe, fig=None, bbox=None):
 if isinstance(axe, gridspec.GridSpec):
  x0, x1, y0, y1 = axe.left, axe.right, axe.bottom, axe.top
 elif isinstance(axe, gridspec.GridSpecFromSubplotSpec):
  #figBottoms, figTops, figLefts, figRights = axe.get_grid_positions(fig)
  #x0, x1, y0, y1 = min(figLefts), max(figRights), min(figBottoms), max(figTops)
  spar = axe.get_subplot_params(fig)
  x0, x1, y0, y1 = spar.left, spar.right, spar.bottom, spar.top
 else:
  bbox = axe.get_position(fig) if bbox is None else bbox
  x0, x1, y0, y1 = bbox.x0, bbox.x1, bbox.y0, bbox.y1
 return x0, x1, y0, y1
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def get_axe_colorbar(axe, fig=None, position='right', pad=5, shrink=1.0, 
	aspect=0.05, rpad=None, rpad2=0.0, pt=72., xf=None, yf=None):
  position = position.lower().strip()

  # If the axes is an AxesImage object (imshow), need to draw the object to update the axes position
  if fig is not None:
   fig.canvas.draw()

  x0, x1, y0, y1 = get_axe_position(axe)
  dx = x1 - x0
  dy = y1 - y0

  pad /= pt

  if position == 'left':
   pad = pad if rpad is None else dx * rpad
   cx1 = x0 - pad
   cx0 = cx1 - dx * aspect
   cy0 = y0 + dy / 2. - dy * shrink / 2. + dy * rpad2
   cy1 = cy0 + dy * shrink

  if position == 'right':
   pad = pad if rpad is None else dx * rpad
   cx0 = x1 + pad
   cx1 = cx0 + dx * aspect
   cy0 = y0 + dy / 2. - dy * shrink / 2. + dy * rpad2
   cy1 = cy0 + dy * shrink

  if position == 'top':
   pad = pad if rpad is None else dy * rpad
   cy0 = y1 + pad
   cy1 = cy0 + dy * aspect
   cx0 = x0 + dx / 2. - dx * shrink / 2. + dx * rpad2
   cx1 = cx0 + dx * shrink

  if position == 'bottom':
   pad = pad if rpad is None else dy * rpad
   cy0 = y0 - pad
   cy1 = cy0 - dy * aspect
   cx0 = x0 + dx / 2. - dx * shrink / 2. + dx * rapd2
   cx1 = cx0 + dx * shrink

  ndx = cx1 - cx0
  ndy = cy1 - cy0
  nx0 = cx0
  ny0 = cy0

  if xf is not None:
   nx0 = xf - ndx / 2.

  if yf is not None:
   ny0 = yf - ndy / 2.

  cax = [nx0, ny0, ndx, ndy]

  if fig is not None:
   cax = fig.add_axes(cax)

  return cax
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def add_colorbar_list_colors(ax, colors, bounds, position='right', size="3%", pad="2%", 
	caxes={}, ticks=None, ticklabels=None, center=True, dax=None, **kwargs):
 from mpl_toolkits.axes_grid1 import make_axes_locatable
 cmap = mpl.colors.ListedColormap(colors)
 ncolors = len(colors)
 # Norm goes from 0 to ncolors - 1
 norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
 divider = make_axes_locatable(ax)
 cax = divider.append_axes(position, size, pad=pad, **caxes)
 #cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, **kwargs)
 cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, **kwargs)
 if ticklabels is None:
  ticklabels = bounds[1:] if center else bounds
 if ticks is None:
  if dax is None:
   dax = (max(bounds) - min(bounds)) / float(ncolors) / 2.0 if center else 0.0
  ticks = np.linspace(min(bounds), max(bounds), ncolors + 1) - dax
 ticks = np.interp(ticklabels, bounds, ticks)
 cbar.set_ticks(ticks)
 cbar.set_ticklabels(ticklabels)
 return cax, cbar
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def find_renderer(fig):

 if hasattr(fig.canvas, "get_renderer"):
  #Some backends, such as TkAgg, have the get_renderer method, which #makes this easy
  renderer = fig.canvas.get_renderer()
 else:
  #Other backends do not have the get_renderer method, so we have a work
  #around to find the renderer.  Print the figure to a temporary file
  #object, and then grab the renderer that was used.
  #Got from the matplotlib backend_bases.py
  #print_figure() method.
  import io
  fig.canvas.print_pdf(io.BytesIO())
  renderer = fig._cachedRenderer
 return(renderer)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def addArrowsLabel(fig, ax, x, y, s, force=True, fpad=0.2, arrow_height=0.2, darrow={},
        invert=False, horizontal=True, shrink=1.0, both=False, **kwargs):
 from matplotlib.patches import FancyArrowPatch
 if not isinstance(ax, (int, float)):
  x0, x1, y0, y1 = get_axe_position(ax, fig=fig)
  if force:
   if horizontal:
    x = (x1 + x0) / 2.0
   else:
    y = (y1 + y0) / 2.0
 else:
  if horizontal:
   x0 = 1. - ax
   x1 = ax
  else:
   y0 = 1. - ax
   y1 = ax

 arrow_height /= 2.
 rotation = 0. if horizontal else 90.
 renderer = find_renderer(fig)
 text     = fig.text(x, y, s, **updateDictDefault(kwargs, dict(rotation=rotation)))
 bb       = text.get_window_extent(renderer=renderer)
 bbox     = bb.transformed(fig.transFigure.inverted())
 mscale   = bb.width * arrow_height if horizontal else bb.height * arrow_height
 bwidth   = abs(bbox.x1 - bbox.x0) if horizontal else abs(bbox.y1 - bbox.y0)
 pad      = bwidth * fpad

 if horizontal:
  y_arrow    = abs(bbox.y1 + bbox.y0) / 2.0
  lwidth     = (bbox.x0 - pad - x0) * shrink
  rwidth     = (x1 - bbox.x1 - pad) * shrink
  left_posA  = ((x0 + (bbox.x0 - pad)) / 2.0 - lwidth / 2.0, y_arrow)
  left_posB  = ((x0 + (bbox.x0 - pad)) / 2.0 + lwidth / 2.0, y_arrow)
  right_posA = ((x1 + (bbox.x1 + pad)) / 2.0 - rwidth / 2.0, y_arrow)
  right_posB = ((x1 + (bbox.x1 + pad)) / 2.0 + rwidth / 2.0, y_arrow)
 else:
  x_arrow    = abs(bbox.x1 + bbox.x0) / 2.0
  lwidth     = (bbox.y0 - pad - y0) * shrink
  rwidth     = (y1 - bbox.y1 - pad) * shrink
  left_posA  = (x_arrow, (y0 + (bbox.y0 - pad)) / 2.0 - lwidth / 2.0)
  left_posB  = (x_arrow, (y0 + (bbox.y0 - pad)) / 2.0 + lwidth / 2.0)
  right_posA = (x_arrow, (y1 + (bbox.y1 + pad)) / 2.0 - rwidth / 2.0)
  right_posB = (x_arrow, (y1 + (bbox.y1 + pad)) / 2.0 + rwidth / 2.0)

 if horizontal:
  arrowstyle = '<-' if invert else '->'
  if invert:
   left_arrowstyle  = arrowstyle
   right_arrowstyle = arrowstyle if both else '-'
  else:
   left_arrowstyle  = arrowstyle if both else '-'
   right_arrowstyle = arrowstyle
 else:
  arrowstyle = '->' if invert else '<-'
  if invert:
   left_arrowstyle  = arrowstyle if both else '-'
   right_arrowstyle = arrowstyle
  else:
   left_arrowstyle  = arrowstyle
   right_arrowstyle = arrowstyle if both else '-'

 darrow_left  = updateDictDefault(darrow, dict(lw=1.5, color='k', mutation_scale=mscale, arrowstyle=left_arrowstyle))
 darrow_right = updateDictDefault(darrow, dict(lw=1.5, color='k', mutation_scale=mscale, arrowstyle=right_arrowstyle))

 left_arrow  = FancyArrowPatch(left_posA, left_posB, transform=fig.transFigure, **darrow_left)
 right_arrow = FancyArrowPatch(right_posA, right_posB, transform=fig.transFigure, **darrow_right)

 fig.patches.extend([left_arrow, right_arrow])
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def setTicksLim(ax, xticklabels=True, yticklabels=True, left_idx=None, right_idx=None, bottom_idx=None, 
	top_idx=None, ylim=None, xlim=None, grid=False, legend=False, loc=2, ncol=1, legendfs=9, saxtitle=None, 
	axtitle=True, sxlabel=None, xlabel=True, sylabel=None, ylabel=True, ylabelp='left', artist=False, 
	framealpha=1.0, xticks=None, yticks=None, dxticklabel={}, dyticklabel={}, xhide_id=None, yhide_id=None, 
	dtitle={}, dxlabel={}, dylabel={}, xhide=True, yhide=True, astitle=True, xtit=0.95, ytit=0.92, 
	hatit='right', leg_prop={}, sxticklabels=None, syticklabels=None, fig=None, xgrid=True, 
	ygrid=True, legend_invisible=False, axisbelow=True, xminorticks=False, yminorticks=False, 
	minorticks=False, ldtext=None, xmajor_loc=None, xminor_loc=None, ymajor_loc=None, 
	yminor_loc=None, xticks_major=True, yticks_major=True):
 if fig is not None:
  # Some objects like yticklabels with seaborn are not updated and give empty labels if the text of the labels 
  # are used: [xtick.get_text() for xtick in ax.get_xticklabels()] and need to be updated with fig.canvas.draw()
  # But plt.setp() is used for changing the properties of the labels this is not needed
  fig.canvas.draw() 
 if xminorticks or yminorticks or minorticks:
  ax.minorticks_on()
  if not yminorticks and not minorticks:
   ax.tick_params(axis='y', which='minor', right='off', left='off')
  if not xminorticks and not minorticks:
   ax.tick_params(axis='x', which='minor', bottom='off', top='off')
 if xmajor_loc is not None:
  if isinstance(xmajor_loc, str):
   ax.xaxis.set_major_locator(FormatStrFormatter(xmajor_loc))
  else:
   ax.xaxis.set_major_locator(MultipleLocator(xmajor_loc))
 if xminor_loc is not None:
  if isinstance(xminor_loc, str):
   ax.xaxis.set_minor_locator(FormatStrFormatter(xminor_loc))
  else:
   ax.xaxis.set_minor_locator(MultipleLocator(xminor_loc))
 if ymajor_loc is not None:
  if isinstance(ymajor_loc, str):
   ax.yaxis.set_major_locator(FormatStrFormatter(ymajor_loc))
  else:
   ax.yaxis.set_major_locator(MultipleLocator(ymajor_loc))
 if yminor_loc is not None:
  if isinstance(yminor_loc, str):
   ax.yaxis.set_minor_locator(FormatStrFormatter(yminor_loc))
  else:
   ax.yaxis.set_minor_locator(MultipleLocator(yminor_loc))
 if axisbelow:
  ax.set_axisbelow(True)
 ax.grid(grid)
 if not xgrid:
  ax.xaxis.grid(False)
 if not ygrid:
  ax.yaxis.grid(False)
 if xticks is not None:
  ax.set_xticks(xticks)
 if yticks is not None:
  ax.set_yticks(yticks)
 if sxticklabels is not None:
  ax.set_xticklabels(sxticklabels)
 xticks = ax.get_xticklabels()
 if len(dxticklabel) > 0:
  mpl.pyplot.setp(xticks, **dxticklabel)
 #ax.xaxis.set_tick_params(**dxticklabel) # Different keywords: fontsize/labelsize, ...
 #ax.set_xticklabels(xticks, **dxticklabel)
 if left_idx is not None:
  for idx in checkList(left_idx):
   xticks[idx].set_horizontalalignment('left')
 if right_idx is not None:
  for idx in checkList(right_idx):
   xticks[idx].set_horizontalalignment('right')
 if xhide_id is not None and xhide:
  for idx in checkList(xhide_id):
   xticks[idx].set_visible(False)
 if not xticklabels:
  ax.tick_params(axis='x', which='major', labelbottom='off')
 if not xticks_major:
  ax.tick_params(axis='x', which='major', bottom='off', top='off')
 if not yticks_major:
  ax.tick_params(axis='y', which='major', left='off', right='off')
 if syticklabels is not None:
  ax.set_yticklabels(syticklabels)
 yticks = ax.get_yticklabels()
 if len(dyticklabel) > 0:
  mpl.pyplot.setp(yticks, **dyticklabel)
 if top_idx is not None:
  for idx in checkList(top_idx):
   yticks[idx].set_verticalalignment('top')
 if bottom_idx is not None:
  for idx in checkList(bottom_idx):
   yticks[idx].set_verticalalignment('bottom')
 if yhide_id is not None and yhide:
  for idx in checkList(yhide_id):
   yticks[idx].set_visible(False)
 if not yticklabels:
  ax.tick_params(axis='y', which='major', labelleft='off')
 #ax.xaxis.get_label().set_horizontalalignment('left')
 if xlim is not None:
  ax.set_xlim(xlim)
 if ylim is not None:
  ax.set_ylim(ylim)
 if legend:
  lg = ax.legend(**updateDictDefault(leg_prop, dict(loc=loc, ncol=ncol, fontsize=legendfs, framealpha=framealpha)))
  if artist:
   ax.add_artist(lg)
 if legend_invisible and ax.legend_ is not None:
  # This sets the properties of all inherited legends
  #ax.legend_.remove()
  ax.legend_.set_visible(False)
 if saxtitle is not None:
  if astitle:
   ax.set_title(saxtitle, **dtitle)
  else:
   ax.text(xtit, ytit, saxtitle, transform=ax.transAxes, **updateDictDefault(dtitle, dict(ha=hatit)))
  if not axtitle:
   ax.title.set_visible(False)
 if not xlabel:
  ax.xaxis.label.set_visible(False)
 if sxlabel is not None:
  ax.set_xlabel(mplText2latex(sxlabel, single=True), **dxlabel)
 if not ylabel:
  ax.yaxis.label.set_visible(False)
 if sylabel is not None:
  ax.set_ylabel(mplText2latex(sylabel, single=True), **dylabel)
  if ylabelp is not None:
   ax.yaxis.set_label_position(ylabelp)
 if ldtext is not None and len(ldtext) > 0:
  for dtext in ldtext:
   ax.text(transform=ax.transAxes, **dtext)
 return ax
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class kde2d(object):
 def __init__(self, x, y, bw='scott', gridsize=100, cut=3, clip=None, clean_infinite=True, **kwargs):
  """
  bw : {'scott' | 'silverman' | scalar | pair of scalars }, optional
      Name of reference method to determine kernel size, scalar factor,
      or scalar for each dimension of the bivariate plot.
  gridsize : int, optional
      Number of discrete points in the evaluation grid.
  cut : scalar, optional
      Draw the estimate to cut * bw from the extreme data points.
  clip : pair of scalars, or pair of pair of scalars, optional
      Lower and upper bounds for datapoints used to fit KDE. Can provide
      a pair of (low, high) bounds for bivariate plots.
  """

  self.bw = bw
  self.gridsize = gridsize
  self.cut = cut

  self.set_clip(clip)
  self.set_data(x, y, clean_infinite=clean_infinite)
  self.kde()
  self.func_contours(**kwargs)

 def set_clip(self, clip=None):
  if clip is None:
   clip = [(-np.inf, np.inf), (-np.inf, np.inf)]
  elif np.ndim(clip) == 1:
   clip = [clip, clip]
  self.clip = clip

 def set_data(self, x, y, clip=None, clean_infinite=True):
  self._x = x
  self._y = y
  self.x = x
  self.y = y

  if clean_infinite:
   idf = np.isfinite(self._x) & np.isfinite(self._y)
   self.x = self._x[idf]
   self.y = self._y[idf]

 def kde(self):
  import seaborn as sn

  try:
   xx, yy, z = sn.distributions._statsmodels_bivariate_kde(self.x, self.y, self.bw, self.gridsize, self.cut, self.clip)
  except:
   xx, yy, z = sn.distributions._scipy_bivariate_kde(self.x, self.y, self.bw, self.gridsize, self.cut, self.clip)

  self.xx = xx
  self.yy = yy
  self._z = z
  self.z = z / z.sum()

 def func_contours(self, t=None, n=1000):
  if t is None:
   t = np.linspace(np.nanmin(self.z), np.nanmax(self.z), n)
  self.t = t
  self.it = ((self.z >= t[:, None, None]) * self.z).sum(axis=(1,2))
  self.f = sci.interp1d(self.it, self.t)

 def get_levels(self, levels=10):
  if isinstance(levels, np.int):
   levels = np.linspace(np.nanmax(self.it), np.nanmin(self.it), levels)
  levels = np.atleast_1d(levels)
  return levels

 def contours(self, levels=10):
  levels = self.get_levels(levels)
  return self.f(levels)

 def ax_contours(self, ax, filled=False, levels=10, extent=None, fill_lowest=False, cbar=False, 
	cbar_ax=None, cbar_kws=None, fmt='%.1f', invert_cbar=True, **kwargs):
  levels = self.get_levels(levels)
  contours = self.contours(levels)
  contour_func = ax.contourf if filled else ax.contour
  if extent is None:
   extent = [np.nanmin(self.xx), np.nanmax(self.xx), np.nanmin(self.yy), np.nanmax(self.yy)]
  cset = contour_func(self.z, contours, extent=extent, **kwargs)
  if filled and not fill_lowest:
   cset.collections[0].set_alpha(0)
 
  self.dax = {'ax': ax, 'contours': cset, 'levels': levels, 'contours': contours}

  if cbar:
   cbar_kws = {} if cbar_kws is None else cbar_kws
   cbar = ax.figure.colorbar(cset, cbar_ax, ax, **cbar_kws)
   if fmt is not None:
    levels = [fmt % lev for lev in levels]
    self.dax['format_levels'] = levels
   cbar.set_ticklabels(levels)
   if invert_cbar:
    if len(cbar.ax.get_yticks()) > 0:
     cbar.ax.invert_yaxis()
    elif len(cbar.ax.get_xticks()) > 0:
     cbar.ax.invert_xaxis()

   self.dax['colorbar'] = cbar

  return ax
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def inverse_interp(y, x, dy, kind='linear', bounds_error=False, fill_value=np.nan):
 '''Estimate the x values (dx) where dy occurs. 
    "y" should have the same dimensions as x in the last axis (-1)'''
 dx = []

 if y.ndim == 1:
  dx = sci.interp1d(y, x, kind=kind, bounds_error=bounds_error, fill_value=fill_value)(dy)

 elif y.ndim == 2:
  for i in range(y.shape[0]):
   axis_0 = sci.interp1d(y[i,:], x, kind=kind, bounds_error=bounds_error, fill_value=fill_value)(dy)
   axis_0 = axis_0 if len(axis_0) > 1 else axis_0[0]
   dx.append(axis_0)

 elif y.ndim == 3:
  for i in range(y.shape[0]):
   axis_0 = []
   for j in range(y.shape[1]):
    axis_1 = sci.interp1d(y[i,j,:], x, kind=kind, bounds_error=bounds_error, fill_value=fill_value)(dy)
    axis_1 = axis_1 if len(axis_1) > 1 else axis_1[0]
    axis_0.append(axis_1)
   dx.append(axis_0)

 return np.atleast_1d(dx)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def interp(x, y, dx, kind='linear', bounds_error=False, fill_value=np.nan):
 '''"y" should have the same dimensions as x in the last axis (-1)'''
 dy = []

 dx = np.atleast_1d(dx)
 x  = np.atleast_1d(x)
 if dx.ndim == 1 and y.ndim > 1:
  dx = np.tile(dx, (y.shape[0], 1))
 if x.ndim == 1 and y.ndim > 1:
  x = np.tile(x, (y.shape[0], 1))

 if y.ndim == 1:
  dy = sci.interp1d(x, y, kind=kind, bounds_error=bounds_error, fill_value=fill_value)(dx)

 elif y.ndim == 2:
  for i in range(y.shape[0]):
   axis_0 = sci.interp1d(x[i], y[i,:], kind=kind, bounds_error=bounds_error, fill_value=fill_value)(dx[i])
   axis_0 = np.atleast_1d(axis_0)
   axis_0 = axis_0 if len(axis_0) > 1 else axis_0[0]
   dy.append(axis_0)

 elif y.ndim == 3:
  for i in range(y.shape[0]):
   axis_0 = []
   for j in range(y.shape[1]):
    axis_1 = sci.interp1d(x[i], y[i,j,:], kind=kind, bounds_error=bounds_error, fill_value=fill_value)(dx[i])
    axis_1 = axis_1 if len(axis_1) > 1 else axis_1[0]
    axis_0.append(axis_1)
   dy.append(axis_0)

 return np.atleast_1d(dy)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def interpMaskedArray(x, y, dx, kind='linear', bounds_error=False, fill_value=np.nan, masked=True, inverse=False):
 # scipy interp1d does not handle masked arrays, so better to fill all of them with the same initial mask value
 x, y, dx = [fillMaskedArray(item, fill_value=fill_value) for item in [x, y, dx]]

 func_interp = inverse_interp if inverse else interp
 inv = func_interp(x, y, dx, kind=kind, bounds_error=bounds_error, fill_value=fill_value)

 if masked:
  if np.isnan(fill_value):
   inv = np.ma.array(inv, mask=~np.isfinite(inv))
  else:
   inv = np.ma.array(inv, mask=(inv == fill_value))

 return inv
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def maskCategoricalDataFrame(data, x, y, hue=None, z=None, nmin=None, percentile=None, hue_order=None, 
	order=None, z_order=None, low_per=None, upp_per=None, dict_per=None, copy=True):
 if percentile is None and dict_per is None and low_per is None and upp_per is None and nmin is None:
  return data
 if copy:
  data = data.copy()
 if hue_order is None:
  hue_order = data[hue].unique() if hue in data else [None]
 if order is None:
  order = data[x].unique()
 if z_order is None:
  z_order = data[z].unique() if z in data else [None]
 if percentile is not None:
  if isinstance(percentile, (list, tuple)):
   low_per, upp_per = percentile
  else:
   low_per = percentile
   upp_per = 100. - percentile
 for ihue in hue_order:
  for ix in order:
   for iz in z_order:
    if ihue is not None:
     if iz is not None:
      idxh = (data[hue] == ihue) & (data[x] == ix) & (data[z] == iz)
     else:
      idxh = (data[hue] == ihue) & (data[x] == ix)
    else:
     idxh = data[x] == ix
    if nmin is not None and idxh.sum() < nmin:
     data[y][idxh] = np.nan
    ilow_per = low_per
    iupp_per = upp_per
    if dict_per is not None:
     dkey = ix
     if ihue is not None:
      dkey = (ix, ihue, iz) if iz is not None else (ix, ihue)
     if dkey in dict_per:
      ilow_per = dict_per[dkey][0] if dict_per[dkey][0] is not None else ilow_per
      iupp_per = dict_per[dkey][1] if dict_per[dkey][1] is not None else iupp_per
    if ilow_per is not None:
     idf = np.isfinite(data[y][idxh])
     if np.any(idf):
      vlow_per = np.percentile(data[y][idxh][idf], ilow_per)
      idp = (data[y] < vlow_per) & idxh
      data[y][idp] = np.nan
    if iupp_per is not None:
     # Need to get again finite values in case some have been masked by low_per
     idf = np.isfinite(data[y][idxh])
     if np.any(idf):
      vupp_per = np.percentile(data[y][idxh][idf], iupp_per)
      idp = (data[y] > vupp_per) & idxh
      data[y][idp] = np.nan
 return data
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def flux2Lum(flux,distance,error=None):
 lum = flux*4.*np.pi*np.power(distance,2.)
 if error is not None:
  error = error*4.*np.pi*np.power(distance,2.)
 return lum,error
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def magnitudeErrorFlux(flux,eflux):
 cte = 2.5/np.log(10.)
 mag_error = cte*eflux/flux
 return mag_error
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def getWaveShift(wave,redshift=None,v0=None):
 if redshift is not None and v0 is not None:
  redshift += v0 / CST.c*1.0e5
 if redshift is None and v0 is not None:
  redshift = v0 / CST.c*1.0e5
 if redshift is not None:
  wave = wave/(1.0 + redshift)
 return wave
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def SetWaveResolution(data,res_ini,res_end,axis=0):
 sigma_instrumental = res_ini/(2.*np.sqrt(2.*np.log(2.)))
 sigma_end = res_end/(2.*np.sqrt(2.*np.log(2.)))
 sigma_dif = np.sqrt(np.power(sigma_end,2.) - np.power(sigma_instrumental,2.))
 return gaussian_filter1d(data,sigma_dif,axis=axis)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def setKinematics(wave,data,v0,vd_ini,vd_end):
 from pystarlight.util.velocity_fix import SpectraVelocityFixer
 vfix = SpectraVelocityFixer(wave,v0,vd_ini)
 data = vfix.fix(data,vd_end)
 return data
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def Interp(wave,flux,lresam,intNumpy=True,extrap=True):
 if intNumpy:
  flux = np.interp(lresam,wave,flux)
 else:
  from pystarlight.util.StarlightUtils import ReSamplingMatrixNonUniform
  RSM = ReSamplingMatrixNonUniform(wave,lresam,extrap=extrap)
  flux = np.dot(RSM,flux)
 return flux
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def derivative(x, y, dx=None, der=1, norm=None, bspline=True, s=0, k=3):
 if dx is not None:
  nx = np.arange(x.min(), x.max() + dx/2., dx)
  y = np.interp(nx, x, y)
  x = nx
 if bspline:
  sxy = sci.splrep(x, y, s=s, k=k)
  dydx = sci.splev(x, sxy, der=der)
 else:
  sxy = sci.UnivariateSpline(x, y, k=k, s=s)
  dydx = sxy.derivative(der)(x)
 if norm is not None:
  norm_dydx = np.trapz(dydx, x)
  dydx *= norm / norm_dydx
 return dydx
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class LinearRegression(object):
 def __init__(self, x, y, ey=None, weights=None, method='RLM', alpha=0.05, constant=True, 
	weights_diff=0.85, order=1, clean_infinite=False, clean_masked=True, npoints=100,
	ci=None, n_boot=1000, units=None, **kwargs):
  self._x = x
  self._y = y
  self._ey = ey
  self.weights = weights
  self.weights_diff = weights_diff
  self.alpha = alpha
  self.order = order
  self.npoints = npoints
  self.n_boot = n_boot
  self.units = units
  self.ci = ci
  self.models = {}
  self.empty = False

  self.X = None
  self.a = None
  self.b = None
  self.ea = None
  self.eb = None

  self.methods = ['ols', 'wls', 'rlm', 'wrlm', 'poly', 'ransac', 'br']

  self.fit_methods(method, constant=constant, clean_infinite=clean_infinite, clean_masked=clean_masked, **kwargs)

 def set_xy(self, x=None, y=None, ey=None, weights=None, method=None, constant=True, clean_infinite=False, clean_masked=True, verbose=True):
  self.x  = self._x.copy() if x is None else x
  self.y  = self._y.copy() if y is None else y
  self.ey = ey if ey is not None else (self._ey.copy() if self._ey is not None else None)
  if clean_masked:
   self.clean_masked()
  if clean_infinite:
   self.clean_infinite()
  self.check_data(verbose=verbose)
  if self.empty:
   return

  self.grid = np.linspace(self.x.min(), self.x.max(), self.npoints)

  if method is not None and (method.lower() in ['ransac', 'br']):
   self.X = self.x.reshape(self.x.size, 1)
  elif method is not None and method.lower() == 'poly':
   self.X = self.x
  else:
   if constant:
    self.X = np.c_[np.ones(self.x.size), self.x]
   else:
    self.X = np.c_[np.zeros(self.x.size), self.x]
  self.set_weights(weights=weights)

 def check_data(self, verbose=True):
  self.empty = False
  if self.x.size < 2 or self.y.size < 2:
   if verbose:
    print ('>>> WARNING: "x" (%i) and "y" (%i) have to have each at least 2 elements!' % (self.x.size, self.y.size))
   self.empty = True

 def clean_infinite(self):
  idf = np.isfinite(self.x) & np.isfinite(self.y)
  self.x = self.x[idf]
  self.y = self.y[idf]
  if self.ey is not None:
   self.ey = self.ey[idf]

 def clean_masked(self):
  if isinstance(self.x, np.ma.MaskedArray) and isinstance(self.y, np.ma.MaskedArray):
   idm = (~self.x.mask) & (~self.y.mask)
  elif isinstance(self.x, np.ma.MaskedArray) and not isinstance(self.y, np.ma.MaskedArray):
   idm = ~self.x.mask
  elif not isinstance(self.x, np.ma.MaskedArray) and isinstance(self.y, np.ma.MaskedArray):
   idm = ~self.y.mask
  else:
   idm = None
  if idm is not None:
   self.x = self.x[idm]
   self.y = self.y[idm]

 def set_weights(self, ey=None, weights=None):
  ey = self.ey if ey is None else ey
  if weights is not None:
   self.weights = weights
  if ey is not None and weights is None:
   self.weights = 1. / np.power(self.ey, 2.)
  if self.weights is None:
   self.weights = np.ones(self.x.shape)

 def fit_method(self, method, constant=True, clean_infinite=False, clean_masked=True, verbose=True, **kwargs):
  self.set_xy(method=method, constant=constant, clean_infinite=clean_infinite, clean_masked=clean_masked, verbose=verbose)
  if self.empty:
   return

  if method.lower() == 'ols':
   self.ols(model=method, **kwargs)

  elif method.lower() == 'wls':
   self.wls(model=method, **kwargs)

  elif method.lower() == 'rlm':
   self.rlm(model=method, **kwargs)

  elif method.lower() == 'wrlm':
   self.wrlm(model=method, **kwargs)

  elif method.lower() == 'poly':
   self.poly(model=method, **kwargs)

  elif method.lower() == 'ransac':
   self.ransac(model=method, **kwargs)

  elif method.lower() == 'br':
   self.bayesian_ridge(model=method, **kwargs)

  else:
   print('>>> WARNING: Method "%s" NOT available [%s]' % (method, ' | '.join(self.methods)))

 def fit_methods(self, methods=None, constant=True, clean_infinite=False, clean_masked=True, fit_all=False, verbose=True, **kwargs):
  methods = checkList(methods)
  if fit_all:
   methods = self.methods
  if methods is None:
   return
  for method in methods:
   self.fit_method(method, constant=constant, clean_infinite=clean_infinite, clean_masked=clean_masked, verbose=verbose, **kwargs)
   if self.empty:
    continue
   self.bootstrap(method)
   self.fitted(method)
   self.fill(method)

 def bootstrap(self, model):
  if self.ci is None or self.n_boot is None or not 'reg_func' in self.models[model]:
   return
  try:
   import seaborn as sn
  except:
   print ('>>> WARNING: seaborn needed for bootstrap!')
   return
  yhat_boots = sn.algorithms.bootstrap(self.X, self.y, func=self.models[model]['reg_func'], n_boot=self.n_boot, units=self.units)
  err_bands = sn.utils.ci(yhat_boots, self.ci, axis=0)
  self.models[model]['err_bands'] = err_bands
  self.models[model]['err_lower'] = err_bands[0]
  self.models[model]['err_upper'] = err_bands[1]

 def fill(self, model):
  self.err_lower = None
  self.err_upper = None
  if model in self.models:
   if 'interp' in self.models[model]:
    self.b  = self.models[model]['interp']   
   if 'coef' in self.models[model]:
    self.a  = self.models[model]['coef']     
   if 'e_interp' in self.models[model]:
    self.eb = self.models[model]['e_interp'] 
   if 'e_coef' in self.models[model]:
    self.ea = self.models[model]['e_coef']
   if 'err_lower' in self.models[model]:
    self.err_lower = self.models[model]['err_lower'] 
   if 'err_upper' in self.models[model]:
    self.err_upper = self.models[model]['err_upper'] 

 def fitted(self, model):
  if model in self.models:
   self.models[model]['fitted'] = (self.models[model]['coef'] * self.x + self.models[model]['interp']).squeeze()

 def ols(self, alpha=None, model='OLS', **kwargs):
  from statsmodels.sandbox.regression.predstd import wls_prediction_std
  from statsmodels.stats.outliers_influence import summary_table
  import statsmodels.api as sm

  self.models[model] = {'model': sm.OLS(self.y, self.X, **kwargs)}
  self.models[model]['fit'] = self.models[model]['model'].fit()
  grid = np.c_[np.ones(len(self.grid)), self.grid]
  self.models[model]['reg_func'] = lambda _x, _y: sm.OLS(_y, _x, **kwargs).fit().predict(grid)

  self.model = self.models[model]['model']
  self.fit = self.models[model]['fit']
  self.name_model = model

  alpha = self.alpha if alpha is None else alpha
  st, data, ss2 = summary_table(self.fit, alpha=alpha)
  self.models[model]['st'] = st
  self.models[model]['data'] = data
  self.models[model]['ss2'] = ss2
  self.models[model]['fitted'] = data[:,2].squeeze()
  self.models[model]['predict_mean_se'] = data[:,3]
  self.models[model]['predict_mean_ci_low'] = data[:,4].T
  self.models[model]['predict_mean_ci_upp'] = data[:,5].T
  self.models[model]['predict_ci_low'] = data[:,6].T
  self.models[model]['predict_ci_upp'] = data[:,7].T

  prstd, iv_l, iv_u = wls_prediction_std(self.fit)
  self.models[model]['prstd'] = prstd
  self.models[model]['iv_l']  = iv_l
  self.models[model]['iv_u']  = iv_u

  self.models[model]['interp']   = self.fit.params[0]
  self.models[model]['coef']     = self.fit.params[1]
  self.models[model]['e_interp'] = self.fit.bse[0]
  self.models[model]['e_coef']   = self.fit.bse[1]

 def rlm(self, model='RLM', **kwargs):
  import statsmodels.api as sm

  self.models[model] = {'model': sm.RLM(self.y, self.X, **kwargs)}
  self.models[model]['fit'] = self.models[model]['model'].fit()
  grid = np.c_[np.ones(len(self.grid)), self.grid]
  self.models[model]['reg_func'] = lambda _x, _y: sm.RLM(_y, _x, **kwargs).fit().predict(grid)

  self.model = self.models[model]['model']
  self.fit = self.models[model]['fit']
  self.name_model = model

  self.models[model]['interp']   = self.fit.params[0]
  self.models[model]['coef']     = self.fit.params[1]
  self.models[model]['e_interp'] = self.fit.bse[0]
  self.models[model]['e_coef']   = self.fit.bse[1]

 def wls(self, model='WLS', **kwargs):
  from statsmodels.sandbox.regression.predstd import wls_prediction_std
  import statsmodels.api as sm

  self.models[model] = {'model': sm.WLS(self.y, self.X, weights=self.weights, **kwargs)}
  self.models[model]['fit'] = self.models[model]['model'].fit()
  grid = np.c_[np.ones(len(self.grid)), self.grid]
  self.models[model]['reg_func'] = lambda _x, _y: sm.WLS(_y, _x, weights=self.weights, **kwargs).fit().predict(grid)

  self.model = self.models[model]['model']
  self.fit = self.models[model]['fit']
  self.name_model = model

  prstd, iv_l, iv_u = wls_prediction_std(self.fit)
  self.models[model]['prstd'] = prstd
  self.models[model]['iv_l']  = iv_l
  self.models[model]['iv_u']  = iv_u

  self.models[model]['interp']   = self.fit.params[0]
  self.models[model]['coef']     = self.fit.params[1]
  self.models[model]['e_interp'] = self.fit.bse[0]
  self.models[model]['e_coef']   = self.fit.bse[1]

 def ransac(self, model='RANSAC'):
  from sklearn import linear_model
  def reg_func(_x, _y): 
   model = linear_model.RANSACRegressor(linear_model.LinearRegression())
   model.fit(_x, _y)
   return model.predict(self.grid[:, np.newaxis])
  
  self.models[model] = {'model': linear_model.RANSACRegressor(linear_model.LinearRegression())}
  self.models[model]['fit'] = self.models[model]['model'].fit(self.X, self.y)
  self.models[model]['reg_func'] = reg_func

  self.model = self.models[model]['model']
  self.fit = self.models[model]['fit']
  self.name_model = model

  self.models[model]['inlier_mask'] = self.model.inlier_mask_
  self.models[model]['outlier_mask'] = np.logical_not(self.model.inlier_mask_)

  self.models[model]['coef']   = self.model.estimator_.coef_
  self.models[model]['interp'] = self.model.estimator_.intercept_

 def bayesian_ridge(self, model='BR', **kwargs):
  from sklearn import linear_model
  def reg_func(_x, _y):
   model = linear_model.BayesianRidge(**kwargs)
   model.fit(_x, _y)
   return model.predict(self.grid[:, np.newaxis])
  
  self.models[model] = {'model': linear_model.BayesianRidge(**kwargs)}
  self.models[model]['fit'] = self.models[model]['model'].fit(self.X, self.y)
  self.models[model]['reg_func'] = reg_func

  self.model = self.models[model]['model']
  self.fit = self.models[model]['fit']
  self.name_model = model

  self.models[model]['coef']   = self.model.coef_[0]
  self.models[model]['interp'] = self.model.intercept_

 def wrlm(self, weights_diff=None, model='WRLM', omodel='RLM'):
  if weights_diff is not None:
   self.weights_diff = weights_diff
  if self._ey is None:
   print ('>>> WARNING: Errors are needed for model "%s"' % model)
   return
  if not omodel in self.models:
   self.rlm()
  idw = self.models[omodel]['fit'].weights >= self.weights_diff
  self.set_xy(x=self._x[idw], y=self._y[idw], ey=self.ey[idw])
  if self.x.size > 2:
   self.wls(model=model)
   self.models[model]['id_weights'] = idw

 def poly(self, order=None, xcov=None, model='POLY', use_weights=False):
  order = self.order if order is None else order
  w = np.sqrt(self.weights) if (self.weights is not None and use_weights) else None
  p, C_p = np.polyfit(self.x, self.y, order, w=w, cov=True)
  self.models[model] = {'coeffs': p, 'cov': C_p, 'e_coeffs': np.sqrt(np.diag(C_p)), 'order': order, 'use_weights': use_weights}
  self.models[model]['reg_func'] = lambda _x, _y: np.polyval(np.polyfit(_x, _y, self.order), self.grid)

  x = self.x if xcov is None else xcov

  TT = np.vstack([np.power(x, (order-i)) for i in range(order + 1)]).T
  C_yi = np.dot(TT, np.dot(C_p, TT.T))
  self.models[model]['sigma'] = np.sqrt(np.diag(C_yi))
  #fill_between(x, y + models[method]['sigma'], y - models[method]['sigma'], alpha=0.2)

  self.models[model]['interp']   = p[1]
  self.models[model]['coef']     = p[0]
  self.models[model]['e_interp'] = self.models[model]['e_coeffs'][1]
  self.models[model]['e_coef']   = self.models[model]['e_coeffs'][0]
  self.name_model = model

  #self.models[model]['fitted'] = np.polyval(p, self.x).squeeze()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class SimpleLatexTableReader(object):
 def __init__(self,tabname,ncol=None,tcol=None,comment='%',delimiter='&',nheader=1,mask=np.nan,null=['','\\ldots'],dtype=np.float):
  '''
  Simple Latex Table Reader

  Parameters
  ----------
  tabname : String with the name of the file
  ncol : Integer or string. If interger, the column number (in python indexing) of the columns
        that wants to be extracted should be given. If a string, the string of the column header
        (stripped, without whitespaces at the end or beginning) should be given. If not give,
        no "col" attribute will be given
  tcol : Type of numpy array of the column "ncol" that wants to be converted. If not given, a
        string array will be given
  comment : String with the characted for commented lines
  delimiter : Delimiter for the column separation
  nheader : Integer with the number of lines used by the header
  mask : Mask value for coversion of the numpy array for null values
  null : String or list with the null values in the table (to be replaced by "mask")

 Returns
  -------
  Object: obj.table, obj.plain_table, obj.nptable, obj.dtable, obj.col, obj.header, obj.names, 
	  obj.nheader, obj.ncol, obj.tcol, obj.mask, obj.null, obj.comment, obj.delimiter, 
	  obj.dtype

  '''
  self.mask = mask
  self.comment = comment
  self.delimiter = delimiter
  self.nheader = nheader
  self.header = None
  self.table = None
  self.plain_table = None
  self.ncol = ncol
  self.tcol = tcol
  self.mask = mask
  self.col = None
  self.null = checkList(null)
  self.names = None
  self.nptable = None
  self.dtable = None
  self.dtype = dtype
  if tabname is not None:
   self.readtable(tabname)

 def readtable(self,tabname):
  f = open(tabname,'r')
  fr = f.readlines()
  f.close()
  #self.plain_table = [line.strip().replace('\\','') for line in fr]
  self.plain_table = [line.strip().split('\\\\')[0] for line in fr]
  tmptable = [line for line in self.plain_table if not (line.startswith('/') or line.startswith(self.comment) or len(line) == 0)]
  self.table = tmptable[self.nheader:]
  self.header = tmptable[:self.nheader]
  if len(self.header) == 0:
   self.header = None
  if self.header is not None:
   self.names = [line.strip() for line in self.header[0].split(self.delimiter)]
  if self.ncol is not None:
   if isinstance(self.ncol,(str,np.str)):
    try:
     self.ncol = self.names.index(self.ncol.strip())
    except:
     sys.exit('*** Column name "%s" NOT found ***' % self.ncol)
   self.col = np.array([line.split(self.delimiter)[self.ncol].strip() for line in self.table],dtype=np.str)
   if self.tcol is not None:
    if self.null is not None:
     for inull in self.null:
      self.col[self.col == inull] = self.mask
    self.col = np.array(self.col,dtype=self.tcol)
  tmptab = []
  for row in [line.split(self.delimiter) for line in self.table]:
   tmptab.append([col.strip() for col in row])
  self.nptable = np.array(tmptab).T
  if self.null is not None:
   for inull in self.null:
    self.nptable[self.nptable == inull] = self.mask
  self.dtable = OrderedDict()
  for i,iarray in enumerate(self.nptable):
   name = 'col%i' % (i+1)
   if self.names is not None and len(self.names) == self.nptable.shape[0]:
    name = self.names[i] 
   try:
    self.dtable[name] = iarray.astype(self.dtype)
   except:
    self.dtable[name] = iarray
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def WriteAtpyTable(nfile,vars,hvars=None,dicthcolumns=None,hmag=None,hdes=None,
	formats=None,writer=None,califa=False,hcolumns=False,header=True,
	hcomments=None,hsort=None,comchar='#',**kwargs):
 '''
 Writes a table using "asciitable" module.

 Parameters
 ----------
 nfile : String value with the name of the table to be written
 vars : Array or list/tuple of arrays
 hvars: None or list of strings, optional
	List of strings with the same lenght as "vars" with the name of the 
	arrays in "vars" used to named the columns of the tables. If not 
	provided, names in the format "col1", "col2", ... will be used
 dicthcolumns : None or dictionary, optional (used for CALIFA tables; califa = True)
	Dictionary with the variables of the CALIFA tables. By default:
		'AUTHOR': 'Ruben Garcia-Benito (RGB)',
                'SOURCE': 'CALIFA Collaboration',
                'DATE':    Today's date,
                'VERSION': '1.0',
                'COLAPRV': 'None',
                'PUBAPRV': 'None'
		'COLUMNX:  Name, type array, physical magnitude, description
	For the "COLUMNX" entry, two options can be provided. 1) A tuple with 
	2 elements. It will change (physical magnitude, description). 2) A 
	4 element tuple will change all the 4 elements of the "COLUMNX"
	This is an alternative to "hmag" and "hdes", but in dictionry form
 hmag: None or list of strings, optional (used for CALIFA tables; califa = True)
	List of strings with the same lenght as "vars" with the description of the 
	physical magnitude of the correspoding array. If is not a string with 
	length > 0, it will not be used: (None, '', 0) will leave this description 
	blank
 hdes: None or list of strings, optional (used for CALIFA tables; califa = True)
	List of strings with the same lenght as "vars" with a description of the 
	of the correspoding array. If is not a string with length > 0, it will not 
	be used: (None, '', 0) will leave this description blank
 formats : None, string, list or dict, optional
	Formats of the table columns for each array. 1) None: it will be adapted to 
	the size of each array type. 2) String: the same format will be applied to 
	all columns. 3) List: a list of strings of the same length as "vars" with 
	the format of each array. If a particular item is not a string with length > 0, 
	it will not be used (i.e.: None, '', 0). 4) Dictionary: a dictionary with 
	the name of the arrays given by "hvars" (or "col1" type if not given) and 
	the requiered format
 writer : None or Writer "asciitable" instance, optional
 califa : Bool
	Write the table in CALIFA's format
 hcolumns : Bool
	Write the CALIFA COLUMNS description if the table is not CALIFA's format
 header : Bool
	If the table is not a CALIFA table, write the header with the name of the 
	variable in each column
 hcomments : None, string or list of strings, optional
	If the table is not a CALIFA table, write lines comments
 hsort : List of Strings
	Sorts the table according to one column, named by its "hvars" name
 comchar : '#' or None
	Character used to write the header
 kwargs : optional
	If provided, should be arguments for the Writer "asciitable" instance
 '''

 import datetime
 import asciitable

 table = OrderedDict()

 if isinstance(vars,np.ndarray):
  if len(vars.shape) == 1:
   vars = [vars]
  else:
   vars = [item for item in vars]
 elif not isinstance(vars,(list,tuple)):
  vars = [vars]

 if formats is not None and not isinstance(formats,(dict,str)) and len(formats) != len(vars):
  sys.exit('*** "vars" (%s) and "formats" (%s) should have the same length!!! ***' % (len(vars),len(formats)))

 if hmag is not None and len(hmag) != len(vars):
  sys.exit('*** "vars" (%s) and "hmag" (%s) should have the same length!!! ***' % (len(vars),len(hmag)))

 if hdes is not None and len(hdes) != len(vars):
  sys.exit('*** "vars" (%s) and "hdes" (%s) should have the same length!!! ***' % (len(vars),len(hdes)))

 if hvars is not None:
  if len(hvars) != len(vars):
   sys.exit('*** "vars" (%s) and "hvars" (%s) should have the same length!!! ***' % (len(vars),len(hvars)))
  else:
   names = hvars
 else:
  names = ['col'+str(i+1) for i in range(len(vars))]

 if writer is None and not califa:
  if header: 
   writer = asciitable.FixedWidth
   if 'bookend' not in kwargs.keys(): kwargs['bookend'] = False
  else:      
   writer = asciitable.FixedWidthNoHeader
  if 'delimiter' not in kwargs.keys(): kwargs['delimiter'] = None
 
 if hsort is not None:
  if not isinstance(hsort,(list,tuple)):
   hsort = [hsort]

 # Fix AsciiTable Commented Header
 if not califa and header: 
  if hsort is not None and comchar is not None:
   hsort = ['%s ' % comchar + names[0] if item == names[0] else item for item in hsort]
  if comchar is not None:
   names[0] = '%s ' % comchar + names[0]

 if comchar is not None:
  comchar = '%s ' % comchar
 else:
  comchar = ''
   
 for name,var in zip(names,vars):
  table[name] = var

 dicthd = {     'AUTHOR': 'Ruben Garcia-Benito (RGB)',
                'SOURCE': 'CALIFA Collaboration',
                'DATE':    str(datetime.date.today()),
                'VERSION': '1.0',
                'COLAPRV': 'None',
                'PUBAPRV': 'None'}

 for i,var in enumerate(vars):
  vtype = 'None'
  if isinstance(var,np.ndarray):
   if var.dtype == np.bool:         vtype = 'bool'
   if var.dtype == np.int16:        vtype = 'short'
   if var.dtype == np.int32:        vtype = 'int'
   if var.dtype == np.int64:        vtype = 'long'
   if var.dtype == np.float32:      vtype = 'float'
   if var.dtype == np.float64:      vtype = 'double'
   if np.issubdtype(var.dtype,str): vtype = 'string'
   if var.dtype.kind == 'S':        vtype = 'string'
  else:
   if isinstance(var[0],types.BooleanType):  vtype = 'bool'
   if isinstance(var[0],types.IntType):      vtype = 'int'
   if isinstance(var[0],types.LongType):     vtype = 'long'
   if isinstance(var[0],types.FloatType):    vtype = 'float'
   if isinstance(var[0],types.StringType):   vtype = 'string'
  dicthd['COLUMN'+str(i+1)] = [names[i],vtype,None,None]

 if dicthcolumns is not None:
  for key in dicthcolumns.keys():
   if 'COLUMN' in key:
    val = list(dicthcolumns[key])
    if isinstance(val,list):
     if len(val) == 4: dicthd[key] = val
     if len(val) == 2: dicthd[key][2:] = val
   else:
    dicthd[key] = dicthcolumns[key]

 if hmag is None and hdes is not None: hmag = [None]*len(vars)
 if hdes is None and hmag is not None: hdes = [None]*len(vars)

 if hmag is not None and hdes is not None:
  for i,mkey,dkey in izip(count(),hmag,hdes):
   hd = 'COLUMN'+str(i+1)
   if isinstance(mkey,str) and len(mkey) > 0: dicthd[hd][2] = mkey
   if isinstance(dkey,str) and len(dkey) > 0: dicthd[hd][3] = dkey

 if formats is not None:
  if isinstance(formats,dict):
   dformat = formats.copy()
  else:
   if isinstance(formats,str): formats = len(names)*[formats]
   dformat = {}
   for name,format in zip(names,formats):
    if isinstance(format,str) and len(format) > 0:
     dformat[name] = format

 # Fix AsciiTable Commented Header
 if not califa and header and hcolumns:
  dicthd['COLUMN1'][0] = dicthd['COLUMN1'][0].replace(comchar,'')

 # Open file
 file = open(nfile,'w')

 if not califa and hcomments is not None:
  if not isinstance (hcomments, (list,tuple)): hcomments = [hcomments]
  for item in hcomments:
   file.write('%s%s\n' % (comchar,str(item)))

 if califa:
  lorder = ['AUTHOR','SOURCE','DATE','VERSION','COLAPRV','PUBAPRV']
  for key in lorder:
   file.write('%s%s: %s\n' % (comchar,key,dicthd[key]))

 if califa or hcolumns:
  for key in Nsort(dicthd.keys()):
   if 'COLUMN' in key:
    file.write(('%s%s: %s, %s, %s, %s\n' % ((comchar,key)+tuple(dicthd[key]))).replace('None',' '))

 if califa:
  writer = asciitable.NoHeader
  if 'delimiter' not in kwargs.keys(): kwargs['delimiter'] = ','

 # Sort option
 if hsort is not None:
  tmptb = CreateAtpyTable(vars,hvars=names,hsort=hsort,fname=None)
  for tname in names:
   table[tname] = tmptb[tname]

 if formats is not None:
  asciitable.write(table,file,names=names,formats=dformat,Writer=writer,**kwargs)
 else:
  asciitable.write(table,file,names=names,Writer=writer,**kwargs)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def WriteTable(nfile,vars,hvars=None,dicthcolumns=None,hmag=None,hdes=None,
	formats=None,writer=None,califa=False,hcolumns=False,header=True,
	hcomment=None,shcomment=None,dcomment=None,sdcomment=None,hsort=None,
	use_astropy=True,**kwargs):
 '''
 Writes a table using "astropy" module.

 Parameters
 ----------
 nfile : String value with the name of the table to be written
 vars : Array or list/tuple of arrays
 hvars: None or list of strings, optional
	List of strings with the same lenght as "vars" with the name of the 
	arrays in "vars" used to named the columns of the tables. If not 
	provided, names in the format "col1", "col2", ... will be used
 dicthcolumns : None or dictionary, optional (used for CALIFA tables; califa = True)
	Dictionary with the variables of the CALIFA tables. By default:
		'AUTHOR': 'Ruben Garcia-Benito (RGB)',
                'SOURCE': 'CALIFA Collaboration',
                'DATE':    Today's date,
                'VERSION': '1.0',
                'COLAPRV': 'None',
                'PUBAPRV': 'None'
		'COLUMNX:  Name, type array, physical magnitude, description
	For the "COLUMNX" entry, two options can be provided. 1) A tuple with 
	2 elements. It will change (physical magnitude, description). 2) A 
	4 element tuple will change all the 4 elements of the "COLUMNX"
	This is an alternative to "hmag" and "hdes", but in dictionry form
 hmag: None or list of strings, optional (used for CALIFA tables; califa = True)
	List of strings with the same lenght as "vars" with the description of the 
	physical magnitude of the correspoding array. If is not a string with 
	length > 0, it will not be used: (None, '', 0) will leave this description 
	blank
 hdes: None or list of strings, optional (used for CALIFA tables; califa = True)
	List of strings with the same lenght as "vars" with a description of the 
	of the correspoding array. If is not a string with length > 0, it will not 
	be used: (None, '', 0) will leave this description blank
 formats : None, string, list or dict, optional
	Formats of the table columns for each array. 1) None: it will be adapted to 
	the size of each array type. 2) String: the same format will be applied to 
	all columns. 3) List: a list of strings of the same length as "vars" with 
	the format of each array. If a particular item is not a string with length > 0, 
	it will not be used (i.e.: None, '', 0). 4) Dictionary: a dictionary with 
	the name of the arrays given by "hvars" (or "col1" type if not given) and 
	the requiered format
 writer : None or Writer "asciitable" instance, optional
 califa : Bool
	Write the table in CALIFA's format
 hcolumns : Bool
	Write the CALIFA COLUMNS description if the table is not CALIFA's format
 header : Bool
	If the table is not a CALIFA table, write the header with the name of the 
	variable in each column
 hcomment : None, string or list of strings, optional
	If the table is not a CALIFA table, write lines comments
 shcomment : None (default value for the Writer)
	Character used to write the header
 dcomment : None, string or list of strings, optional
	If the table is not a CALIFA table, write data comments at the end of the file
 sdcomment : None (default value for the Writer)
	Character used to write the comments of the data at the end of the file
 hsort : List of Strings or Int if "use_astropy=True", String or Int, otherwise
	Sorts the table according to one column, named by its "hvars" name or 
	by the index (integer) position in the "var" list
 kwargs : optional
	If provided, should be arguments for the Writer "asciitable" instance
 '''

 import datetime

 table = OrderedDict()

 if isinstance(vars,np.ndarray):
  if len(vars.shape) == 1:
   vars = [vars]
  else:
   vars = [item for item in vars]
 elif not isinstance(vars,(list,tuple)):
  vars = [vars]

 if formats is not None and not isinstance(formats,(dict,str)) and len(formats) != len(vars):
  sys.exit('*** "vars" (%s) and "formats" (%s) should have the same length!!! ***' % (len(vars),len(formats)))

 if hmag is not None and len(hmag) != len(vars):
  sys.exit('*** "vars" (%s) and "hmag" (%s) should have the same length!!! ***' % (len(vars),len(hmag)))

 if hdes is not None and len(hdes) != len(vars):
  sys.exit('*** "vars" (%s) and "hdes" (%s) should have the same length!!! ***' % (len(vars),len(hdes)))

 if hvars is not None:
  if len(hvars) != len(vars):
   sys.exit('*** "vars" (%s) and "hvars" (%s) should have the same length!!! ***' % (len(vars),len(hvars)))
  else:
   names = hvars
 else:
  names = ['col'+str(i+1) for i in range(len(vars))]

 if writer is None and not califa:
  if header: 
   writer = 'fixed_width'
  else:      
   writer = 'fixed_width_no_header'
  if 'bookend' not in kwargs.keys(): kwargs['bookend'] = False
  if 'delimiter' not in kwargs.keys(): kwargs['delimiter'] = None
 
 if hsort is not None:
  if not isinstance(hsort,(list,tuple)):
   hsort = [hsort]

 for name,var in zip(names,vars):
  table[name] = var

 dicthd = {     'AUTHOR': 'Ruben Garcia-Benito (RGB)',
                'SOURCE': 'CALIFA Collaboration',
                'DATE':    str(datetime.date.today()),
                'VERSION': '1.0',
                'COLAPRV': 'None',
                'PUBAPRV': 'None'}

 for i,var in enumerate(vars):
  vtype = 'None'
  if isinstance(var,np.ndarray):
   if var.dtype == np.bool:         vtype = 'bool'
   if var.dtype == np.int16:        vtype = 'short'
   if var.dtype == np.int32:        vtype = 'int'
   if var.dtype == np.int64:        vtype = 'long'
   if var.dtype == np.float32:      vtype = 'float'
   if var.dtype == np.float64:      vtype = 'double'
   if np.issubdtype(var.dtype,str): vtype = 'string'
   if var.dtype.kind == 'S':        vtype = 'string'
  else:
   if isinstance(var[0],types.BooleanType):  vtype = 'bool'
   if isinstance(var[0],types.IntType):      vtype = 'int'
   if isinstance(var[0],types.LongType):     vtype = 'long'
   if isinstance(var[0],types.FloatType):    vtype = 'float'
   if isinstance(var[0],types.StringType):   vtype = 'string'
  dicthd['COLUMN'+str(i+1)] = [names[i],vtype,None,None]

 if dicthcolumns is not None:
  for key in dicthcolumns.keys():
   if 'COLUMN' in key:
    val = list(dicthcolumns[key])
    if isinstance(val,list):
     if len(val) == 4: dicthd[key] = val
     if len(val) == 2: dicthd[key][2:] = val
   else:
    dicthd[key] = dicthcolumns[key]

 if hmag is None and hdes is not None: hmag = [None]*len(vars)
 if hdes is None and hmag is not None: hdes = [None]*len(vars)

 if hmag is not None and hdes is not None:
  for i,mkey,dkey in izip(count(),hmag,hdes):
   hd = 'COLUMN'+str(i+1)
   if isinstance(mkey,str) and len(mkey) > 0: dicthd[hd][2] = mkey
   if isinstance(dkey,str) and len(dkey) > 0: dicthd[hd][3] = dkey

 if formats is not None:
  if isinstance(formats,dict):
   dformat = formats.copy()
  else:
   if isinstance(formats,str): formats = len(names)*[formats]
   dformat = {}
   for name,format in zip(names,formats):
    if isinstance(format,str) and len(format) > 0:
     dformat[name] = format

 if not califa and hcomment is not None:
  if not isinstance (hcomment, (list,tuple)): 
   hcomment = [hcomment]

 if califa:
  lorder = ['AUTHOR','SOURCE','DATE','VERSION','COLAPRV','PUBAPRV']
  if hcomment is None:
   hcomment = []
  for key in lorder:
   hcomment.append('%s: %s' % (key,dicthd[key]))

 if califa or hcolumns:
  if hcomment is None:
   hcomment = []
  for key in Nsort(dicthd.keys()):
   if 'COLUMN' in key:
    hcomment.append(('%s: %s, %s, %s, %s' % ((key,)+tuple(dicthd[key]))).replace('None',' '))

 if califa:
  writer = 'no_header'
  if 'delimiter' not in kwargs.keys(): kwargs['delimiter'] = ','

 # Sort option
 if hsort is not None:
  tmptb = CreateTable(vars,hvars=names,hsort=hsort,fname=None)
  for tname in names:
   table[tname] = tmptb[tname]

 sshcomment = '' if shcomment is None else shcomment
 # Fix AsciiTable Commented Header
 table = OrderedDict([('%s%s' % (sshcomment,k),v) if k == table.keys()[0] else (k,v) for k,v in table.items()])
 if formats is not None:
  dformat = OrderedDict([('%s%s' % (sshcomment,k),v) if k in table.keys()[0] else (k,v) for k,v in dformat.items()])

 if formats is not None:
  writeTableTxt(table,output=nfile,format=writer,hcomment=hcomment,dcomment=dcomment,shcomment=shcomment,sdcomment=sdcomment,formats=dformat,**kwargs)
 else:
  writeTableTxt(table,output=nfile,format=writer,hcomment=hcomment,dcomment=dcomment,shcomment=shcomment,sdcomment=sdcomment,**kwargs)
# ---------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def writeTableTxt(table,output=None,format=None,Writer=None,hcomment=None,
	dcomment=None,shcomment=None,sdcomment=None,**kwargs):

 import astropy.io.ascii as asc

 if isinstance(table,dict):
  table = Table(table)

 if 'delimiter' not in kwargs.keys(): 
  kwargs['delimiter'] = None

 Writer = asc.ui._get_format_class(format,Writer,'Writer')
 if 'bookend' not in kwargs.keys() and isinstance(Writer,(asc.FixedWidth,asc.FixedWidthNoHeader,asc.FixedWidthTwoLine)):   
  kwargs['bookend'] = False
 writer = asc.get_writer(Writer,fast_writer=False,**kwargs) # fast_writer in Astropy 1.0
 lines = writer.write(table)

 if shcomment is None:
  try:
   shcomment = writer.header.write_comment
   if shcomment is None:
    shcomment = '# '
  except:
   shcomment = '# '
   if format is not None:
    if 'latex' in format.lower():
     shcomment = '% '
 if sdcomment is None:
  try:
   sdcomment = writer.data.write_comment
   if sdcomment is None:
    sdcomment = '# '
  except:
   sdcomment = '# '
   if format is not None:
    if 'latex' in format.lower():
     sdcomment = '% '

 if hcomment is not None:
  hcomment = checkList(hcomment)
  hclines = []
  for line in hcomment:
   hclines.append('%s%s' % (shcomment,line))
  lines = hclines + lines

 if dcomment is not None:
  dcomment = checkList(dcomment)
  dclines = []
  for line in dcomment:
   dclines.append('%s%s' % (sdcomment,line))
  lines = lines + dclines

 # Write the lines to output
 if output is not None:
  outstr = os.linesep.join(lines)
  output = open(output, 'w')
  output.write(outstr)
  output.write(os.linesep)
  output.close()
 else:
  return lines
# ----------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def FitsHeaderKeys2Array(listfits,lheader,hdu=0):
 '''
 Returs a tuple/list of arrays with the values of provided FITS header keys in 
 a FITS file or list of FITS files.

 Parameters
 ----------
 listfits : string or list of strings
	String of list of strings with the name of FITS files
 lheader :  string or list of strings
	String or list of strings with the name of the FITS header key to 
	look for. If the key is not found, a 0.0 value will be added
 hdu : int
	Integer with the HDU value to read the header

 Returns
 -------
 A tuple or list of the same number of arrays as the number of "lheader" keys
 '''

 larrays = []

 if not isinstance(listfits, (list,tuple)): listfits = [listfits]

 if not isinstance(lheader, (list,tuple)):  lheader = [lheader]
 for item in lheader: larrays.append([])

 for item in listfits:
  phdu = pyfits.open(item)
  hd = phdu[hdu].header
  phdu.close()
  for i,key in enumerate(lheader):
   try:    larrays[i].append(hd[key])
   except: larrays[i].append(float(0.0))

 for i in range(len(larrays)):
  larrays[i] = np.array(larrays[i])

 return larrays
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def FitsHeaderKeys2Dict(listfits,lheader,hdu=0,fkey='filename',abspath=False,table=False,sep=None,sep_id=0):
 lheader = checkList(lheader)
 larrays = FitsHeaderKeys2Array(listfits,lheader,hdu=hdu)
 if not abspath:
  listfits = [os.path.basename(item) for item in listfits]
 if sep is not None and isinstance(sep,str):
  listfits = [item.split(sep)[sep_id] for item in listfits]
 dfits = OrderedDict([(fkey,listfits)])
 for key,value in zip(lheader,larrays):
  dfits[key] = value
 if table:
  dfits = Table(dfits)
 return dfits
# ---------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
class LoadFits:

 def __init__(self,name,exdata=None,exflag=None,exerror=None,exerw=None,exfibc=None,exflat=None,
	exhdr=0,error_file=None,flag_file=None,wave_file=None,exwave=None,hdisp='DISPAXIS',
	hcrval='CRVAL',hcdelt='CDELT',crval=None,cdelt=None,haxis=True,specaxis=None,
	hvel=None,z=None,velocity=None,guess=True,flip2disp=True,rss=False,f2e=0.1,
	c=299792.458,dist_pc=None,filters=None,filter_props=None,filter_mode=None,
	units=None,verbo=True):
  '''
  Reads a FITS file and returns an object with some attributes

  Parameters
  ----------
  name : string
	String with the name of the FITS file
  exdata : None or int, optional
	Number of the HDU to read the data. By default, 0
  exflag : None or int, optional
	Number of the HDU to read the mask/flag data. If it is CALIFA data, it will search 
	for 'BADPIX' automatically if "getcalifa = True"
  exerror : None or int, optional
	Number of the HDU to read the error data. If it is CALIFA data, it will search 
        for 'ERROR' automatically if "getcalifa = True"
  exerw : None or int, optional
	Number of the HDU to read the error weighted data. If it is CALIFA data, it will search 
        for 'ERRWEIGHT' automatically if "getcalifa = True"
  exfibc : None or int, optional
	Number of the HDU to read the fiber cover data. If it is CALIFA data, it will search 
        for 'FIBCOVER' automatically if "getcalifa = True"
  exflat : None or int, optional
	Number of the HDU to read the flat data. If it is CALIFA data, it will search 
        for 'FLAT' automatically if "getcalifa = True"
  exhdr : int
	Number of the HDU where to read the Header
  error_file : string
	External FITS file to read the ERRORs (see function ReadError())
  flag_file : string
	External FITS file to read the FLAGs (see function ReadFlag())
  wave_file : string
	External FITS file to read the WAVELENGTH solution (see function ReadWave())
  exwave : string of integer
	Number or name (string) of the HDU where to read the wavelength array
  hdisp : string
	String for reading the dispersion axis axis ('DISPAXIS')
  hcrval : string
	String for reading the 'CRVAL' information
  hdelt : string
	String for reading the 'CDELT' information
  crval : float
	Value for 'CRVAL' information. It overwrites "hcrval"
  haxis : Bool
	Uses "specaxis" to add the dimension integer to the header "hcrval" and "hcdelt" 
	keywords. If "hcrval" and "hcdelt" are given and are absolute (nothing to add), 
	set "haxis = False"
  delt : float
	Value for 'CDELT' information. It overwrites "cdelt"
  specaxis : None or int, optional
	Number to read the wavelenght (CRVAL/CDELT) information. If 'DISPAXIS' 
	exists, it will be used. If does not exists, by default it will search 
	the 1 axis (i.e.: CRVAL1 --> CRVAL+specaxis). If it fails (because there 
	is not wavelength information or the provided numbers are wrong), it 
	will give "None" results for the wavelength attributes
  hvel : string
	String for reading the recessional velocity (km/s) from the header
  z : None or float, optional
	If the redshift of the object is provided (or is readed from the header), a 
	"wave_rest" attribute will be written with the rest wavelength for that redshift
	(see RestWave())
  velocity : None or float, optinal
	Velocity in km/s to obtain the "wave_rest" attribute (used instead of z if both provided)
  guess : Bool
	For CALIFA data, it will search for the 'BADPIX', 'ERROR' and 'ERRWEIGHT' HDUs. 
	Also, it will calculate the "wave_rest" wavelength from the recessional velocity 
	of the CALIFA header ('MED_VEL' or 'V500 MED_VEL'), if present. It will be overwritten 
	if "z" is used. It can also read MANGA cubes and FITS Pycasso Files 
  flip2disp : Bool
	If the data is 2D, it will transpose (flip) the Matrix according to the DISPAXIS 
	axis so that a matrix in shape (wavelength, number of spectra) is provided
  rss : Bool
	It gets an RSS object instead of a Cube object. Applies only to Pycasso objects
  f2e : float or None
	Conversion factor (default: 0.1) of Flux to Error in case the ERROR is not provided:
                                f_error = f2e * np.abs(flux)
  c : Float
   	Speed of light in km/s. Used to obtain the redshift from the recessional velocity of 
 	CALIFA cubes
  dist_pc : Float
	Distance of the object in pc (to estimate luminosities)
  filters : List of list or dictionary
	filters = {'r': 'sloarn_r.txt'}   |   ['sloan_r.txt']   |   [['r', 'sloan_r.txt']]
  filter_mode : String or list of strings
	'Obs' to estimate the filter in the observed frame or 'ObsR' in rest frame
  filter_props : String or list of strings
	List of properties to calculate the magnitudes. By default:
		filter_props = ['Flux','Mag','AB','Lum','L']
  units : Float
	Units of the flux (i.e. 1e-16)
  verbo : Bool
	Verbosity for some Warnings

  Returns
  -------
  Object: obj.data, obj.hdr, obj.phdr, obj.flag, obj.error, obj.errorw, obj.fibcover, obj.wave, obj.wave_rest, 
		obj.redshift, obj.velocity, obj.cdelt, obj.crval, obj.nwave, self.syn, self.dispaxis
  '''
  if not os.path.exists(name):
   sys.exit('*** File "' + os.path.basename(name) + '" does NOT exists!!! ***')
  try:    hdu = pyfits.open(name)
  except: sys.exit('*** File "' + os.path.basename(name) + '" has a PROBLEM!!! ***')
  self.hdr  = hdu[exhdr].header
  self.phdr = hdu[0].header

  # Setting and reading variables
  naxis = self.hdr.get('NAXIS')
  self.califaid = self.hdr.get('CALIFAID')
  self.kalifaid = 'K'+str(self.califaid).zfill(4) if self.califaid is not None else None
  self.survey = self.hdr.get('SURVEY')
  self.instrument = self.hdr.get('INSTRUME')
  self.pycasso = self.hdr.get('QVERSION')
  if self.survey is not None: 
   self.survey = self.survey.strip()
  if self.instrument is not None: 
   self.instrument = self.instrument.strip()
  self.filter_mode = checkList(filter_mode) if filter_mode is not None else ['Obs', 'ObsR']
  self.filter_props = ['Flux', 'Mag', 'AB', 'Lum', 'L'] if filter_props is None else checkList(filter_props)
  self.dist_pc  = dist_pc
  self.filters  = filters
  self.units    = units
  self.dpass    = None
  self.dmag     = None
  self.pversion = None
  self.dispaxis = None
  self.velocity = None
  self.redshift = None
  self.ivar     = None
  self.error    = None
  self.flag     = None
  self.errorw   = None
  self.fibcover = None
  self.flat     = None
  self.wave     = None
  self.syn      = None
  self.K        = None

  # Velocity read in case of CALIFA
  # QBICK redshift. Sometimes MED_VEL is external, but is stored as CZ
  if 'CZ' in self.hdr.keys():
   self.velocity = self.hdr['CZ']
  if 'MED_VEL' in self.hdr.keys():
   self.velocity = self.hdr['MED_VEL']
  if 'V500 MED_VEL' in self.hdr.keys(): # COMBO V500+V1200
   self.velocity = self.hdr['V500 MED_VEL']
  if hvel is not None and hvel in self.hdr.keys():
   self.velocity = float(self.hdr[hvel])
  if self.velocity is not None:
   self.redshift = float(self.velocity)/c

  # Read HDUs
  self.data = hdu[0].data
  if len(hdu)>1 and guess:
   for i in range(1, len(hdu)):
    if 'EXTNAME' in hdu[i].header.keys():
     if hdu[i].header['EXTNAME'].split()[0]   == 'ERROR':
      self.error = hdu[i].data
     elif hdu[i].header['EXTNAME'].split()[0] == 'BADPIX':
      self.flag = hdu[i].data.astype('bool')
     elif hdu[i].header['EXTNAME'].split()[0] == 'ERRWEIGHT':
      self.errorw = hdu[i].data
     elif hdu[i].header['EXTNAME'].split()[0] == 'FIBCOVER':
      self.fibcover = hdu[i].data
     elif hdu[i].header['EXTNAME'].split()[0] == 'WAVE':
      self.wave = hdu[i].data
     elif hdu[i].header['EXTNAME'].split()[0] == 'FLAT':
      self.flat = hdu[i].data
     # MaNGA
     if self.instrument == 'MaNGA':
      if hdu[i].header['EXTNAME'].split()[0]   == 'FLUX':
       self.data  = hdu['FLUX'].data
      if hdu[i].header['EXTNAME'].split()[0]   == 'IVAR':
       self.ivar  = hdu['IVAR'].data
       self.error = 1. / np.sqrt(hdu['IVAR'].data)
      if hdu[i].header['EXTNAME'].split()[0]   == 'WAVE':
       self.wave  = hdu['WAVE'].data
      if hdu[i].header['EXTNAME'].split()[0]   == 'MASK':
       self.flag  = hdu['MASK'].data
     # MUSE
     if self.instrument == 'MUSE':
      if hdu[i].header['EXTNAME'].split()[0]   == 'DATA':
       self.data = hdu['DATA'].data
       self.hdr  = hdu['DATA'].header
       naxis = self.hdr.get('NAXIS')
      if hdu[i].header['EXTNAME'].split()[0]   == 'STAT':
       self.error = hdu['STAT'].data
  if exdata is not None:
   self.data = hdu[exdata].data
  if exflag is not None:
   self.flag = hdu[exflag].data
  if exerror is not None:
   self.error = hdu[exerror].data
  if exerw is not None:
   self.errorw = hdu[exerw].data
  if exfibc is not None:
   self.fibcover = hdu[exfibc].data
  if exflat is not None:
   self.flat = hdu[exflat].data

  # Close file
  hdu.close()

  # Wavelength
  disp = self.hdr.get(hdisp)
  if disp is not None:
   if specaxis is None: specaxis = disp
  else:
   if specaxis is None: specaxis = 3 if (naxis == 3) else 1
  self.dispaxis = specaxis

  self.nwave = self.hdr.get('NAXIS'+str(specaxis))
  if haxis:
   scrval = hcrval + str(specaxis)
   scdelt = hcdelt + str(specaxis)
  else:
   scrval = hcrval
   scdelt = hcdelt
  self.crval = self.hdr.get(scrval)
  self.cdelt = self.hdr.get(scdelt)
  if crval is not None:
   self.crval = crval
  if cdelt is not None:
   self.cdelt = cdelt
  if self.cdelt is not None and self.crval is not None and self.nwave is not None and self.wave is None:
   self.wave = self.crval + self.cdelt*np.arange(self.nwave)
  if self.cdelt is None and self.wave is None and self.crval is not None: 
   # SDSS/MANGA simulated cubes
   #if ('CTYPE3' in self.hdr.keys()) and (self.hdr['CTYPE3'].find('LOG10') != -1) and ('CD3_3' in self.hdr.keys()):
   # self.cdelt = self.hdr.get('CD3_3') # MANGA
   # self.wave = np.sort(np.power(10.,float(self.crval) - float(self.cdelt)*np.arange(self.nwave)))
   # For MUSE
   if ('CTYPE3' in self.hdr.keys()) and (self.hdr['CTYPE3'].find('AWAV') != -1) and ('CD3_3' in self.hdr.keys()):
    self.cdelt = self.hdr.get('CD3_3') # MUSE
    self.wave = self.crval + self.cdelt*np.arange(self.nwave)

  # Read external files for WAVE, ERROR and FLAG
  self.ReadHDU(name,exwave)
  self.ReadWave(wave_file)
  self.ReadError(error_file)
  self.ReadFlag(flag_file)

  # External redshift & Rest Wavelength  
  self.RestWave(z=z,velocity=velocity)

  # Guess Pycasso Format
  if self.pycasso is not None:
   self.GuessPycasso(name)
  # Read Pycasso Format
  if self.pycasso is not None and guess:
   self.ReadPycasso(name,rss=rss)

  # Create Error and Flag Files if do not exist 
  if self.error is None and f2e is not None:
   if verbo: print('*** No ERROR found!! We create one as (%2.1f*F_lambda)!! ***' % f2e)
   self.error_file = 'No error file provided. Created one as (%2.1f*F_lambda)' % f2e
   self.error = f2e*np.abs(self.data)

  if self.flag is None:
   if verbo: print('*** No FLAG found!! We create one and set all to 0!! ***')
   self.flag_file = 'No flag file provided. Created one and set all to 0'
   self.flag = np.zeros((self.data.shape))

  # Transpose if required
  #if naxis == 2 and disp == 1 and flip2disp:
  if naxis == 2 and specaxis == 1 and flip2disp:
   self.data = self.data.T
   if self.error is not None:  self.error  = self.error.T
   if self.flag is not None:   self.flag   = self.flag.T
   if self.errorw is not None: self.errorw = self.errorw.T

  if self.units is not None:
   self.data *= self.units
   if self.error is not None:
    self.error *= self.units
  
  # Magnitudes
  self.setPassbands(filters=self.filters)
  self.getPassbands()

 # Functions -------------
 def OpenSimpleFits(self,name_fits,ax=0):
  im = pyfits.open(name_fits)
  data = im[ax].data
  im.close()
  return data

 def ReadError(self,error_file=None,ax=0):
  if error_file is not None:
   self.error = self.OpenSimpleFits(error_file,ax=ax)
   self.error_file = os.path.basename(error_file)
  else:
   self.error_file = None

 def ReadFlag(self,flag_file=None,ax=0):
  if flag_file is not None:
   self.flag = self.OpenSimpleFits(flag_file,ax=ax)
   self.flag_file = os.path.basename(flag_file)
  else:
   self.flag_file = None

 def ReadWave(self,wave_file=None,ax=0):
  if wave_file is not None:
   self.wave = self.OpenSimpleFits(wave_file,ax=ax)
   self.wave_file = os.path.basename(wave_file)
  else:
   self.wave_file = None

 def ReadHDU(self,name,exhdu=None):
  if exhdu is not None:
   try:
    self.wave = pyfits.getdata(name,exhdu)
   except:
    print('*** FITS EXTENSION "%s" NOT found!!! ***' % exhdu)

 def RestWave(self,z=None,velocity=None):
  if z is not None:
   self.redshift = z
  if velocity is not None:
   self.velocity = velocity
   self.redshift = float(velocity)/c
  if self.redshift is not None and self.wave is not None:
   self.wave_rest = self.wave/(1.0 + self.redshift)
  else:
   self.wave_rest = None

 def GuessPycasso(self,name):
  hdu = pyfits.open(name)
  lhdu = [ihdu.name for ihdu in hdu]
  hdu.close()
  lhpy = ['F_OBS','F_SYN','F_WEI','A_V']
  if not all(ilhpy in lhdu for ilhpy in lhpy):
   self.pycasso = None

 def ReadPycasso(self,name,rss=False):
  try:
   import pycasso
   K = fitsQ3DataCube(name)
   self.pversion  = pycasso.__version__
   self.hdr       = K.header
   self.wave_rest = K.l_obs
   self.wave      = K.l_obs
   if rss:
    self.data      = K.f_obs
    self.error     = K.f_err
    self.flag      = K.f_flag
    self.errorw    = K.f_wei
    self.syn       = K.f_syn
   else:
    self.data      = K.zoneToYX(K.f_obs, extensive=False)
    self.error     = K.zoneToYX(K.f_err, extensive=False)
    self.flag      = K.zoneToYX(K.f_flag, extensive=False)
    self.errorw    = K.zoneToYX(K.f_wei, extensive=False)
    self.syn       = K.zoneToYX(K.f_syn, extensive=False)
   self.K         = K
  except:
   sys.exit('*** PyCASSO FITS format. You need PyCasso module! ***')

 def setPassbands(self, filters=None):
  if filters is None:
   return
  from pyfu.passband import PassBand
  self.dpass = OrderedDict()
  if isinstance(filters, (str, list, tuple)):
   filters = checkList(filters)
   for filt in filters:
    if isinstance(filt, str):
     passband = PassBand(file=filt)
     self.dpass[passband._name_filter] = passband
    elif isinstance(filt, list):
     passband = PassBand(file=filt[1])
     self.dpass[filt[0]] = passband
  elif isinstance(filters, dict):
   for filt in list(filters.keys()):
    passband = PassBand(file=filters[filt])
    self.dpass[filt] = passband

 def getPassbands(self, filter_props=None, filter_mode=None, suffix=None, prefix=None, dist_pc=None, error_fmt='e_%s', **kwargs):
  if self.dpass is None or len(self.dpass) < 1:
   return
  if filter_mode is None:
   filter_mode = self.filter_mode
  if filter_props is None:
   filter_props = self.filter_props
  filter_mode = checkList(filter_mode)
  filter_props = checkList(filter_props)
  dist_pc = dist_pc if dist_pc is not None else self.dist_pc
  if filter_mode is None or filter_props is None:
   return
  if self.dmag is None:
   self.dmag = OrderedDict()
  for filt in self.dpass:
   pb = self.dpass[filt]
   for mode in filter_mode:
    if not 'Obs' in mode or 'D' in mode:
     continue
    wave = self.wave_rest if 'R' in mode else self.wave
    flux, eflux = pb.getFluxPass(wave, self.data, error=self.error, mask=self.flag, **kwargs)
    for prop in filter_props:
     key = '%s_%s_%s' % (mode, prop, filt)
     if suffix is not None:
      key = '%s%s' % (key, suffix)
     if prefix is not None:
      key = '%s%s' % (prefix, key)
     ekey = error_fmt % key
     if 'Flux' == prop:
      self.dmag[key]  = flux
      self.dmag[ekey] = eflux
     if 'Mag' == prop:
      mag, emag = pb.fluxToMag(flux, error=eflux, system='Vega')
      self.dmag[key]  = mag
      self.dmag[ekey] = emag
     if 'AB' == prop:
      mag, emag = pb.fluxToMag(flux, error=eflux, system='AB')
      self.dmag[key]  = mag
      self.dmag[ekey] = emag
     if ('Lum' == prop or 'L' == prop) and dist_pc is not None:
      lum, elum = pb.fluxToLuminosity(d_pc=dist_pc, flux=flux, error=eflux)
      if 'Lum' == prop:
       self.dmag[key]  = lum
       self.dmag[ekey] = elum
      if 'L' == prop:
       sun_lum = pb.sunLuminosity()
       self.dmag[key]  = lum / sun_lum
       self.dmag[ekey] = elum / sun_lum if elum is not None else None
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def RestWave(z,crval=None,cdelt=None,lspecaxis=None,wave=None):
 '''
 Returns a Rest-Frame wavelength array given a redshift z

 Parameters
 ----------
 z : float
	Redshift of the object
 crval : None or float, optional
 	CRVAL value for the wavelength array (starting value)
 cdelt : None or float, optional
	CDELT value for the wavelength array (step value)
 lspecaxis : None or integer, optional
	Number of elements in the wavelength array
 wave : None or numpy array, optional
	Observed wavelength array. If given, all properties will be 
	calculated from this array. A simple method is used to 
	estimate the step (CDELT), so it is advisable to provide 
	the CDELT parameter as well

 Warning
 -------
 If "wave" is not provided, then "crval", "cdelt" and "lspecaxis" should be 
 given. If provided, any additional value (crval,cdelt,lspecaxis) will 
 be used instead of the associated "wave" property
 '''

 if (crval is None or cdelt is None or lspecaxis is None) and wave is None:
  sys.exit('*** You need to provide "wave" or "[crval,cdelt,lspecaxis]"!!! ***')

 if wave is not None:
  if crval is None:      crval = wave[0]
  if lspecaxis is None:  lspecaxis = len(wave)
  if cdelt is None:      cdelt = np.unique(np.diff(wave)).mean()

 return (crval/(1.0+z)) + (cdelt/(1.0+z))*np.arange(lspecaxis)
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def Nsort(l): 
 """ Sort the given iterable in the way that humans expect.""" 
 convert = lambda text: int(text) if text.isdigit() else text 
 alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
 return sorted(l, key = alphanum_key)
# -------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def CreateAtpyTable(vars,hvars=None,hsort=None,fname=None,writer=None,**kwargs):
 import asciitable
 import atpy
 if hvars is not None:
  if len(hvars) != len(vars):
   sys.exit('*** "vars" (%s) and "hvars" (%s) should have the same length!!! ***' % (len(vars),len(hvars)))
  else:
   names = hvars
 else:
  names = ['col'+str(i+1) for i in range(len(vars))]
 table = atpy.Table()
 for var,hvar in zip(vars,names):
  table.add_column(hvar,var)
 if isinstance(hsort,int):
  table.sort(names[hsort])
 if isinstance(hsort,types.StringType):
  table.sort(hsort)
 if fname is not None:
  if writer is None: writer = asciitable.FixedWidth
  if 'delimiter' not in kwargs.keys(): kwargs['delimiter'] = None
  table.write(fname,type='ascii',Writer=writer,overwrite=True,**kwargs)
 else:
  return table
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def CreateTable(vars,hvars=None,hsort=None,fname=None,writer=None,formats=None,**kwargs):
 if hvars is not None:
  if len(hvars) != len(vars):
   sys.exit('*** "vars" (%s) and "hvars" (%s) should have the same length!!! ***' % (len(vars),len(hvars)))
  else:
   names = hvars
 else:
  names = ['col'+str(i+1) for i in range(len(vars))]
 table = Table(vars,names=hvars)
 if not isinstance(hsort,(list,tuple)):
  hsort = [hsort]
 if isinstance(hsort,(list,tuple)):
  if isinstance (hsort[0],int):
   hsort = [names[i-1] for i in hsort]
  table.sort(hsort)
 if fname is not None:
  from astropy.io import ascii
  if writer is None: writer = 'fixed_width'
  if 'delimiter' not in kwargs.keys(): kwargs['delimiter'] = None
  ascii.write(table,fname,format=writer,formats=formats,**kwargs)
 else:
  return table
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def SaveSpecFits(output,wave,flux,dw=None,w1=None,w2=None,origfile=None,
	info=None,header=None,dkeys=None,suf='fits',savewave=False):
 '''
 Save a spectra in FITS with the corresponding WCS wavelength information

 Parameters
 ----------
 output : string
	String with the name of the output FITS file
 wave : Numpy array
	Wavelength array
 flux : Numpy array
	Flux array
 dw : Float
	Wavelength step. If not provided, uses  the  wavelength  values  to  
	estimate them   to   a   linear dispersion  between  the  first  and 
	last wavelength values. The dispersion per pixel is  determined  by  
	the  number  of pixels and the endpoint wavelengths
 w1 : Float
  	Starting value for the wavelength array (if wants to be cutted)
 w2 : Float
  	Final value for the wavelength array (if wants to be cutted)
 origfile : String
	Name of the original file to be written in the header
 info : String
	Info to be added to the header (INFO keywords will be used)
 header : Pyfits Header
	Header to be added to the header of the new created FITS file
 dkeys : List, tuple of dictionary
	List/tuple or dictionary to be added to the header. If "dkeys" is a 
	list/tuple, should be of the form: 
		[['key1','value1'],['key2','value2'],...]
 suf : String
	Suffix for the FITS output file
 savewave : Bool
	Save the "wave" array in an additional HDU (in case of doubt of the
        estimation of CRVAL1 and CDELT1)
 '''

 output = '.'.join([output.replace('.fits','').replace('..','.'),suf])

 if w1 is None:
  w1 = wave[0]
 if w2 is None:
  w2 = wave[-1]

 idw = np.bitwise_and(wave >= w1,wave <= w2)
 wave = wave[idw]
 flux = flux[idw]
 w1 = wave[0]
 w2 = wave[-1]

 if dw is None:
  dw = (w2-w1)/float(flux.size-1)

 hdu = pyfits.PrimaryHDU(flux,header=header)
 if dkeys is not None:
  if isinstance(dkeys,(list,tuple)):
   for item in lkeys:
    hdu.header.set(item[0],item[1])
  if isinstance(dkeys,dict):
   for key in sorted(dkeys.keys()):
    try:
     hdu.header.set(key,dkeys[key])
    except:
     hdu.header.set(key,str(dkeys[key]))
 hdu.header.set('CRVAL1',w1)
 hdu.header.set('CDELT1',dw)
 if origfile is not None:
  hdu.header.set('ORIGFILE',origfile)
 if info is not None:
  hdu.header.set('INFO',info)
 hdu.header.set('AUTHOR','RGB')
 hdulist = pyfits.HDUList([hdu])
 if savewave:
  hdulist.append(pyfits.ImageHDU(wave, name='WAVE'))
 hdulist.writeto(output,clobber=True)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class Spec2Starlight():

 def __init__(self,file,z,dw,w1=None,w2=None,seval='',cols=(0,1),kcorr=True,
	kpow=1,zlum=None,ext_flux=None,ext_wave=None,ext_error=None,ext_flag=None,
	getError=False,window=50,intNumpy=False,savetxt=True,output=None,suf='txt',
	wave_fmt='%5.1f',flux_fmt='%6.5e',flag_fmt='%2i',saveError=True,saveFlag=True,
	fill=False,**kwargs):
  '''
  Reads a FITS file and writes an ASCII file arranged for STARLIGHT

  Parameters
  ----------
  file : string
	String with the name of the FITS file or ASCII file. If not provided (None), is 
	expected to give "ext_flux", "ext_wave" and "output"
  z : Float
	Redshift value to restframe the spectra
  dw : Float or Numpy array
	Wavelength step (float) or wavelength array for the new rest-frame wavelength. If 
	"dw" is an array, it will be cutted according to "w1" and "w2" values
  w1 : Float
  	Starting value for the new restframe wavelength array
  w2 : Float
  	Final value for the new restframe wavelength array
  seval : string
	If the data has several dimensions, string with the index evalation. Ex:
		data = flux[0,0,:] --> seval = '[0,0,:]'
  cols : Tuple or List
	It gives the columns to be extracted from the ASCII file. For 1 column, it 
	represents the flux; for 2, wavelength and flux; for 3, wavelength, flux and 
	error; and for 4, wavelength, flux, error and flag
  kcorr : Bool
	Apply K-correction to the spectra, multiply the flux and error by (1+z)**kpow
  kpow : Float or Integer
	Power of the K-correction (1+z)**kpow for the flux. For Starlight, kpow=1 is only 
	needed if the Luminosity Distance is provided in the Starlight grid file, since 
	the Luminosity Distance contains alredady a (1+z)**2 factor
  zlum : Float
	Redshift equivalent to the luminosity distance to apply for K-correction. If not
	given, "z" will be used
  ext_flux : Numpy array
	Array with the flux of the spectra
  ext_wave : Numpy array
	Array with the wavelength of the spectra
  ext_error : Numpy array
	External Error file. Restframed but not interpolated (previous to interpolation by "dw")
  ext_flag : Numpy array
	External Flag file. Restframed but not interpolated (previous to interpolation by "dw")
  getError : Bool
	If the error is not included, estimate the error [uses pySherpa.EstimateFluxError()]
  window : int
	Number of items (NOT wavelength) to consider in the window for "getError" [see 
	pySherpa.EstimateFluxError()]
  intNumpy : Bool
	Uses pystarlight ReSamplingMatrixNonUniform() for interpolation (flux conservation). 
	Otherwise, "numpy.interp"
  savetxt : Bool
	Save prepared starlight input spectrum file to ASCII
  output : string
	Output name of the starlight spectrum file. If not give, uses "file" root's name
  suf : string
	Suffix to add to the "output" file if "output = None"
  wave_fmt : string
	Format for the wavelength column
  flux_fmt : string
	Format for the flux AND error columns
  flag_fmt : string
	Format for the flag column
  saveError : Bool
	Write error column in ASCII file
  saveFlag : Bool
	Write flag column in ASCII file
  fill : Bool
	It fills with zeros if saveError or saveFlag are True but are not provided
  **kwargs 
	Variables for LoadFits() function or numpy.loadtxt()

  Returns
  -------
  Object: obj.zwave, obj.owave, obj.oflux, obj.oflag, obj.wave, obj.flux, obj.error, obj.flag, 
	obj.file 

  USE:

  	error = pyCalifa.LoadFits('err_file.fits',z=z,hcdelt='CD1_').data[0,0,:]
  	obj = pyCalifa.Spec2Starlight('data.fits',z,dw,seval='[0,0,:]',saveFlag=False,
		hcdelt='CD1_',ext_error=error)
  '''

  self.oerror = None
  self.oflag  = None
  self.error  = None
  self.flag   = None

  if file is not None:
   self.file = file
   ftype = Popen(['file', file], stdout=PIPE).communicate()[0]
   # ASCII
   if (ftype.find('ASCII') != -1):
    if len(cols) == 1:
     self.oflux = np.loadtxt(file,usecols=cols,unpack=True,**kwargs)
    elif len(cols) == 2:
     self.zwave, self.oflux = np.loadtxt(file,usecols=cols,unpack=True,**kwargs)
     self.RestWave(z)
    elif len(cols) == 3:
     self.zwave, self.oflux, self.oerror = np.loadtxt(file,usecols=cols,unpack=True,**kwargs)
     self.RestWave(z)
    else:
     self.zwave, self.oflux, self.oerror, self.oflag = np.loadtxt(file,usecols=cols,unpack=True,**kwargs)
     self.RestWave(z)
   # FITS
   else:
    obj = LoadFits(file,z=z,verbo=False,**kwargs)
    self.zwave = obj.wave
    self.owave = obj.wave_rest
    self.oflux = eval('obj.data'+seval)
    self.oerror = eval('obj.error'+seval)
    self.oflag = eval('obj.flag'+seval)
  else:
   if output is None:
    sys.exit('*** You need and output file name!!! ***')

  if ext_flux is not None:
   self.oflux = ext_flux
 
  if ext_wave is not None:
   self.zwave = ext_wave
   self.RestWave(z)

  if isinstance(dw,(list,tuple,np.ndarray)):
   self.wave = dw
   if w1 is not None and w2 is None:
    self.wave = self.wave[self.wave >= w1]
   if w1 is None and w2 is not None:
    self.wave = self.wave[self.wave <= w2]
   if w1 is not None and w2 is not None:
    self.wave = self.wave[np.bitwise_and(self.wave >= w1,self.wave <= w2)]
  else:
   if w1 is None:
    w1 = int(self.owave[0]) + 1
   if w2 is None:
    w2 = int(self.owave[-1]) - 1
   self.wave = np.arange(w1,w2,dw)

  if ext_error is not None:
   self.oerror = ext_error 
  if ext_flag is not None:
   self.oflag = ext_flag

  self.Interp(intNumpy)

  if getError:
   self.EstimateError(window)

  if self.error is None and fill:
   self.error = np.zeros_like(self.flux)

  if self.flag is None and fill:
   self.flag = np.zeros_like(self.flux)

  if kcorr:
   if zlum is None:
    zlum = z
   self.Kcorr(zlum,kpow)

  if savetxt:
   self.WriteTxt(output,saveError=saveError,saveFlag=saveFlag,wave_fmt=wave_fmt,
        flux_fmt=flux_fmt,flag_fmt=flag_fmt,suf=suf)

 def RestWave(self,z):
  self.owave = self.zwave/(1.0 + z)

 def EstimateError(self,window):
  import pySherpa as ps
  self.oerror = ps.EstimateFluxError(self.oflux,window=window)
  self.error =  ps.EstimateFluxError(self.flux,window=window)

 def Interp(self,intNumpy=True):
  if intNumpy:
   self.flux = np.interp(self.wave,self.owave,self.oflux)
   if self.oerror is not None:
    self.error = np.interp(self.wave,self.owave,self.oerror)
   if self.oflag is not None:
    self.flag = np.interp(self.wave,self.owave,self.oflag)
  else:
   from pystarlight.util.StarlightUtils import ReSamplingMatrixNonUniform
   RSM = ReSamplingMatrixNonUniform(self.owave,self.wave)
   self.flux = np.dot(RSM,self.oflux)
   if self.oerror is not None:
    self.error = np.dot(RSM,self.oerror)
   if self.oflag is not None:
    self.flag = np.dot(RSM,self.oflag)

 def Kcorr(self,zlum,kpow):
  self.flux  *= np.power(1. + zlum, kpow)
  self.error *= np.power(1. + zlum, kpow)

 def WriteTxt(self,output=None,saveError=True,saveFlag=True,wave_fmt='%5.1f',
        flux_fmt='%6.5e',flag_fmt='%2i',suf='txt'):
  if output is None:
   output = os.path.basename('.'.join(self.file.split('.')[:-1])+'.'+suf)
  fmt = [wave_fmt,flux_fmt]
  farray = [self.wave,self.flux]
  if saveError and self.error is not None:
   farray.append(self.error)
   fmt.append(flux_fmt)
  if saveFlag and self.flag is not None:
   farray.append(self.flag)
   fmt.append(flag_fmt)
  WriteTable(output,farray,formats=fmt,header=False)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class ReadStarlightOutput():

 def __init__(self,slfile,distance=None,base=None,bfile=None,basesdir=None,
	dered=False,error=None,flag=None,ewave=None,fwave=None,redlaw='CCM',
	RV=3.1,Lsun=3.839e33,parsec=3.08568e18,redshift=None,plot=False,
	show=True,idfig=None,figsize=(12,8.5),tauY=1.42e8,tauI=1.42e9,
	log=False,setpop=False,savetxt=False,output=None,saveObs=True,
	saveSyn=True,saveRes=True,saveError=True,saveFlag=True,saveWei=True,
	resample=False,lresam=None,intNumpy=True,zlum=None,kpow=-1,wave_fmt='%5.1f',
	obs_fmt='%6.5e',flag_fmt='%6.5g',suf='txt',savefits=None,outfits=None,
	savewave=False,savekeys=True,calc=None,filters=None,sunfile=None,
	prop=None,zsun=0.019,table_type='starlight',**kwargs):
  '''
  Reads a STARLIGHT output ASCII file 

  Parameters
  ----------
  slfile : string
	String with the name of the STARLIGHT output ASCII file
  distance : Float
	Distance in Mpc of the object. It is readed from "slfile", but it can 
	be overwritten with this variable. It is used in constructing the 
	global synthetic spectra from the BASE
  base : BASE class
	BASE class containing the SSP
  bfile : string
	Name of the BASE file (ex: BASE.gsd01)
  basesdir : string
	Path containing the SSP in the "bfile". Both variables ("bfile" and "basesdir") 
	are compulsory to estimate the synthetic spectra in the whole BASE wavelength 
	range
  dered : Bool
	Estimate the reddening corrected synthetic spectra from "bfile"
  error : Numpy array
	Array with the error spectrum
  flag : Numpy array
	Array with the flag spectrum
  ewave : Numpy array
	Array with the wavelength corresponding to the "error" array. It is used to 
	interpolate "error" to the same length as "obj.f_obs"
  eflag : Numpy array
	Array with the wavelength corresponding to the "flag" array. It is used to 
	interpolate "flag" to the same length as "obj.f_obs". If not given, "ewave" 
	will be used instead
  redlaw : string
	Redlaw to perform the reddening correction of the synthetic spectra
  RV : float
	Value of the RV parameter [RV = AV / E(B-V)]
  Lsun : float
	Bolometric luminosity of the Sun (ergs/s). Used to build the synthetic spectra
  parsec : float
	Value of a parsec in cma
  redshift : Float
	Redshift of the object. If not provided, is calculated from "distance"
  zlum : Float
	Redshift equivalent to the luminosity distance to apply for K-correction. If given,
	the K-correction will be applied according to "kpow"
  kpow : Float or Integer
	Power of the K-correction (1+z)**kpow for the flux. For Starlight, kpow=1 is only 
	needed if the Luminosity Distance is provided in the Starlight grid file, since 
	the Luminosity Distance contains alredady a (1+z)**2 factor. If the value is 
	NEGATIVE, the K-correction will be undone (dividing by the factor)
  resample : Bool
	Resample synthetic spectrum created from the a BASE file ("base") to "lresam"
  lresam : Numpy Array
	Wavelength array to resample the synthetic spectrum created from a BASE file. If not
	provided, "wave" (from the starlight input) is used
  intNumpy : Bool
	Use "numpy.interp" to interpolate the synthetic spectrum ("synthetic"). Use
	RSM from pystarlight, otherwise.
  plot :  Bool
	Plot Stalight results using PlotStalight()

  show : Bool
	Interactive plotting for PlotStalight()
  idfig : Int
	ID of the figure for PlotStalight()
  figsize : Tuple/List
	Tuple/List with the size of the figure for PlotStalight()
  tauY : float
	Upper limit age to calculate fraction of Young population (see "log")
  tauI : float
	Upper limit age to calculate fraction of Intermediate population (see "log")
  log : Bool
	If "log = True", values provided in "tauI" and "tauY" are given in log10
  setpop : Bool
	Set up variables for estimation of fraction of Young/Intermediate/Old 
	populations
  savetxt : Bool
	Save spectra in ASCII file
  output : string
	Name of the ASCII file
  saveObs: Bool
	Save the Observed spectra in the ASCII file
  saveSyn: Bool
	Save the Synthetic spectra in the ASCII file
  saveRes: Bool
	Save the Residual spectra in the ASCII file
  saveError: Bool
	Save the Error spectra in the ASCII file (if "error" is provided)
  saveFlag: Bool
	Save the Flag/Error spectra in the ASCII file (if "flag" is provided)
  saveWei: Bool
	Save the Weight/Flag (Starlight outpu) spectra in the ASCII file
  wave_fmt : string
	Format for the wavelength column
  obs_fmt : string
	Format for the Observed, Error, Synthetic AND Residual spectra
  flag_fmt : string
	Format for the flag/wei column
  suf : String
	String suffix for the output ASCII file
  savefits : String, List or Tuple
	Type of spectra to save as a FITS file. It should contain at least 
	(case insensitive) 'obs' (Observed), 'syn' (Synthetic) or 'res' 
	(Residual)
  outfits : String, List or Tuple
	It should contain the same number of elements as "savefits" with the 
	corresponding name for each spectra. If not provided, a name will be 
	given according to name and type
  savewave : Bool
	Save the wavelength array in a second HDU (in case of doubt of the 
	estimation of CRVAL1 and CDELT1)
  savekeys : Bool
	Save the keywords of the Starlight fit (ob.slout.keywords)
  calc : List
	List with the properties to estimate the magnitudes:
	  calc = ['Obs','ObsD','ObsR','ObsRD','Syn','SynD','SynR','SynRD']
  filters : Dictionary
	Dictionary with the name of the filter and file used to calculate the 
	magnitudes [see getMagSpectrum()]. Example: {'r': 'filters/sloan_r.txt'}
  sunfile : String or pyfits object
	The Sun file data (wavelength and flux) used to calculate the Luminosity for
	the magnitudes [see getMagSpectrum()]
  prop : List
	List of properties to calculate the magnitudes. By default:
		prop = ['Flux','Mag','AB','Lum','L']
  zsun : float
	Reference value for the solar metallicity
  **kwargs :
	Kwargs variables for PlotStalight()

  Returns
  -------
  Object: obj.wave, obj.f_obs, obj.f_syn, obj.f_res, obj.base, obj.error, obj.flag, obj.bwave, 
	obj.synthspec, obj.av, obj.v0, obj.flux_unit, obj.fobs_norm, obj.distance_Mpc, obj.redshift,
	obj.metBase, obj.ageBase, obj. metBaseSize, obj.ageBaseSize, obj.redlaw, obj.oerror,
	obj.oflag, obj.filename, obj.slout, obj.magnitudes and PyCASSO-like properties
  '''

  from pystarlight.util.base import StarlightBase
  import pystarlight.io.starlighttable
  import atpy

  self.base  = base
  self.kcorr = None
  self.synthspec = None
  self.zsun = zsun
  self.Lsun  = Lsun
  self.parsec = parsec
  self.redshift = redshift
  self.filename = os.path.basename(slfile)
  self.slout = atpy.TableSet(slfile,type=table_type)
  self.setProp(distance,error=error,flag=flag,ewave=ewave,fwave=fwave)
  self.magnitudes = None

  if bfile is not None and basesdir is not None:
   self.base  = StarlightBase(bfile,basesdir)

  self.setSynth(dered=dered,redlaw=redlaw,RV=RV,resample=resample,intNumpy=intNumpy)

  if setpop:
   self.setAgePopulations(tauY=tauY,tauI=tauI,log=log)

  if zlum is not None:
   self.Kcorr(zlum,kpow)

  if calc is not None and filters is not None:
   self.getMag(filters=filters,calc=calc,prop=prop,sunfile=sunfile,RV=RV)

  if savetxt:
   self.WriteTxt(output,saveObs=saveObs,saveSyn=saveSyn,saveRes=saveRes,saveFlag=saveFlag,
	saveWei=saveWei,saveError=saveError,wave_fmt=wave_fmt,
	obs_fmt=obs_fmt,flag_fmt=flag_fmt,suf=suf)

  if savefits is not None:
   self.SaveFits(savefits,output=outfits,savewave=savewave,savekeys=savekeys)

  if plot:
   self.plot(show=show,idfig=idfig,figsize=figsize,**kwargs)

 def setProp(self,distance,error=None,flag=None,ewave=None,fwave=None):
  self.error  = error
  self.flag   = flag
  self.oerror = error
  self.oflag  = flag

  self.distance_Mpc =  self.slout.keywords['LumDistInMpc'] if 'LumDistInMpc' in self.slout.keywords else None
  if distance is not None:
   self.distance_Mpc = distance
  if self.distance_Mpc is not None and self.redshift is None:
   self.redshift = self.distance_Mpc*CST.H0*1.0e5/CST.c
  self.flumdist = 4*np.pi*np.power(self.distance_Mpc*1.0e6*CST.PC,2.0) if self.distance_Mpc is not None else None

  # File names
  self.arq_spec = os.path.basename(self.slout.keywords['arq_spec'])
  self.arq_base = os.path.basename(self.slout.keywords['arq_base'])
  self.arq_masks = os.path.basename(self.slout.keywords['arq_masks'])
  self.arq_config = os.path.basename(self.slout.keywords['arq_config'])

  # Dimensions
  self.metBase     = np.unique(self.slout.population.popZ_base)
  self.ageBase     = np.unique(self.slout.population.popage_base)
  self.logMetBase  = np.log10(self.metBase/self.zsun) if self.zsun is not None else np.log10(self.metBase)
  self.metBaseSize = len(self.metBase)
  self.ageBaseSize = len(self.ageBase)
  self.nBase       = self.slout.keywords['N_base']
  self.nChains     = self.slout.keywords['N_chains']

  # Mask for non-square Bases
  self.baseMask = np.zeros((self.metBaseSize, self.ageBaseSize), dtype='bool')
  for a, Z in zip(self.slout.population.popage_base, self.slout.population.popZ_base):
   i = np.where(self.metBase == Z)[0]
   j = np.where(self.ageBase == a)[0]
   self.baseMask[i, j] = True

  # Synthesis Results - Best model
  self.chi2 = self.slout.keywords['chi2']
  self.adev = self.slout.keywords['adev']
  self.chi2_tot = self.slout.keywords['chi2_TOT'] if 'chi2_TOT' in self.slout.keywords else None

  self.flux_unit = self.slout.keywords['flux_unit'] if 'flux_unit' in self.slout.keywords else None
  self.fobs_norm = self.slout.keywords['fobs_norm']
  self.Lobs_norm = self.slout.keywords['Lobs_norm'] if 'Lobs_norm' in self.slout.keywords else None
  self.Flux_tot  = self.slout.keywords['Flux_tot']
  self.qnorm     = self.slout.keywords['q_norm']
  self.Mcor_tot  = self.slout.keywords['Mcor_tot']
  self.Mini_tot  = self.slout.keywords['Mini_tot']
  self.av        = self.slout.keywords['A_V']
  self.v0        = self.slout.keywords['v_0']
  self.vd        = self.slout.keywords['v_d']

  self.wave  = self.slout.spectra.l_obs
  self.f_obs = self.slout.spectra.f_obs*self.fobs_norm
  self.f_syn = self.slout.spectra.f_syn*self.fobs_norm
  self.f_res = self.f_obs-self.f_syn
  self.f_wei = self.slout.spectra.f_wei
  id_f_wei = self.f_wei > 0.0
  self.f_wei[id_f_wei] = self.f_obs[id_f_wei] / self.f_wei[id_f_wei]
  self.wave_obs = None
  if self.redshift is not None:
   self.wave_obs  = getWaveShift(self.wave,-self.redshift)

  if np.all(self.baseMask):
   self.popx      = self.slout.population.popx.reshape([self.metBaseSize, self.ageBaseSize]).T
   self.popmu_cor = self.slout.population.popmu_cor.reshape([self.metBaseSize, self.ageBaseSize]).T
   self.popmu_ini = self.slout.population.popmu_ini.reshape([self.metBaseSize, self.ageBaseSize]).T
   try: 
    self.popAV_tot = self.slout.population.popAV_tot.reshape([self.metBaseSize, self.ageBaseSize]).T
    self.popexAV_flag = self.slout.population.popexAV_flag.reshape([self.metBaseSize, self.ageBaseSize]).T

    self.SSP_chi2r = self.slout.population.SSP_chi2r.reshape([self.metBaseSize, self.ageBaseSize]).T
    self.SSP_adev  = self.slout.population.SSP_adev.reshape([self.metBaseSize, self.ageBaseSize]).T
    self.SSP_AV    = self.slout.population.SSP_AV.reshape([self.metBaseSize, self.ageBaseSize]).T
    self.SSP_x     = self.slout.population.SSP_x.reshape([self.metBaseSize, self.ageBaseSize]).T
   except:
    pass

  else:
   # Create Arrays
   shape2d            = [self.metBaseSize, self.ageBaseSize]
   self.popx          = new_array(shape2d, dtype='>f8')
   self.popmu_cor     = new_array(shape2d, dtype='>f8')
   self.popmu_ini     = new_array(shape2d, dtype='>f8')
   self.popfbase_norm = new_array(shape2d, dtype='>f8')
   self.popAV_tot     = new_array(shape2d, dtype='>f8')
   self.popexAV_flag  = new_array(shape2d, dtype='>f8')

   self.SSP_chi2r = new_array(shape2d, dtype='>f8')
   self.SSP_adev  = new_array(shape2d, dtype='>f8')
   self.SSP_AV    = new_array(shape2d, dtype='>f8')
   self.SSP_x     = new_array(shape2d, dtype='>f8')

   self.popx[self.baseMask]         = self.slout.population.popx
   self.popmu_cor[self.baseMask]    = self.slout.population.popmu_cor
   self.popmu_ini[self.baseMask]    = self.slout.population.popmu_ini
   self.popAV_tot[self.baseMask]    = self.slout.population.popAV_tot
   self.popexAV_flag[self.baseMask] = self.slout.population.popexAV_flag

   self.SSP_chi2r[self.baseMask] = self.slout.population.SSP_chi2r
   self.SSP_adev[self.baseMask]  = self.slout.population.SSP_adev
   self.SSP_AV[self.baseMask]    = self.slout.population.SSP_AV
   self.SSP_x[self.baseMask]     = self.slout.population.SSP_x

   # Transpose as in OLD mode [Age, Metallicity]
   self.popx         = self.popx.T
   self.popmu_cor    = self.popmu_cor.T
   self.popmu_ini    = self.popmu_ini.T   
   self.popAV_tot    = self.popAV_tot.T   
   self.popexAV_flag = self.popexAV_flag.T
                     
   self.SSP_chi2r = self.SSP_chi2r.T
   self.SSP_adev  = self.SSP_adev.T 
   self.SSP_AV    = self.SSP_AV.T   
   self.SSP_x     = self.SSP_x.T    

  try:    
   self.chains_best_par = self.slout.chains_par.best
   self.chains_ave_par  = self.slout.chains_par.average
   self.chains_par      = self.slout.chains_par.chains

   self.chains_best_LAx = self.slout.chains_LAx.best
   self.chains_ave_LAx  = self.slout.chains_LAx.average
   self.chains_LAx      = self.slout.chains_LAx.chains

   self.chains_best_mu_cor = self.slout.chains_mu_cor.best
   self.chains_ave_mu_cor  = self.slout.chains_mu_cor.average
   self.chains_mu_cor      = self.slout.chains_mu_cor.chains
  except:
   pass

  if ewave is not None and fwave is None:
   fwave = ewave
  if error is not None:
   self.error = error if ewave is None else np.interp(self.wave,ewave,error)
  if flag is not None:
   self.flag = flag if fwave is None else np.interp(self.wave,fwave,flag)

 def setAgePopulations(self,tauY=1.42e8,tauI=1.42e9,log=False):
  self.tauY = np.power(10., tauY) if log else tauY
  self.tauI = np.power(10., tauI) if log else tauI
  self.aBtauY = self.ageBase <= self.tauY
  self.aBtauI = np.bitwise_and(self.ageBase > self.tauY, self.ageBase <= self.tauI)
  self.aBtauO = self.ageBase > self.tauI

 @property
 def Lobn__tZ(self):
  tmp = self.popx / 100.
  tmp *= self.Lobs_norm
  return tmp

 @property
 def Lobn(self):
  return self.Lobn__tZ.sum(axis=1).sum(axis=0)

 @property
 def DeRed_Lobn__tZ(self):
  return self.Lobn__tZ * np.power(10.0, 0.4 * self.q_norm * self.av)

 @property
 def DeRed_Lobn(self):
  return self.DeRed_Lobn__tZ.sum(axis=1).sum(axis=0)

 @property
 def Mcor__tZ(self):
  tmp = self.popmu_cor / 100.0
  tmp *= self.Mcor_tot
  return tmp

 @property
 def Mcor(self):
  return self.Mcor__tZ.sum(axis=1).sum(axis=0)

 @property
 def Mini__tZ(self):
  tmp = self.popmu_ini / 100.0
  tmp *= self.Mini_tot
  return tmp

 @property
 def Mini(self):
  return self.Mini__tZ.sum(axis=1).sum(axis=0)

 @property
 def M2L(self):
  return self.Mcor / self.Lobn

 @property
 def DeRed_M2L(self):
  return self.Mcor / self.DeRed_Lobn

 @property
 def popx__Z(self):
  return self.popx.sum(axis=0)

 @property
 def popx__t(self):
  return self.popx.sum(axis=1)

 @property
 def popx_sum(self):
  return self.popx.sum(axis=1).sum(axis=0)

 @property
 def popx_norm(self):
  return self.popx / self.popx_sum

 @property
 def popx__t_norm(self):
  return self.popx__t / self.popx_sum

 @property
 def popx__Z_norm(self):
  return self.popx__Z / self.popx_sum

 @property
 def popmu_cor_sum(self):
  return self.popmu_cor.sum(axis=1).sum(axis=0)

 @property
 def popmu_cor_norm(self):
  return self.popmu_cor / self.popmu_cor_sum

 @property
 def popmu_cor__Z(self):
  return self.popmu_cor.sum(axis=0)

 @property
 def popmu_cor__t(self):
  return self.popmu_cor.sum(axis=1)

 @property
 def popmu_cor__Z_norm(self):
  return self.popmu_cor__Z / self.popmu_cor_sum

 @property
 def popmu_cor__t_norm(self):
  return self.popmu_cor__t / self.popmu_cor_sum

 @property
 def at_flux(self):
  '''
  Flux-weighted average log. age [light weighted mean (log) age image]

     * Units: :math:`[\log Gyr]`
     *  Type: float
  '''
  return np.tensordot(self.popx__t_norm, np.log10(self.ageBase), (0, 0))

 @property
 def at_mass(self):
  '''
  Mass-weighted average log. age [mass  weighted mean (log) age image]

     * Units: :math:`[\log Gyr]`
     *  Type: float
  '''
  return np.tensordot(self.popmu_cor__t_norm, np.log10(self.ageBase), (0, 0))

 @property
 def aZ_flux(self):
  '''
  Flux-weighted average metallicity (light weighted mean Z)

     * Units: dimensionless
     *  Type: float
  '''
  return np.tensordot(self.popx__Z_norm, self.metBase, (0, 0))

 @property
 def aZ_mass(self):
  '''
  Mass-weighted average metallicity (mass weighted mean Z)

     * Units: dimensionless
     *  Type: float
  '''
  return np.tensordot(self.popmu_cor__Z_norm, self.metBase, (0, 0))

 @property
 def alogZ_flux(self):
  '''
  Flux-weighted average metallicity in log scale according to RGD14 (log light weighted mean Z)

     * Units: dimensionless
     *  Type: float
  '''
  return np.tensordot(self.popx__Z_norm, self.logMetBase, (0, 0))

 @property
 def alogZ_mass(self):
  '''
  Mass-weighted average metallicity in log scale according to RGD14 (log mass weighted mean Z)

     * Units: dimensionless
     *  Type: float
  '''
  return np.tensordot(self.popmu_cor__Z_norm, self.logMetBase, (0, 0))

 @property
 def Ypopx(self):
  return self.popx_norm[self.aBtauY].sum(axis=1).sum(axis=0)

 @property
 def Ipopx(self):
  return self.popx_norm[self.aBtauI].sum(axis=1).sum(axis=0)

 @property
 def Opopx(self):
  return self.popx_norm[self.aBtauO].sum(axis=1).sum(axis=0)

 @property
 def Ypopmu_cor(self):
  return self.popmu_cor_norm[self.aBtauY].sum(axis=1).sum(axis=0)

 @property
 def Ipopmu_cor(self):
  return self.popmu_cor_norm[self.aBtauI].sum(axis=1).sum(axis=0)

 @property
 def Opopmu_cor(self):
  return self.popmu_cor_norm[self.aBtauO].sum(axis=1).sum(axis=0)

 def setSynth(self,dered=False,redlaw='CCM',RV=3.1,resample=False,lresam=None,intNumpy=True):
  if self.base is not None:
   self.bwave = self.base.l_ssp
   Lsun_to_flux = self.Lsun / (4.0 * np.pi * (self.distance_Mpc * 1.0e6 * self.parsec)**2)
   self.synthspec = Lsun_to_flux*np.tensordot(self.Mini__tZ,self.base.f_ssp,((1,0),(0,1))).T
   if not dered:
    from pystarlight.util.redenninglaws import calc_redlaw
    self.redlaw = calc_redlaw(self.bwave, RV, redlaw=redlaw) # Reddening law
    self.synthspec *= np.power(10., -0.4 * self.redlaw * self.av)
   if self.kcorr is not None:
    self.synthspec *= self.kcorr
  if resample:
   if lresam is None:
    lresam = self.wave
   self.synthspec = Interp(self.bwave,self.synthspec,lresam,intNumpy=intNumpy,extrap=True)
   self.bwave = lresam

 def getKinematics(self,data,v0=None,vd_ini=0.0,vd_end=None,wave=None):
  if v0 is None:
   v0 = self.v0
  if vd_end is None:
   vd_end = self.vd
  if wave is None:
   wave = self.wave
  return setKinematics(wave,data,v0,vd_ini,vd_end)

 def Kcorr(self,zlum,kpow):
  self.kcorr = np.power(1. + zlum, kpow)
  self.f_obs *= self.kcorr
  self.f_syn *= self.kcorr
  self.f_res *= self.kcorr

 def getMag(self,filters,calc,prop=None,sunfile=None,RV=3.1):
  self.magnitudes = OrderedDict()
  for key in calc:
   AV    = self.av if 'D' in key else None
   flux  = self.f_obs if 'Obs' in key else self.f_syn
   wave  = self.wave if 'R' in key else self.wave_obs
   gm    = getMagSpectrum(filters,wave,flux,error=self.error,flag=self.flag,sunfile=sunfile,
	RV=RV,AV=AV,redshift=None,distance_Mpc=self.distance_Mpc,finite=True,prop=prop)
   self.magnitudes[key] = gm.magnitudes

 def WriteTxt(self,output=None,saveObs=True,saveSyn=True,saveRes=True,saveError=True,
	saveFlag=True,saveWei=True,wave_fmt='%5.1f',obs_fmt='%6.5e',flag_fmt='%6.5g',
	save_order=None,suf='txt',header=True,hWave='Wave',hObs='ObSpec',hSyn='SynSpec',
	hRes='ResidualSpec',hError='Error',hWei='Weight',hFlag='Flag'):
  if save_order is None:
   save_order = [hObs,hError,hSyn,hRes,hWei,hFlag]
  if hWave not in save_order:
   save_order.insert(0,hWave)
  if output is None:
   output = self.filename.replace('.txt','').replace('.dat','')
   output = '.'.join([output,'spec',suf])
  dic = {hWave: {'data': self.wave, 'fmt': wave_fmt}}
  if saveObs:
   dic[hObs] = {'data': self.f_obs, 'fmt': obs_fmt}
  if saveError and self.error is not None:
   dic[hError] = {'data': self.error, 'fmt': obs_fmt}
  if saveSyn:
   dic[hSyn] = {'data': self.f_syn, 'fmt': obs_fmt}
  if saveRes:
   dic[hRes] = {'data': self.f_res, 'fmt': obs_fmt}
  if saveWei:
   dic[hWei] = {'data': self.f_wei, 'fmt': flag_fmt}
  if saveFlag and self.flag is not None:
   dic[hFlag] = {'data': self.flag, 'fmt': flag_fmt}
  farray = [dic[key]['data'] for key in save_order if key in dic]
  fmt    = [dic[key]['fmt'] for key in save_order if key in dic]
  names  = [key for key in save_order if key in dic]
  WriteTable(output,farray,formats=fmt,hvars=names,header=header)

 def SaveFits(self,savefits,output=None,savewave=False,savekeys=True):
  if not isinstance(savefits,(tuple,list)):
   savefits = [savefits]
  if output is not None:
   if not isinstance(output,(tuple,list)):
    output = [output]

  for i,item in enumerate(savefits):
   item = item.lower()
   if 'obs' in item:
    sflux = self.f_obs
    suf = 'obs'
    info = 'Observed Spectra'
   if 'syn' in item:
    sflux = self.f_syn
    suf = 'syn'
    info = 'Synthetic Spectra'
   if 'res' in item:
    sflux = self.f_res
    suf = 'res'
    info = 'Residual Spectra'
   if output is None:
    sname = '.'.join([self.filename,suf]).replace('txt','').replace('dat','').replace('..','.')
   else:
    sname = output[i]
   dkeys = self.slout.keywords if savekeys else None
   SaveSpecFits(sname,self.wave,sflux,origfile=self.filename,
	info=info,savewave=savewave,dkeys=dkeys)

 def plot(self,show=True,idfig=None,figsize=(11.7,8.1),**kwargs):
  PlotStarlight(self,show=show,idfig=idfig,figsize=figsize,**kwargs)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class PlotStarlight(object):

 def __init__(self,sl_obj,show=True,idfig=None,figsize=(11.7,8.1),**kwargs):
  '''
  Plots the fits and/or SFH of STARLIGHT 

  USE:

	obj = pyCalifa.ReadStarlightOutput('file.out')
        pyCalifa.PlotStarlight(obj.slout,basefile='BASE.bca01',
		basedir='/Users/rgb/',fewhb=True,legend=False)

     or

	pyCalifa.ReadStarlightOutput('forshost.bca01.out',plot=True)
  '''

  import matplotlib
  import matplotlib.pyplot as plt
    
  self.sl_obj = None
  self.sl_out = None
  try:
   self.sl_out = sl_obj.slout
   self.sl_obj = sl_obj
  except:
   self.sl_out = sl_obj
  self.fig = plt.figure(idfig, figsize = figsize)

  if show:
   matplotlib.pyplot.interactive(False)
  else:
   matplotlib.pyplot.interactive(True)

  self.plot(idfig=idfig,**kwargs)

  if show:
   plt.show()

 # Main Plotting function
 def plot(self,Bwidth=None,spec=True,alpha=0.8,zcol=None,clip=True,basefile=None,basedir=None,
	basefiledir=None,fewhb=False,lhb=[4850.0,4870.0],lage=[10.0e6,20.0e6],loc=2,legend=True,
	mass_lim=[0.0,100.0],light_lim=[0.0,100.0],spec_lim=[0,3],res_lim=[-0.5,0.5],age_lim=[5.5,10.5],
	wave_ticks=[500,100],spec_ticks=[1,0.2],res_ticks=[0.2,0.1],light_ticks=[20.0,10.0],
	mass_ticks=[20.0,10.0],age_ticks=[1.0,0.2],axticks=True,wave_axlim=True,savefig=False,
	outfile=None,fmt='png',collapse_mass=True,log_mass=False,collapse_light=False,log_light=False,
	norm=True,oldplot=False,idfig=None,close=False,sortbars=True,true_width=True,log_age=None,
	balign='center',bfill=True,lw=None,edge_color='k',zfmt='%.4f',cumsum=False):

  if zcol is None:
   zcol = ['b','#9fb6cd','#996666','#800080','#800000','#667c26']
   #matplotlib.rcParams['axes.color_cycle'] = zcol

  filename = self.sl_out.keywords['arq_synt']

  self.oldplot = True
  if self.sl_obj is not None:
   if not (np.all(self.sl_obj.baseMask) and oldplot):
    self.oldplot = False

  if self.oldplot:
   xj = self.sl_out.population.popx
   mu_cor = self.sl_out.population.popmu_cor
   if log_age is None:
    log_age = np.log10(self.sl_out.population.popage_base)
  else:
   if log_age is None:
    log_age = np.log10(self.sl_obj.ageBase)
   xj = self.sl_obj.popx
   mu_cor = self.sl_obj.popmu_cor
   mass_sum = self.sl_obj.popmu_cor__t
   light_sum = self.sl_obj.popx__t

  Z = self.sl_out.population.popZ_base
  uZ = np.unique(Z)
  sz = ' $\oplus$ '.join(map((lambda n: zfmt % n),uZ))

  av = self.sl_out.keywords['A_V']
  vd = self.sl_out.keywords['v_d']
  v0 = self.sl_out.keywords['v_0']
  chi2 = self.sl_out.keywords['chi2']
  mcor_tot = self.sl_out.keywords['Mcor_tot']
  fobs_norm = 1.0 if norm else self.sl_out.keywords['fobs_norm']

  # Spectra
  l_obs = self.sl_out.spectra.l_obs
  f_obs = self.sl_out.spectra.f_obs*fobs_norm
  f_syn = self.sl_out.spectra.f_syn*fobs_norm
  wmin = l_obs.min()
  wmax = l_obs.max()
  res = f_obs - f_syn

  # Define bar spacing
  if Bwidth is None:
   if true_width and not oldplot:
    Bwidth = np.append(np.diff(log_age), np.diff(log_age)[-1])
   else:
    d = log_age[1:] - log_age[:-1]
    Bwidth = np.min(d[d>0])

  # Option SFH + Spec&Fit
  if spec:
   box_mass  = [0.72,0.07,0.25,0.35]
   box_light = [0.72,0.45,0.25,0.35]
   box_spec = [0.06,0.35,0.6,0.45]
   box_res = [0.06,0.07,0.6,0.25]
   # Spectra & Fit
   self.ax_spec = self.fig.add_axes(box_spec) # sharex=self.ax_res, #xticklabels=[]
   self.ax_spec.plot(l_obs,f_obs,lw=0.5)
   self.ax_spec.plot(l_obs,f_syn,'r')
   if spec_lim is not None:
    self.ax_spec.set_ylim(self.MultiplyList(spec_lim,fobs_norm))
   if wave_axlim: 
    self.ax_spec.set_xlim(wmin,wmax)
   flux_units_label = r'$F_{\lambda}$ (normalized)' if norm else r'$F_{\lambda}$ (erg s$^{-1}$ cm$^{-2}$)'
   self.ax_spec.set_ylabel(flux_units_label)
   if axticks: 
    self.axticks(self.ax_spec,self.setVars(wave_ticks,spec_ticks)) 
   # Residual
   self.ax_res = self.fig.add_axes(box_res)
   self.ax_res.plot(l_obs,res,lw=0.5,color='blue')
   # Plot masked, clipped and flagged values
   if clip:
    aux = np.ma.masked_where(self.sl_out.spectra.f_wei < 0, res)
    self.ax_res.plot(l_obs, aux, color='black') #If wei > 0
    aux = np.ma.masked_where(self.sl_out.spectra.f_wei != 0, res)
    self.ax_res.plot(l_obs, aux, color='magenta') #If wei == 0 -> Masked
    aux = np.ma.masked_where(self.sl_out.spectra.f_wei != -1, res)
    self.ax_res.plot(l_obs, aux, 'x', color='red') #If wei == -1 -> Clipped
    aux = np.ma.masked_where(self.sl_out.spectra.f_wei != -2, res)
    self.ax_res.plot(l_obs, aux, 'o', color='green') #If wei == -2 -> Flagged
   if res_lim is not None: 
    self.ax_res.set_ylim(self.MultiplyList(res_lim,fobs_norm))
   if wave_axlim: 
    self.ax_res.set_xlim(wmin,wmax)
   self.ax_res.set_xlabel(r'Wavelength ($\AA$)')
   self.ax_res.set_ylabel('Residuals')
   if axticks: 
    self.axticks(self.ax_res,self.setVars(wave_ticks,res_ticks)) 
   #self.ax_res.set_yticks(self.ax_res.get_yticks()[:-1]) # Remove last lower ytick
  # Option SFH
  else:
   box_mass  = [0.1,0.1,0.8,0.35]
   box_light = [0.1,0.5,0.8,0.35]

  # Mass, Light & Text
  box_text = [0.08,0.86,0.85,0.10]
  self.ax_mass = self.fig.add_axes(box_mass)
  self.ax_light = self.fig.add_axes(box_light)
  self.ax_text = self.fig.add_axes(box_text,frameon=False,xticks=[],yticks=[])
  self.ax_text.axis([0.0,1.0,0.0,1.0])
  self.ax_mass.set_xlabel('Log(Age)')
  self.ax_mass.set_ylabel('Mass Fraction (%)')
  self.ax_light.set_ylabel('Light Fraction (%)')
   
  # SFH (Mass & Light)
  apl = np.array([])
  apm = np.array([])
  if self.oldplot:
   # Old SFH Plot -----------------------------------------------
   mass_sum = np.zeros(np.shape(Z[Z == Z[0]]))
   light_sum = np.zeros(np.shape(Z[Z == Z[0]]))
   for i,z in enumerate(uZ):
    ecolor = edge_color if bfill else zcol[i]
    idz = (Z == z)
    # Light
    if not collapse_light:
     lbottom = light_sum if cumsum else None
     tmpapl = self.ax_light.bar(log_age[idz],xj[idz],width=Bwidth,align=balign,fill=bfill,edgecolor=ecolor,
	color=zcol[i],label=(zfmt % z),alpha=alpha,log=log_light,lw=lw,bottom=lbottom)
     apl = np.append(apl,tmpapl)
    light_sum = light_sum + xj[idz] # After bar to work with cumsum
    # Mass
    if not collapse_mass:
     mbottom = mass_sum if cumsum else None
     tmpapm = self.ax_mass.bar(log_age[idz],mu_cor[idz],width=Bwidth,align=balign,fill=bfill,edgecolor=ecolor,
	color=zcol[i],label=(zfmt % z),alpha=alpha,log=log_mass,lw=lw,bottom=mbottom)
     apm = np.append(apm,tmpapm)
    mass_sum = mass_sum + mu_cor[idz]  # After bar to work with cumsum

   # Collapsed histogram
   if collapse_light:
    ecolor = edge_color if bfill else '#DA70D6'
    self.ax_light.bar(log_age[idz],light_sum,width=Bwidth,align=balign,fill=bfill,edgecolor=ecolor,
	color='#DA70D6',alpha=alpha,log=log_light,lw=lw)
   if collapse_mass:
    ecolor = edge_color if bfill else '#FF9966'
    self.ax_mass.bar(log_age[idz],mass_sum,width=Bwidth,align=balign,fill=bfill,edgecolor=ecolor,
	color='#FF9966',alpha=alpha,log=log_mass,lw=lw)
  # New SFH plot
  else:
   bmass_sum = np.zeros(np.shape(Z[Z == Z[0]]))
   blight_sum = np.zeros(np.shape(Z[Z == Z[0]]))
   if collapse_light:
    ecolor = edge_color if bfill else '#DA70D6'
    self.ax_light.bar(log_age,light_sum,width=Bwidth,align=balign,fill=bfill,edgecolor=ecolor,
	color='#DA70D6',alpha=alpha,log=log_light,lw=lw)
   else:
    for i,z in enumerate(uZ):
     ecolor = edge_color if bfill else zcol[i]
     lbottom = blight_sum if cumsum else None
     tmpapl = self.ax_light.bar(log_age,xj[:,i],width=Bwidth,align=balign,fill=bfill,edgecolor=ecolor,
	color=zcol[i],label=(zfmt % z),alpha=alpha,log=log_light,lw=lw,bottom=lbottom)
     apl = np.append(apl,tmpapl)
     blight_sum = blight_sum + xj[:,i] # After bar to work with cumsum
   if collapse_mass:
    ecolor = edge_color if bfill else '#FF9966'
    self.ax_mass.bar(log_age,mass_sum,width=Bwidth,align=balign,fill=bfill,edgecolor=ecolor,
	color='#FF9966',alpha=alpha,log=log_mass,lw=lw)
   else:
    for i,z in enumerate(uZ):
     ecolor = edge_color if bfill else zcol[i]
     mbottom = bmass_sum if cumsum else None
     tmpapm = self.ax_mass.bar(log_age,mu_cor[:,i],width=Bwidth,align=balign,fill=bfill,edgecolor=ecolor,
	color=zcol[i],label=(zfmt % z),alpha=alpha,log=log_mass,lw=lw,bottom=mbottom)
     apm = np.append(apm,tmpapm)
     bmass_sum = bmass_sum + mu_cor[:,i] # After bar to work with cumsum
  
  # zorder for bars in histogram
  if sortbars:
   if not collapse_light:
    self.sortHistBars(apl)
   if not collapse_mass:
    self.sortHistBars(apm)

  # Legends and Axis
  if len(uZ) > 1 and legend:
   if not collapse_light:
    self.ax_light.legend(loc=loc)
   if not collapse_mass:
    self.ax_mass.legend(loc=loc)
  self.ax_mass.axis(self.setVars(age_lim,mass_lim))
  self.ax_light.axis(self.setVars(age_lim,light_lim))
  if axticks:
   self.axticks(self.ax_light,self.setVars(age_ticks,light_ticks)) 
   self.axticks(self.ax_mass,self.setVars(age_ticks,mass_ticks)) 

  # Text
  self.ax_text.text(1.0,1.0,r'$v_\star$ = %.2f km/s' % v0,horizontalalignment='right')
  self.ax_text.text(1.0,0.5,r'$\sigma_\star$ = %.2f km/s' % vd,horizontalalignment='right')
  self.ax_text.text(1.0,0.0,r'A$_V$ = %.4f mag'  % av, horizontalalignment='right')
  self.ax_text.text(0.5,1.0,filename,horizontalalignment='center',fontsize=15)
  self.ax_text.text(0.5,0.5,r'$\chi^2$ = ' + ('%4.3f' % chi2),horizontalalignment='center')
  self.ax_text.text(0.5,0.0,'Z = '+ sz,horizontalalignment='center')
  self.ax_text.text(0.0,0.0,r'M$_{*}$ = %8.3e $\times$ M$_{\odot}$' % mcor_tot,fontsize=13)
  #self.ax_text.text(0.0,0.0,r'M$_{*}$ = %8.3e $\times$ D$^{2}$ M$_{\odot}$' % mcor_tot,fontsize=13)
  if fewhb:
   self.EWHb(basefile=basefile,basedir=basedir,basefiledir=basefiledir,lhb=lhb,lage=lage)
   self.ax_text.text(0.0,1.0,r'M$_{ion}$ =  %3.2f%% | %3.2f%%  (%2iMyr|%2iMyr)' % tuple(self.ionpop+self.age),fontsize=11)
   self.ax_text.text(0.0,0.5,r'EW(H$\beta$) = %5.3f | %5.3f $\frac{Cont_{ion}}{Cont_{tot}}$' % tuple(self.fewhb),fontsize=11)

  # Save Figure
  if savefig:
   self.savefig(outfile=outfile,fmt=fmt,idfig=idfig,close=close)

 # Function to plot bars according to Zorder ---------------------------------------------
 def sortHistBars(self,lpatch):
  #zor = np.lexsort(zip(*[(p._height,p._x) for p in lpatch]))
  x = np.array([p._x for p in lpatch])
  y = np.array([p._height for p in lpatch])
  zor = np.lexsort((y,x))
  d = {}
  for i,ix,iy in izip(count(),x[zor],y[zor]):
   d[(ix,iy)] = -i
  [p.set_zorder(d[(p._x,p._height)]) for p in lpatch]

 # Function to get the EW(Hb) contribution -----------------------------------------------
 def EWHb(self,basefile=None,basedir=None,basefiledir=None,lhb=[4850.0,4870.0],lage=[10.0e6,20.0e6]):
  if basedir is None:
   basedir = os.environ['HOME'] + '/iraf/starlight/BasesDir/'
  if basefile is None:
   if basefiledir is None:
    basefile = self.sl_out.keywords['arq_base']
   else:
    basefile = os.path.join(basefiledir,self.sl_out.keywords['arq_base'])

  baselist = np.loadtxt(basefile,skiprows=1,usecols=(0,),dtype=np.str)
  xj = self.sl_out.population.popx
  mu_cor = self.sl_out.population.popmu_cor

  fhb = []
  for i,bfile in enumerate(baselist):
   if (xj[i] > 0.0):
    wave,flux = np.loadtxt(os.path.join(basedir,bfile),unpack=True,usecols=(0,1))
    tmp_fhb = float(np.interp(list(lhb),wave,flux).sum()/2.0)
    fhb.append(tmp_fhb)
   else:
    fhb.append(0.0)
  fhb = np.array(fhb)

 # iipX = Poblacion ionizante
  age = self.sl_out.population.popage_base
  self.age = list(np.array(lage)/1.0e6)
  iip1 = (age <= lage[0])
  iip2 = (age <= lage[1])

 # Porcentaje de EW(HB)
  fewhb1  = sum(fhb[iip1]*xj[iip1]) / sum(fhb*xj)
  fewhb2  = sum(fhb[iip2]*xj[iip2]) / sum(fhb*xj)
  self.fewhb = [fewhb1,fewhb2]

 # Porcentaje masa ionizante
  pmi1 = sum(mu_cor[iip1])
  pmi2 = sum(mu_cor[iip2])
  self.ionpop = [pmi1,pmi2]

 # Ticks function ----------------------------------------------
 def axticks(self,ax,limits):
  from matplotlib.ticker import MultipleLocator
  xma,xmi,yma,ymi = limits
  if xma is not None:
   ax.xaxis.set_major_locator(MultipleLocator(xma))
  if xmi is not None:
   ax.xaxis.set_minor_locator(MultipleLocator(xmi))
  if yma is not None:
   ax.yaxis.set_major_locator(MultipleLocator(yma))
  if ymi is not None:
   ax.yaxis.set_minor_locator(MultipleLocator(ymi))

 # Save figure function -----------------------------------------
 def savefig(self, outfile=None, fmt='png', idfig=None, close=False):
  if outfile is None:
   outfile = '.'.join([self.sl_out.keywords['arq_synt'],fmt])
  self.fig.savefig(outfile, format=fmt)
  if close:
   import matplotlib.pyplot as plt
   if idfig is None:
    plt.close()
   else:
    plt.close(idfig)

 # Set variable for None cases ---------------------------------
 def setVar(self,var,n=2):
  if var is None:
   var = n * [None]
  return var 

 # Set variables -----------------------------------------------
 def setVars(self,var1,var2):
  var1 = self.setVar(var1)
  var2 = self.setVar(var2)
  return var1 + var2

 # Multiply Tuple -----------------------------------------------
 def MultiplyList(self,lvar,factor):
  if lvar is not None:
   lvar = tuple(item * factor if item is not None else item for item in lvar)
  return lvar
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class getGalProp(object):

 def __init__(self,gal,sunfile=None,filters=None,table=None,RV=3.1,dered=False,synth=False,rest=False,
	base=None,basesdir=None,basename=None,extensive=False,surface_density=False,set_smooth=False,idint=0,
	calc=None,lresam=None,dw=None,savecube=False,ncube=None,cwave=None,slinterp=False,extrap=False,
	redshift=None,res_ini=None,res_end=None,v0=None,vd_ini=None,vd_end=None,iv0=None,ivd_ini=None,
	ivd_end=None,kine=False,map2d=True,radial=False,masked=True,fill_value=np.nan,prop=None,
	kcorr=False,zlum=None,kpow=-1,nproc=1,fmt='%12.1f   %16.10e',use_atpy=True,calc_apertures=False,
	calc_bin_r_ap=False,dprop_radial=None,prop_radial_func=None,mode_ap=None,percentage_wave=None,
	vmask=np.nan,cube_file=None,check_masked=False,verbose=False,dcube={},**kwargs):
  '''
  USE:
	1) Save input SED in Cloudy format:
 	  obj = getGalProp(fits,sunfile,synth=True,dered=True,rest=True,base=base)
	  obj.CloudySED('sed.ascii',integrated=True)

	2) Compute passband magnitudes:
	  calc = ['Obs','ObsD','ObsR','ObsRD','Syn','SynD','SynR','SynRD']
	  filters = [['sloan_u.txt','u'],['sloan_g.txt','g'],['sloan_r.txt','r']]
	  ts = atpy.TableSet()
	  getGalProp(gal,sunfile,filters=filters,table=ts,base=base,calc=calc,savecube=True)
	  ts.write(name,overwrite=True,verbose=True)
  '''

  from pystarlight.util.StarlightUtils import ReSamplingMatrixNonUniform
  from pystarlight.util.redenninglaws import calc_redlaw
  from pystarlight.util.base import StarlightBase
  from Py3D.core.spectrum1d import Spectrum1D
  from pyfu.passband import PassBand

  if isinstance(gal, str):
   self.K = fitsDataCube(gal,**kwargs)
   self.filename = os.path.basename(gal)
  else:
   self.K = gal
   self.filename = self.K.namefits
  self.rootname = '.'.join(self.filename.split('.')[:-1])
  self.K.setSmoothDezonification(set_smooth) # Smooth = False & extensive = True --> Dividido por #pixeles Zona
  self.surface_density = surface_density
  self.extensive = extensive
  self.redshift = redshift
  self.sunfile = sunfile
  self.filters = filters
  self.base = base
  self.basesdir = basesdir
  self.basename = basename
  self.RV = RV
  self.intdict = {}
  self.sundict = {}
  self.cwave = cwave
  self.olresam = lresam
  self.extrap  = extrap
  self.res_ini = res_ini
  self.res_end = res_end
  self.v0 = v0
  self.vd_ini = vd_ini
  self.vd_end = vd_end
  self.iv0 = iv0
  self.ivd_ini = ivd_ini
  self.ivd_end = ivd_end
  self.kine = kine
  self.fill_value = fill_value
  self.masked = masked
  self.map2d = map2d
  self.radial = radial
  self.kcorr = kcorr
  self.zlum = zlum
  self.kpow = kpow
  self.kc = None
  self.nproc = nproc
  self.slinterp = slinterp
  self.savecube = savecube
  self.name_cube = ncube
  self.specname = None
  self.dw = dw
  self.fmt = fmt
  self.use_atpy = use_atpy
  self.KT = None
  self.cube = None
  # Modules
  self.PassBand = PassBand
  self.Spectrum1D = Spectrum1D
  self.calc_redlaw = calc_redlaw
  self.StarlightBase = StarlightBase
  self.ReSamplingMatrixNonUniform = ReSamplingMatrixNonUniform
  # Default props
  self.dprop = ['Flux','Mag','AB','Lum','L']
  self.dcalc = ['Obs','ObsD','ObsR','ObsRD','Syn','SynD','SynR','SynRD']

  self.checkCalc(calc)
  self.setRedshift(redshift=redshift)
  self.setProp(prop)
  filters = self.setTable(filters)

  if self.calc is None:
   self.Main(filters,synth,dered,rest,None,calc_apertures,mode_ap,percentage_wave=percentage_wave,vmask=vmask,bin_r=calc_bin_r_ap,check_masked=check_masked,verbose=verbose)
   if cube_file is not None and not dered and not synth:
    calc = 'ObsR' if rest else 'Obs'
    self.getMagCube(cube_file, filters=filters, calc=[calc], **updateDictDefault(dcube, dict(radial=self.radial, vmask=vmask, percentage_wave=percentage_wave, check_masked=check_masked,verbose=verbose)))
  else:
   for icalc in self.calc:
    isynth = True if 'Syn' in icalc else False
    irest  = True if 'R'   in icalc else False
    idered = True if 'D'   in icalc else False
    self.Main(filters,isynth,idered,irest,icalc,calc_apertures,mode_ap,percentage_wave=percentage_wave,vmask=vmask,bin_r=calc_bin_r_ap,check_masked=check_masked,verbose=verbose)
   if cube_file is not None:
    self.getMagCube(cube_file, filters=filters, **updateDictDefault(dcube, dict(radial=self.radial, vmask=vmask, percentage_wave=percentage_wave, check_masked=check_masked, verbose=verbose)))
  if dprop_radial is not None:
   self.getMagRadial(dprop_radial, func=prop_radial_func, check_masked=check_masked, verbose=verbose)
  self.Close(filters,table=table)

 def setRedshift(self, redshift=None, key_redshift='REDSHIFT'):
  self._redshift = redshift
  if self.redshift is None and key_redshift in self.K.header:
   self._redshift = float(self.K.header[key_redshift])
  if self._redshift is None:
   print ('>>> WARNING: Redshift NOT provided and NOT found in object header [%s]!' % key_redshift)

 def checkCalc(self, calc=None):
  self.scalc = None
  self.calc = checkList(calc)
  if self.calc is not None:
   self.calc = [calc.strip() for calc in self.calc]
   non_calc  = [calc for calc in self.calc if not calc in self.dcalc]
   self.calc = [calc for calc in self.calc if calc in self.dcalc]
   if len(non_calc) > 0:
    print ('>>> WARNING: Some calc properties [%s] NOT available [%s]' % (' | '.join(non_calc), ' | '.join(self.dcalc)))
   self.scalc = self.calc[:]

 def setTable(self,filters):
  if filters is not None:
   if self.use_atpy: 
    import atpy
    self.KT = atpy.Table(name=self.K.califaID)
   else:
    self.KT = Table(masked=self.masked)
   filters = self.getFilters(filters)
  if self.radial and self.map2d and self.use_atpy:
   self.radial = False
   print (' >>> ATPY does NOT allow to save 1D (radial) and 2D (map2d) information')
   print (' >>> Radial option has been switch off (radial = False). Use "use_atpy = False" instead') 
  return filters

 def getFilters(self,filters):
  if filters is None:
   return
  if isinstance(filters,dict):
   filters = [[filters[key],key] for key in filters]
  return filters

 def setProp(self,prop=None):
  if prop is None:
   self.prop = self.dprop
  else:
   if not isinstance(prop, (list,tuple)):
    prop = [prop]
   if all(key in self.dprop for key in prop):
    self.prop = prop
   else:
    sys.exit('*** Key properties [%s] NOT valid!!! [valid: %s] ***' % (','.join(prop),','.join(self.dprop)))

 def Main(self, filters, synth, dered, rest, spectype=None, apertures=False, mode_ap=None, percentage_wave=None,
	vmask=np.nan, bin_r=False, check_masked=False, verbose=False):
  self.getBase()
  self.GetSun()
  if filters is not None and self.sunfile is None:
   sys.exit('*** You need to provide the "sunfile" file in order to estimate Magnitudes ***')
  self.SetSpecType(synth,dered,rest,spectype)
  self.SetData(synth,dered,rest)
  self.SetWaveResolution()
  if self.kine:
   self.SetKinematics()
  if self.kcorr:
   self.SetKcorr()
   self.Kcorr()
  self.GetMag(filters,percentage_wave=percentage_wave,vmask=vmask,check_masked=check_masked,verbose=verbose)
  if apertures and self.K.apertures is not None:
   self.GetMagAp(filters,mode=mode_ap,percentage_wave=percentage_wave,vmask=vmask,bin_r=False,check_masked=check_masked,verbose=verbose)
  if bin_r and self.K.bin_apr is not None:
   self.GetMagAp(filters,mode=mode_ap,percentage_wave=percentage_wave,vmask=vmask,bin_r=True,check_masked=check_masked,verbose=verbose)
  if self.savecube:
   self.SaveCube()

 def getBase(self,base=None,basedir=None):
  if base is not None:
   self.base = base
  if basedir is not None:
   self.basedir = basedir
  if self.base is not None and isinstance(self.base,str) and self.basesdir is not None:
   self.basename = os.path.basename(self.base)
   self.base  = self.StarlightBase(self.base,self.basesdir)

 def GetSun(self,sunfile=None,wave=None,data=None):
  if sunfile is not None:
   self.sunfile = sunfile
  if self.sunfile is not None:
   fsun = pyfits.getdata(self.sunfile)
   self.sun = self.Spectrum1D(fsun['WAVELENGTH'],fsun['FLUX'])
  if wave is not None and data is not None:
   self.sun = self.Spectrum1D(wave=wave,data=data)

 def SetSpecType(self,synth,dered,rest,spectype):
  if self.calc is None:
   self.hprop = self.prop
   self.iprop = None
   self.scalc = {'synth': synth, 'dered': dered, 'rest': rest}
   self.specname = 'Syn' if synth else 'Obs'
   if rest:
    self.specname += 'R'
   if dered:
    self.specname += 'D'
  else:
   self.iprop = spectype
   self.hprop = [self.iprop + '_' + p for p in self.prop]
   self.specname = spectype

 def Close(self,filters,table=None):
  if filters is not None and self.KT is not None:
   if len(filters) > 0:
    filfile,filkey = map(list,zip(*filters))
    filfile = [os.path.basename(item) for item in filfile]
   self.add_keyword('SUNDICT',str(self.sundict).replace('\n',''))
   self.add_keyword('SPECTYPE',str(self.scalc).replace('\n',''))
   self.add_keyword('PROP',str(self.prop).replace('\n',''))
   if len(filters) > 0:
    self.add_keyword('FILTERS',str(list(filkey)).replace('\n',''))
    self.add_keyword('FILEFIL',str(list(filfile)).replace('\n',''))
   self.add_keyword('INTDICT',str(self.intdict).replace('\n',''))
   self.add_keyword('FILENAME',self.filename) # FITS file used
   self.add_keyword('BASEFITS',self.K.keywords['ARQ_BASE']) # Base used in FILENAME
   if self.base is not None and self.basename is not None:
    self.add_keyword('BASESYNT',self.basename) # Base used for SYNTH
   if table is not None and self.use_atpy:
    table.append(self.KT)

 def add_keyword(self,key,value):
  if self.use_atpy:
   self.KT.add_keyword(key,value)
  else:
   self.KT._meta[key] = value

 def add_column(self, name, column, check_masked=False, verbose=False):
  if check_masked:
   masked = False
   if isinstance(column, np.ma.MaskedArray):
    if np.all(column.mask):
     masked = True
    else:
     masked = np.all(~np.isfinite(column[~column.mask]))
   else:
    masked = np.all(~np.isfinite(column))
   if masked:
    if verbose:
     print ('>>> WARNING: Column "%s" is all masked (or NaN), NOT saved!' % name)
    return
  if self.use_atpy:
   self.KT.add_column(name, column)
  else:
   if column.ndim > 0 and column.size > 1:
    column = np.ma.expand_dims(column, axis=0)
   self.KT.add_column(MaskedColumn(name=name, data=column))

 def SetKcorr(self,zlum=None,kpow=None):
  if zlum is not None:
   self.zlum = zlum
  if kpow is not None:
   self.kpow = kpow
  if self.zlum is None:
   self.zlum = self.K.header['K_COR']
  self.kc = np.power(1 + self.zlum, self.kpow)

 def Kcorr(self):
  if self.kc is not None:
   self.data *= self.kc
   self.derr *= self.kc
   self.idata *= self.kc
   self.iderr *= self.kc

 def getApertureData(self, data, isError=False, area=True, mode=None, apertures=None, bin_r=False):
  if apertures is None:
   apertures = self.K.apertures if not bin_r else self.K.bin_apr
  if area:
   data__ayx = self.K.zoneToYX(data, extensive=False, surface_density=False) * self.K.zoneWeight
  else:
   data__ayx = self.K.zoneToYX(data, extensive=False, surface_density=False)
  if mode is None:
   data_ap = self.K.getAperturesPropLXY(data__ayx, apertures, area=False, isError=isError, mask_value=self.K.fill_value)
  else:
   data_ap = propApertures(data__ayx, apertures, mode=mode)
  return data_ap

 def SetData(self,synth,dered,rest):
  # Other way:
  #bt = atpy.Table(basefile, basedir, read_basedir=True, type='starlightv4_base')
  #ts = atpy.TableSet(output_, type = 'starlight')
  #factor_vel = ts.keywords['v_0']/300000. + 1
  #lamb = lamb*factor_vel
  #flux_base = bt['f_ssp'][i]
  #flux_0 = flux_0 + ts.population.popmu_ini[i]*ts.keywords['Mini_tot']*flux_base/np.sum(ts.population.popmu_ini)
  #flux_0 = flux_0*(3.826e33)/(4*np.pi*(ts.keywords['LumDistInMpc']* 3.0857e24)**2)
  #vfix = SpectraVelocityFixer(lambda,0,0)
  #flux = vfix.fix(flux_0, ts.keywords['v_d'])[0]
  if rest:
   self.redshift = None
  else:
   self.redshift = -self._redshift
  if synth:
   if self.base is not None:
    self.wave = self.base.l_ssp.copy()
    Lsun_to_flux = CST.Lsun / (4.0 * np.pi * (self.K.distance_Mpc * 1e6 * CST.PC)**2)
    self.idata = Lsun_to_flux*np.tensordot(self.K.integrated_Mini__tZ,self.base.f_ssp,((1,0),(0,1)))
    tflag = np.where(self.K.integrated_f_flag > 1, False, True)
    self.iderr = np.median(self.K.integrated_f_err[tflag])*np.ones(self.idata.shape)
    self.iflag = np.zeros(self.idata.shape)
    self.iav = self.K.integrated_keywords['A_V']
    self.data = Lsun_to_flux*np.tensordot(self.K.Mini__tZz,self.base.f_ssp,((1,0),(0,1))).T
    tflag = np.where(self.K.f_flag > 1, False, True)
    self.derr = np.median(self.K.f_err[tflag])*np.ones(self.data.shape)
    self.flag = np.zeros(self.data.shape)
    self.av = self.K.A_V
    self.rl = self.calc_redlaw(self.wave, self.RV, redlaw='CCM') # Reddening law
    if not dered:
     self.idata *= np.power(10., -0.4 * self.rl * self.iav)
     self.iderr *= np.power(10., -0.4 * self.rl * self.iav)
     self.data  *= np.power(10., -0.4 * self.rl[:,np.newaxis] * self.av)
     self.derr  *= np.power(10., -0.4 * self.rl[:,np.newaxis] * self.av)
   else:
    self.wave = self.K.l_obs.copy()
    self.idata = self.K.integrated_f_syn.copy()
    self.iderr = self.K.integrated_f_err.copy()
    self.iflag = np.where(self.K.integrated_f_flag > 1,True,False)
    self.iav = self.K.integrated_keywords['A_V']
    self.data = self.K.f_syn.copy()
    self.derr = self.K.f_err.copy()
    self.flag = np.where(self.K.f_flag > 1,True,False)
    self.av = self.K.A_V
    self.rl = self.calc_redlaw(self.wave, self.RV, redlaw='CCM') # Reddening law
    if dered:
     self.idata *= np.power(10., 0.4 * self.rl * self.iav)
     self.iderr *= np.power(10., 0.4 * self.rl * self.iav)
     self.data  *= np.power(10., 0.4 * self.rl[:,np.newaxis] * self.av)
     self.derr  *= np.power(10., 0.4 * self.rl[:,np.newaxis] * self.av)
  else:
   self.wave = self.K.l_obs.copy()
   self.idata = self.K.integrated_f_obs.copy()
   self.iderr = self.K.integrated_f_err.copy()
   self.iflag = np.where(self.K.integrated_f_flag > 1,True,False)
   self.iav = self.K.integrated_keywords['A_V']
   self.data = self.K.f_obs.copy()
   self.derr = self.K.f_err.copy()
   self.flag = np.where(self.K.f_flag > 1,True,False)
   self.av = self.K.A_V
   self.rl = self.calc_redlaw(self.wave, self.RV, redlaw='CCM') # Reddening law
   if dered:
    self.idata *= np.power(10., 0.4 * self.rl * self.iav)
    self.iderr *= np.power(10., 0.4 * self.rl * self.iav)
    self.data  *= np.power(10., 0.4 * self.rl[:,np.newaxis] * self.av)
    self.derr  *= np.power(10., 0.4 * self.rl[:,np.newaxis] * self.av)
  self.RedshiftWave()

 def SetWaveResolution(self,res_ini=None,res_end=None):
  if res_ini is not None:
   self.res_ini = res_ini
  if res_end is not None:
   self.res_end = res_end
  if self.res_ini is not None and self.res_end is not None:
   sigma_instrumental = self.res_ini/(2.*np.sqrt(2.*np.log(2.)))
   sigma_end = self.res_end/(2.*np.sqrt(2.*np.log(2.)))
   sigma_dif = np.sqrt(np.power(sigma_end,2.) - np.power(sigma_instrumental,2.))
   self.data  = gaussian_filter1d(self.data,sigma_dif,axis=0)
   self.idata = gaussian_filter1d(self.idata,sigma_dif,axis=0)

 def SetKinematics(self,v0=None,vd_ini=None,vd_end=None,iv0=None,ivd_ini=None,ivd_end=None,wave=None):
  # Example: SSP with res_ini = 3A (FWHM) and need to be with instrumental resolution res_end = 6A (FWHM) AND vd
  # First reduce resolution (final resolution of 6A WITHOUT kinematics):
  #  sigma_instrumental = res_ini/(2*np.sqrt(2*np.log(2)))
  #  sigma_end = res_end/(2*np.sqrt(2*np.log(2)))
  #  sigma_dif = np.sqrt(sigma_end**2 - sigma_instrumental**2)
  #  flux0 = sc.gaussian_filter1d(flux,sigma_dif)
  # Add kinematics. vd_ini = 0 since flux0 has no kinematics and vd_end = vd final velocity dispersion. The final 
  # spectra has 6A (FWHM) instrumental resolution and vd dispersion. The function convolves 
  # with v = sqrt(vd_end**2 - vd_ini**2):
  #  vfix = SpectraVelocityFixer(lamb,v0,vd_ini)
  #  flux = vfix.fix(flux0,vd_fin)
  # Example with CALIFA spectrum (flux0) with 6A (FWHM) y vd of 150 km/s and we want one with vd_end = 357 km/s:
  #  vfix = SpectraVelocityFixer(lamb,v0,150)
  #  flux = vfix.fix(flux0,357)
  #
  # Raw interpolation over lambda is not nice. Better a "histogram" interpolation (conservative):
  # f_ssp_resam = base.f_sspResam(K.l_obs)
  # q = calc_redlaw(K.l_obs, R_V=3.1, redlaw='CCM')
  # A_V = K.integrated_keywords['A_V']
  # if_syn = np.tensordot(f_ssp_resam.filled(), K.integrated_popx, ((1,0),(0,1))) * 10**(-0.4 * A_V * q)
  from pystarlight.util.velocity_fix import SpectraVelocityFixer
  #from pystarlight.util.velocity_fix import apply_kinematics
  if self.v0 is None:
   self.v0 = self.K.v_0
  if self.vd_ini is None:
   self.vd_ini = np.zeros(self.data.shape[1])
  if self.vd_end is None:
   self.vd_end = self.K.v_d
  if self.iv0 is None:
   self.iv0 = self.K.integrated_keywords['V_0']
  if self.ivd_ini is None:
   self.ivd_ini = 0.0
  if self.ivd_end is None:
   self.ivd_end = self.K.integrated_keywords['V_D']
  if v0 is not None:
   self.v0 = v0 
  if vd_ini is not None:
   self.vd_ini = vd_ini
  if vd_end is not None:
   self.vd_end = vd_end
  if iv0 is not None:
   self.iv0 = iv0 
  if ivd_ini is not None:
   self.ivd_ini = ivd_ini
  if ivd_end is not None:
   self.ivd_end = ivd_end
  if wave is not None:
   self.wave = wave
  if np.isscalar(self.vd_end):
   tmpvd = self.vd_end
   self.vd_end = np.zeros(self.data.shape[1])
   self.vd_end.fill(tmpvd)
  if np.isscalar(self.vd_ini):
   tmpvd_ini = self.vd_ini
   self.vd_ini = np.zeros(self.data.shape[1])
   self.vd_ini.fill(tmpvd_ini)
  if np.isscalar(self.v0):
   tmpv0 = self.v0
   self.v0 = np.zeros(self.data.shape[1])
   self.v0.fill(tmpv0)
  vfix = SpectraVelocityFixer(self.wave,self.v0,self.vd_ini)
  self.data = vfix.fix(self.data,self.vd_end)
  ivfix = SpectraVelocityFixer(self.wave,self.iv0,self.ivd_ini,nproc=self.nproc)
  self.idata = ivfix.fix(self.idata,self.ivd_end)
  #self.data = apply_kinematics(self.wave,self.data,self.v0,self.vd_end,nproc=self.nproc)
  
 def ReadPassBand(self,file):
  w,f = np.loadtxt(file,usecols=(0,1),unpack=True)
  return self.PassBand(wave=w,data=f,name_filter=os.path.basename(file))

 def getPassBandProp(self,passband,units=1.,percentage_wave=None,vmask=np.nan):
  self.bprop  = OrderedDict()
  self.ebprop = OrderedDict()
  flux, eflux = passband.getFluxPass(self.wave,self.flux,error=self.error,mask=self.mask,percentage_wave=percentage_wave,vmask=vmask)
  for key in self.prop:
   if 'Flux' == key:
    self.bprop[key]  = flux
    self.ebprop[key] = eflux
   if 'Mag' == key:
    vmag, evmag = passband.fluxToMag(flux,error=eflux,system='Vega',units=units)
    self.bprop[key]  = vmag
    self.ebprop[key] = evmag
   if 'AB' == key:
    mab, emab = passband.fluxToMag(flux,error=eflux,system='AB',units=units)
    self.bprop[key]  = mab
    self.ebprop[key] = emab
   if 'Lum' == key or 'L' == key:
    lum, elum = self.Flux2Lum(flux,self.K.distance_Mpc*CST.PC*1.0e6,eflux)
    if 'Lum' == key:
     self.bprop[key]  = lum
     self.ebprop[key] = elum
    if 'L' == key:
     lums, elums = lum/self.lsunband,elum/self.lsunband # Antes L_sun
     self.bprop[key]  = lums
     self.ebprop[key] = elums

 def LsunBand(self,passband,**kwargs):
  flux,eflux = passband.getFluxPass(self.sun._wave,self.sun._data,**kwargs)
  self.lsunband = flux*4.*np.pi*np.power(CST.AU,2)

 def Flux2Lum(self,flux,distance,error=None):
  lum = flux*4.*np.pi*np.power(distance,2.)
  if error is not None:
   error = error*4.*np.pi*np.power(distance,2.)
  return lum,error

 def GetMag(self,filters,percentage_wave=None,vmask=np.nan,check_masked=False,verbose=False):
  if filters is not None:
   for filter,fkey in filters:
    passb = self.ReadPassBand(filter)
    self.LsunBand(passb)
    lsun = self.lsunband if np.isfinite(self.lsunband) else 0.0
    self.sundict[fkey+'_Lsun'] = lsun #self.KT.add_keyword(fkey+'Lsun',lsun)
    self.getData(flux=self.idata,error=self.iderr,mask=self.iflag)
    self.getPassBandProp(passb,percentage_wave=percentage_wave,vmask=vmask)
    for key,hkey in zip(self.prop,self.hprop):
     vl,evl = self.bprop[key], self.ebprop[key]
     if check_masked and not np.isfinite(vl):
      continue
     vl = vl if np.isfinite(vl) else 0.0
     evl = evl if np.isfinite(evl) else 0.0
     skey = 'i_' + hkey + '_' + fkey
     self.intdict[skey] = vl       #self.KT.add_keyword('i_'+fkey+key,vl)
     self.intdict['e_'+skey] = evl #self.KT.add_keyword('i_e'+fkey+key,evl)
    self.getData()
    self.getPassBandProp(passb,percentage_wave=percentage_wave,vmask=vmask)
    if self.map2d or self.radial:
     if not 'Flux' in self.prop:
      flux, eflux = passb.getFluxPass(self.wave,self.flux,error=self.error,mask=self.flag,percentage_wave=percentage_wave,vmask=vmask)
     else:
      flux, eflux = self.bprop['Flux'], self.ebprop['Flux']
     flux      = self.Zonify(flux, 'Flux')
     eflux     = self.Zonify(eflux, 'Flux')
    if self.radial:
     flux__r   = self.K.radialProfileGen(flux)
     eflux__r  = self.K.radialProfileGen(eflux)
    for key,hkey in zip(self.prop,self.hprop):
     skey = hkey + '_' + fkey
     if self.map2d or self.radial:
      mprop  = self.Zonify(self.bprop[key], key)
      meprop = self.Zonify(self.ebprop[key], key)
      if key == 'Mag':
       mprop, meprop = passb.fluxToMag(flux,error=eflux,system='Vega',units=1.)
      elif key == 'AB':
        mprop, meprop = passb.fluxToMag(flux,error=eflux,system='AB',units=1.)
      if self.radial:
       if key == 'Mag':
        mprop__r, meprop__r = passb.fluxToMag(flux__r,error=eflux__r,system='Vega',units=1.)
       elif key == 'AB':
        mprop__r, meprop__r = passb.fluxToMag(flux__r,error=eflux__r,system='AB',units=1.)
       else:
        mprop__r  = self.K.radialProfileGen(mprop)
        meprop__r = self.K.radialProfileGen(meprop)
       if not self.masked and isinstance(mprop__r, np.ma.MaskedArray):
        mprop__r.data[mprop__r.mask]   = self.fill_value
        meprop__r.data[meprop__r.mask] = self.fill_value
        mprop__r  = mprop__r.data
        meprop__r = meprop__r.data
      if not self.masked and isinstance(mprop, np.ma.MaskedArray):
       mprop.data[mprop.mask]   = self.fill_value
       meprop.data[meprop.mask] = self.fill_value
       mprop  = mprop.data
       meprop = meprop.data
      if self.map2d:
       mkey = '%s__yx' % skey
       self.add_column(mkey, mprop, check_masked=check_masked, verbose=verbose)
       self.add_column('e_'+mkey, meprop, check_masked=check_masked, verbose=verbose)
      if self.radial:
       rkey = '%s__r' % skey
       self.add_column(rkey, mprop__r, check_masked=check_masked, verbose=verbose)
       self.add_column('e_'+rkey, meprop__r, check_masked=check_masked, verbose=verbose)
     else:
      self.add_column(skey, self.bprop[key], check_masked=check_masked, verbose=verbose)
      self.add_column('e_'+skey, self.ebprop[key], check_masked=check_masked, verbose=verbose)

 def GetMagAp(self,filters,mode=None,percentage_wave=None,vmask=np.nan,bin_r=False, check_masked=False, verbose=False):
  sap = 'Ap' if not bin_r else 'apr'
  if filters is not None:
   flux  = self.getApertureData(self.data,mode=mode,bin_r=bin_r)
   error = self.getApertureData(self.derr,isError=True,mode=mode,bin_r=bin_r)
   flag  = np.zeros(flux.shape,dtype=np.bool)
   self.getData(flux=flux,error=error,mask=flag)
   for filter,fkey in filters:
    passb = self.ReadPassBand(filter)
    self.LsunBand(passb)
    self.getPassBandProp(passb,percentage_wave=percentage_wave,vmask=vmask)
    for key,hkey in zip(self.prop,self.hprop):
     skey = '%s_%s__%s' % (hkey,fkey,sap)
     self.add_column(skey, self.bprop[key], check_masked=check_masked, verbose=verbose)
     self.add_column('e_'+skey,self.ebprop[key], check_masked=check_masked, verbose=verbose)

 def getData(self,i=None,flux=None,error=None,mask=None,AV=None,wave=None):
  if wave is not None:
   self.wave = wave
  if flux is not None:
   self.flux = flux
  if error is not None:
   self.error = error
  if mask is not None:
   self.mask = mask
  if i is not None:
   if self.data is not None and flux is None:
    self.flux = self.data[:,i]
   if self.error is not None and error is None:
    self.error = self.derr[:,i]
   if self.flag is not None and mask is None:
    self.mask = self.flag[:,i]
  else:
   if self.data is not None and flux is None:
    self.flux = self.data
   if self.error is not None and error is None:
    self.error = self.derr
   if self.flag is not None and mask is None:
    self.mask = self.flag
  if AV is not None:
   self.flux *= np.power(10., 0.4 * self.rl * AV)
   if self.error is not None:
    self.error *= np.power(10., 0.4 * self.rl * AV)

 def Resample(self,iprop):
  prop = iprop.copy()
  if self.lresam is not None:
   if self.slinterp:
    RSM = self.ReSamplingMatrixNonUniform(self.wave, self.lresam, extrap=self.extrap)
    prop = np.dot(RSM,iprop)
   else:
    prop = scipy.interpolate.interp1d(self.wave,iprop,axis=0)(self.lresam)
   return prop
  else:
   return prop[self.idw,...]

 def Zonify(self, prop, sprop=None, extensive=None, surface=None):
  if sprop is None:
   extensive = extensive if extensive is not None else self.extensive
   surface = surface if surface is not None else self.surface_density
  else:
   extensive, surface = False, False
  prop__yx =  self.K.zoneToYX(prop, extensive=extensive, surface_density=surface, fill_value=self.fill_value)
  if 'Flux' == sprop or 'L' == sprop or 'Lum' == sprop:
   prop__yx *= self.K.zoneWeight
  return prop__yx

 def getMagRadial(self, dprop_radial, func=None, check_masked=False, verbose=False):
  for name in dprop_radial:
   dname = dprop_radial[name]
   vfunc = func
   if isinstance(dname, dict):
    if 'func' in dname:
     vfunc = dname['func']
    if 'args' in dname:
     args = dname['args']
   else:
    args = dname
   if vfunc is None:
    if verbose:
     print('>>> WARNING: You need to provide a function ("func") if only arguments are given [%s]' % name)
    continue
   largs = []
   for arg in args:
    varg = getattr(self.K, arg, None)
    if arg in self.KT.colnames:
     varg = self.KT[arg][0]
    largs.append(varg)
    if varg is None:
     if verbose:
      print ('>>> WARNING: Attribute "%s" for Magnitude of fitsDataCube does NOT exists! [%s]' % (arg, name))
   if np.all([arg is None for arg in largs]):
    if verbose:
     print ('>>> WARNING: All arguments for "%s" are None!' % (name))
    continue
   try:
    value = vfunc(*largs)
   except:
    if verbose:
     print ('>>> WARNING: Error evaluating the function for "%s"' % name)
    continue
   prop__r = self.K.radialProfileGen(value)
   self.add_column(name, prop__r, check_masked=check_masked, verbose=verbose)

 def SetWave(self):
  self.lresam = self.olresam
  if self.cwave is None:
   self.wave1 = None
   self.wave2 = None
  else:
   self.wave1 = self.cwave[0]
   self.wave2 = self.cwave[1]
  if self.wave1 is None:
   self.wave1 = self.wave[0]
  if self.wave2 is None:
   self.wave2 = self.wave[-1]
  if self.dw is not None and self.lresam is None:
   self.lresam = np.arange(self.wave[0],self.wave[-1],self.dw)
  if self.lresam is not None:
   self.lresam = self.lresam[(self.lresam >= self.wave1) & (self.lresam <=self.wave2)]
   self.nwave = self.lresam
  else:
   self.idw = (self.wave >= self.wave1) & (self.wave <= self.wave2)
   self.nwave = self.wave[self.idw]

 def getMagCube(self, cube_file, filters, calc=None, prefix='cube_', suffix='__yx', rsuffix='__r', error_fmt='e_%s', 
	mask=True, radial=False, vmask=np.nan, percentage_wave=None, check_masked=False, verbose=False, **kwargs):
  if calc is None:
   calc = self.calc
  calc = checkList(calc)
  self.cube = LoadFits(cube_file, z=self._redshift, **kwargs)
  if isinstance(filters, list):
   filters = [[filt[1], filt[0]] for filt in filters]
  self.cube.setPassbands(filters=filters)
  if self.cube.dpass is None:
   return
  for filt in self.cube.dpass:
   pb = self.cube.dpass[filt]
   for mode in calc:
    if not 'Obs' in mode or 'D' in mode:
     continue
    wave = self.cube.wave_rest if 'R' in mode else self.cube.wave
    flux, eflux = pb.getFluxPass(wave, self.cube.data, error=self.cube.error, mask=self.cube.flag, vmask=vmask, percentage_wave=percentage_wave)
    if mask:
     flux  = np.ma.array(flux, mask=~self.K.qMask)
     eflux = np.ma.array(eflux, mask=~self.K.qMask)
    if radial:
     flux__r  = self.K.radialProfileGen(flux)
     eflux__r = self.K.radialProfileGen(eflux)
    for prop in self.prop:
     key  = '%s_%s_%s' % (mode, prop, filt)
     rkey = '%s_%s_%s' % (mode, prop, filt)
     if suffix is not None:
      key  = '%s%s' % (key, suffix)
      rkey = '%s%s' % (rkey, rsuffix)
     if prefix is not None:
      key  = '%s%s' % (prefix, key)
      rkey = '%s%s' % (prefix, rkey)
     ekey  = error_fmt % key
     erkey = error_fmt % rkey
     if 'Flux' == prop:
      vprop, evprop = flux, eflux
      vprop__r, evprop__r = flux__r, eflux__r
     if 'Mag' == prop:
      vprop, evprop = pb.fluxToMag(flux, error=eflux, system='Vega')
      if radial:
       vprop__r, evprop__r = pb.fluxToMag(flux__r, error=eflux__r, system='Vega')
     if 'AB' == prop:
      vprop, evprop = pb.fluxToMag(flux, error=eflux, system='AB')
      if radial:
       vprop__r, evprop__r = pb.fluxToMag(flux__r, error=eflux__r, system='AB')
     if ('Lum' == prop or 'L' == prop) :
      vprop, evprop = pb.fluxToLuminosity(d_pc=self.K.distance_Mpc * 1.0e6, flux=flux, error=eflux)
      if radial:
       vprop__r, evprop__r = pb.fluxToLuminosity(d_pc=self.K.distance_Mpc * 1.0e6, flux=flux__r, error=eflux__r)
      if 'L' == prop:
       sun_lum = pb.sunLuminosity()
       vprop /= sun_lum
       if radial:
        vprop__r /= sun_lum
       if evprop is not None:
        evprop /= sun_lum
        if radial:
         evprop__r /= sun_lum
     if mask:
      vprop  = np.ma.array(vprop, mask=~self.K.qMask)
      evprop = np.ma.array(evprop, mask=~self.K.qMask)
     self.add_column(key, vprop, check_masked=check_masked, verbose=verbose)
     self.add_column(ekey, evprop, check_masked=check_masked, verbose=verbose)
     if radial:
      self.add_column(rkey, vprop__r, check_masked=check_masked, verbose=verbose)
      self.add_column(erkey, evprop__r, check_masked=check_masked, verbose=verbose)

 def SaveCube(self,name=None):
  if name is not None:
   self.name_cube = name
  if self.name_cube is None:
   nprop = self.iprop if self.iprop is not None else 'RGB'
   self.name_cube = '.'.join([self.rootname,self.K.califaID,nprop,'fits']).replace('..','.')
  self.SetWave()
  cube = self.Zonify(self.Resample(self.data))
  if isinstance(cube,np.ma.MaskedArray):
   hdus = pyfits.HDUList([pyfits.PrimaryHDU(cube.data,header=self.K.header)])
  else:
   hdus = pyfits.HDUList([pyfits.PrimaryHDU(cube,header=self.K.header)])
  hdus.append(pyfits.ImageHDU(self.nwave, name='WAVE'))
  hdus.append(pyfits.ImageHDU(self.Resample(self.idata), name='INTSPEC'))
  hdus.writeto(self.name_cube,clobber=True)

 def RedshiftWave(self):
  if self.redshift is not None and self.wave is not None:
   self.wave = self.wave/(1.0 + self.redshift)

 def LumDistFac(self,norm=1000.): # PC in cm
  self.lumdist   = 4*np.pi*np.power(self.K.distance_Mpc*1.0e6*CST.PC,2.0)
  self.nlumdist  = self.lumdist/(4*np.pi*np.power(norm*CST.PC,2.0))

 def CalcQH(self,spec,hw=912.):
  self.frec = CST.cA/self.wave
  # Conversion to frequency and from flambda to fnu; division by photon energy
  # lambda*f_lambda = nu*f_nu --> f_nu = f_lambda*(lambda/nu) = f_lambda*lambda^2/c
  fnu = spec.T*self.wave*self.wave/(CST.cA*CST.h*self.frec)
  idf = self.frec >= (CST.cA/hw)
  qh = np.log10(-np.trapz(fnu[:,idf],self.frec[idf],axis=1))
  return qh

 def writeSpec(self,zone,name=None,fmt=None):
  if fmt is None:
   fmt = self.fmt
  if name is None:
   name = '%s_%s_%05i.txt' % (self.rootname,self.specname,zone)
  if zone == 0:
   spec = self.idata
  else:
   spec = self.data[:,zone-1]
  wspec(name,self.wave,spec,fmt=fmt)

 def CloudySED(self,name,index=None,idstart=1,integrated=True,cver='20060612'):
  if integrated:
   spec = np.hstack((self.idata.reshape((self.idata.size,1)),self.data))
  else:
   spec = self.data
  if index is None:
   index = np.arange(spec.shape[1])
  self.LumDistFac()
  self.qh = self.CalcQH(spec*self.lumdist)
  spec = spec[:,index]
  spec *= self.nlumdist
  spec = spec.T.flatten()
  #nmod = len(index) + 1 if integrated else len(index)
  nmod = len(index)
  # See hazy manual: Appendix B; INCLUDING USER-DEFINED ATMOSPHERE GRIDS IN CLOUDY (p. 265)
  lfile = [      cver,  # magic number
                    1,  # ndim
                    1,  # npar
             'spaxel',  # label par 1
                 nmod,  # nmod
       self.wave.size,  # nfreq
             'lambda',  # type of independent variable (nu or lambda)
                  1.0,  # conversion factor for independent variable
           'F_lambda',  # type of dependent variable (F_nu/H_nu or F_lambda/H_lambda)
                  1.0,  # conversion factor for dependent variable
          ]
  if integrated:
   zones = np.arange(idstart,idstart+self.K.N_zone+1)[index]
  else:
   zones = np.arange(idstart,idstart+self.K.N_zone)[index]
  self.index = index
  self.zones = zones
  #zones = np.insert(zones,0,0)
  lfile.append('  '.join(map(str,zones)))
  lfile += list(self.wave)
  lfile += list(spec)
  wlarray(name,lfile)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def readStarlightBase(basefile, basedir='', read_basedir=False, comments='#'):
 import atpy
 table = atpy.Table()
 table.keywords['arq_base'] = os.path.basename(basefile)

 dt = np.dtype([ ('sspfile', 'S60'), ('age_base', '>f8'), ('Z_base', '>f8'), ('sspname', 'S60'), ('Mstars', '>f8'), ('YA_V', '>i4'), ('aFe', '>f8') ])
 bdata = np.loadtxt( basefile, dtype=dt, skiprows=1, comments=comments )

 table.table_name = 'basefiles'
 table.add_column('sspfile', bdata['sspfile'])
 table.add_column('age_base', bdata['age_base'])
 table.add_column('Z_base', bdata['Z_base'])
 table.add_column('sspname', bdata['sspname'])
 table.add_column('Mstars', bdata['Mstars'])
 table.add_column('YA_V', bdata['YA_V'])
 table.add_column('aFe', bdata['aFe'])

 # If read_basedir is True, reads also the basedir spectra:
 if(read_basedir):
  f = []
  l = []
  for ssp_file in self.sspfile:
      t = atpy.Table(os.path.join(basedir, ssp_file) , type='ascii', name='SSP_spec', names=('wavelength', 'f_ssp')) #Here we remove any extension to the filename by split()
      f.append(t.f_ssp)
      l.append(t.wavelength)
  table.add_column('f_ssp', f)
  table.add_column('l_ssp', l)
 return table
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class fitsDataCube(fitsQ3DataCube):
 def __init__(self, namefits, smooth=True, preprocess_smooth=True, zsun=0.019, big_errors=1e3, fill_value=np.nan, 
	t_mass_growth=None, fill_mode='hollow',
	# Zone Mask
	residual=False, mask__z=None, r_in_mask__z=None, r_out_mask__z=None, invert_mask_distance__z=False, 
	w1=None, w2=4600., fnorm_max=5.0, percentage=60., filter_method='percentage', 
	# Spectra Mask
	mask=None, flag=True, error=True, weight=True,
        # Region Mask | include_mask_region: include mask_region IN qMask
        region=None, include_mask_region=False, invert_mask_region=False,
	# XY Mask |  include_maskXY: include maskXY IN qMask | mask_region: include mask_region IN maskXY
	maskXY=None, qMask=True, r_in_mask__yx=None, r_out_mask__yx=None, units_r_mask='hlr_fill',
	mask_region=True, include_maskXY=False, invert_mask_distance__yx=False, 
	# Geometry and Radial scale
	rad_scale=None, pa=None, ba=None, elliptical=True, fill_rad=True, set_hmr=False, 
	set_dered=False, set_image_filter=True, 
	# Radial Bin
	bin_r=None, r_in=0.0, r_out=3.0, dr=0.1, bin_r_dr=None, bin_r_in=None, 
	bin_r_out=None, radial_profile_basic=False, radial_mode='mean', 
	# Apertures
        apertures=False, bin_apertures=None, bin_apertures_dr=None, bin_apertures_in=None, bin_apertures_out=None, 
	central=False, integrated=False, integrate_apertures=False, inclusive=False, inclusive_r_ap=False, 
	ap_method='center', phot=False, phot_r=False, right=False, both=False, aperture_areaSD=True, 
	apr_areaSD=True, r_units='rad_scale', aperture_mode='sum', spectral_mode='sum', apr_mode='sum', 
	aperture_intensive_mode='mean', aperture_intensive_area=False, 
	# Label Apertures
	re='R$_{e}$', sign='$\leq$', fmt_re='%.2f', string_central='Nucleus', string_integrated='Total', 
	strip_zero=True, min_ap_in=0.0, replace_min_ap=None, label_in=None,
	# imageFilter
	passband=None, image_filter=None, image_filter_qnorm=None, image_filter_rest_frame=False, dict_image_filter={}, 
	# Age interpolation
	iageBase=None, iage_min=None, iage_max=None, dt=None, s=0, k=3, bspline=True,
	# pop age selection
	pop_age=None, pop_both=False, pop_right=True, pop_fill_None=False, pop_min=None, pop_max=None,
	# SFR
	t_sf=32.0e6, popx_min=None):

  self.baseId = ''
  self.zsun = zsun
  self.fill_value = fill_value
  self.big_errors = big_errors

  fitsQ3DataCube.__init__(self, namefits, smooth)
  self.EL = None
  self.apertures = None
  self.bin_r_ap = None
  self.gprop = None
  self.orig_qMask = self.qMask.copy()
  self.fill_mode = fill_mode
  self.radial_mode = radial_mode
  self.aperture_mode = aperture_mode
  self.aperture_intensive_mode = aperture_intensive_mode
  self.aperture_intensive_area = aperture_intensive_area
  self.spectral_mode = spectral_mode
  self.apr_mode = apr_mode
  self.aperture_areaSD = aperture_areaSD
  self.apr_areaSD = apr_areaSD
  self.radial_profile_basic = radial_profile_basic
  self.t_mass_growth = np.atleast_1d(t_mass_growth) if t_mass_growth is not None else None
  self.dirfits  = os.path.dirname(namefits)
  self.namefits = os.path.basename(namefits)
  self.metadict = {'version': VERSION, 'date': str(datetime.date.today()), 'radial_mode': radial_mode, 'aperture_mode': aperture_mode, 
	'spectral_mode': spectral_mode, 'apr_mode': apr_mode, 'fill_mode': fill_mode, 'radial_profile_basic': radial_profile_basic, 
	'aperture_areaSD': aperture_areaSD, 'apr_areaSD': apr_areaSD, 'aperture_intensive_mode': aperture_intensive_mode, 
	'aperture_intensive_area': aperture_intensive_area}

  self._setZoneWeight(smooth)
  self._setMask__z(mask__z=mask__z, residual=residual, w1=w1, w2=w2, fnorm_max=fnorm_max, percentage=percentage, method=filter_method, r_in=r_in_mask__z, r_out=r_out_mask__z, invert_mask_distance=invert_mask_distance__z)
  self._setMask__lz(mask=mask, flag=flag, error=error, weight=weight)
  self._setMaskRegion(region=region, invert_mask_region=invert_mask_region, include_mask_region=include_mask_region)
  self._setEllipseGeometry(ba=ba, pa=pa, elliptical=elliptical)
  self._setPassbandImageFilter(passband=passband, image_filter=image_filter, qnorm=image_filter_qnorm, rest_frame=image_filter_rest_frame, **dict_image_filter)
  self._setSignalDeRed()
  self._set_rad_scale(rad_scale=rad_scale, fill=fill_rad, set_hmr=set_hmr, set_dered=set_dered, set_image_filter=set_image_filter)
  self._set_bin_r(bin_r=bin_r, r_in=r_in, r_out=r_out, dr=dr, bin_r_dr=bin_r_dr, bin_r_in=bin_r_in, bin_r_out=bin_r_out)
  self._setZoneDistance()
  self._setImageDistance()
  self._set_bin_radial_ap(method=ap_method, phot=phot_r, right=right, both=both, r_units=r_units)
  self._setMaskXY(maskXY=maskXY, qMask=qMask, residual=residual, r_in=r_in_mask__yx, r_out=r_out_mask__yx, mask_region=mask_region, include_maskXY=include_maskXY, invert_mask_distance=invert_mask_distance__yx, units=units_r_mask)
  self._setMstars()
  self._set_age_interp(iageBase=iageBase, iage_min=iage_min, iage_max=iage_max, dt=dt, s=s, k=k, bspline=bspline)
  if apertures:
   self._set_bin_r_ap(inclusive=inclusive_r_ap, method=ap_method, phot=phot_r, right=right, both=both, r_units=r_units)
   self._set_bin_apertures(bin_apertures=bin_apertures, bin_apertures_dr=bin_apertures_dr, bin_apertures_in=bin_apertures_in, bin_apertures_out=bin_apertures_out, method=ap_method, phot=phot, 
	right=right, both=both, r_units=r_units, central=central, integrate_apertures=integrate_apertures, integrated=integrated, inclusive=inclusive)
   self._set_label_apertures(re=re, sign=sign, fmt_re=fmt_re, string_central=string_central, string_integrated=string_integrated, strip_zero=strip_zero, bin_apertures_in=bin_apertures_in, 
	bin_apertures_out=bin_apertures_out, min_ap_in=min_ap_in, replace_min_ap=replace_min_ap, label_in=label_in)
  self._set_pop_age(ages=pop_age, min_age=pop_min, max_age=pop_max, both=pop_both, right=pop_right, fill_None=pop_fill_None)
  self._set_sfr_mask_age(t_sf=t_sf, popx_min=popx_min)

 def _setZoneWeight(self, smooth=True):
  self.smoothZoneWeight = self.getDezonificationWeight(True)
  self.flatZoneWeight   = np.ma.array(self.getDezonificationWeight(False), mask=~self.qMask)
  if smooth:
   self.zoneWeight = self.smoothZoneWeight
  else:
   self.zoneWeight = self.flatZoneWeight

 def _setEllipseGeometry(self, image=None, pa=None, ba=None, elliptical=True):
  mdict = {'setEllipseGeometry': dict(elliptical=elliptical, ba=ba, pa=pa)}
  self.updateMetadict(mdict)
  if image is None:
   image = self.qSignal
  qSignal_fill, mask = self.fillImage(image, mode=self.fill_mode)
  ipa, iba = self.getEllipseParams(qSignal_fill, mask)
  if pa is None:
   pa = ipa if elliptical else 0.0
  if ba is None:
   ba = iba if elliptical else 1.0
  self.setGeometry(pa, ba)

 def updateMetadict(self, mdict):
  if mdict is not None:
   mdict = dict2List(mdict, to_string=False)
   self.metadict = updateNestedDict(self.metadict, mdict)

 def updateFunctionArgs(self, function, mkey=None, mdict=True, **kwargs):
  if mkey is None:
   mkey = function.__name__
   if mkey.startswith('_'):
    mkey = mkey[1:]
  afunc = inspect.getargspec(function)
  args = afunc.args
  if 'self' in args:
   args.remove('self')
  dfunc = {key: value for (key, value) in zip(args, afunc.defaults)}
  if mdict:
   dfunc.update(self.metadict.get(mkey, {}))
  if len(kwargs) > 0:
   for key in dfunc:
    if key in kwargs:
     dfunc[key] = kwargs[key]
  return dfunc

 def _get_bin_r(self, bin_r=None, bin_r_dr=None, bin_r_in=None, bin_r_out=None, r_in=0.0, r_out=3.0, dr=0.1, default=True, mask_negative=True):
  if isinstance(bin_r, dict):
   bin_r_in  = np.array([bin_r[key][0] for key in bin_r])
   bin_r_out = np.array([bin_r[key][1] for key in bin_r])
   bin_r     = np.array([key for key in bin_r])
  else:
   if bin_r is not None and (bin_r_in is None or bin_r_out is None):
    if bin_r_dr is not None:
     bin_r = bin_r
     bin_r_in  = np.array([rbin - bin_r_dr for rbin in bin_r])
     bin_r_out = np.array([rbin + bin_r_dr for rbin in bin_r])
    else:
     bin_r_in = np.array(bin_r[:-1])
     bin_r_out = np.array(bin_r[1:])
     bin_r = (bin_r_in + bin_r_out) / 2.0
  if bin_r is None and bin_r_in is not None and bin_r_out is not None:
   bin_r = (bin_r_in + bin_r_out) / 2.0
  if bin_r is None and default:
   bin_r = np.arange(r_in, r_out + dr / 2.0, dr)
   bin_r_in = bin_r[:-1]
   bin_r_out = bin_r[1:]
   bin_r = (bin_r_in + bin_r_out) / 2.0
  if bin_r is not None:
   bin_r = np.atleast_1d(bin_r)
   if mask_negative:
    bin_r[bin_r < 0.0] = 0.0
  if bin_r_in is not None:
   bin_r_in = np.atleast_1d(bin_r_in)
   if mask_negative:
    bin_r_in[bin_r_in < 0.0] = 0.0
  if bin_r_out is not None:
   bin_r_out = np.atleast_1d(bin_r_out)
   if mask_negative:
    bin_r_out[bin_r_out < 0.0] = 0.0
  return bin_r, bin_r_in, bin_r_out

 def _set_bin_r(self, bin_r=None, bin_r_dr=None, bin_r_in=None, bin_r_out=None, r_in=0.0, r_out=3.0, dr=0.1):
  mdict = {'set_bin_r': dict(bin_r=bin_r, bin_r_dr=bin_r_dr, bin_r_in=bin_r_in, bin_r_out=bin_r_out, r_in=r_in, r_out=r_out, dr=dr)}
  self.updateMetadict(mdict)
  rbin = self._get_bin_r(bin_r=bin_r, bin_r_dr=bin_r_dr, bin_r_in=bin_r_in, bin_r_out=bin_r_out, r_in=r_in, r_out=r_out, dr=dr)
  self.bin_r = rbin[0]
  self.bin_r_in = rbin[1]
  self.bin_r_out = rbin[2]
  self.bin_r_profile = np.unique(np.concatenate((self.bin_r_in, self.bin_r_out)))

 def _set_bin_radial_ap(self, method='exact', phot=False, right=False, both=False, r_units='rad_scale'):
  mdict = {'set_bin_radial_ap': dict(method=method, phot=phot, right=right, both=both, r_units=r_units)}
  self.updateMetadict(mdict)
  self.bin_radial_ap = self.getApertureMask(self.bin_r_in, self.bin_r_out, method=method, phot=phot, right=right, both=both, 
	r_units=r_units, central=False, integrate_apertures=False, integrated=False)
  self.bin_radial_ap_Area_pix = np.array([ap.sum() for ap in self.bin_radial_ap])
  self.bin_radial_ap_Area_pc2 = self.bin_radial_ap_Area_pix * self.parsecPerPixel**2

 def _set_bin_r_ap(self, inclusive=False, method='exact', phot=False, right=False, both=False, r_units='rad_scale'):
  mdict = {'set_bin_r_ap': dict(inclusive=inclusive, method=method, phot=phot, right=right, both=both, r_units=r_units)}
  self.updateMetadict(mdict)
  self.bin_r_ap = self.bin_r.copy()
  self.bin_r_ap_in = self.bin_r_in.copy()
  self.bin_r_ap_out = self.bin_r_out.copy()
  if inclusive:
   self.bin_r_ap_in = np.tile(min(self.bin_r_ap_in), self.bin_r_ap_in.size)
  self.bin_apr = self.getApertureMask(self.bin_r_ap_in, self.bin_r_ap_out, method=method, phot=phot, right=right, both=both, 
	r_units=r_units, central=False, integrate_apertures=False, integrated=False)
  self.bin_r_ap_Area_pix = np.array([ap.sum() for ap in self.bin_apr])
  self.bin_r_ap_Area_pc2 = self.bin_r_ap_Area_pix * self.parsecPerPixel**2

 def _set_bin_apertures(self, bin_apertures=None, bin_apertures_dr=None, bin_apertures_in=None, bin_apertures_out=None, method='exact', phot=False, 
	right=False, both=False, r_units='rad_scale', central=False, integrate_apertures=False, integrated=False, inclusive=False):
  if bin_apertures is None and (bin_apertures_in is None or bin_apertures_out is None):
   self.bin_apertures = self.bin_r.copy()
   self.bin_apertures_in = self.bin_r_in.copy()
   self.bin_apertures_out = self.bin_r_out.copy()
  else:
   rbin = self._get_bin_r(bin_r=bin_apertures, bin_r_dr=bin_apertures_dr, bin_r_in=bin_apertures_in, bin_r_out=bin_apertures_out, default=False)
   self.bin_apertures = rbin[0]
   self.bin_apertures_in = rbin[1]
   self.bin_apertures_out = rbin[2]
  if inclusive:
   self.bin_apertures_in = np.tile(min(self.bin_apertures_in), self.bin_apertures.size)
   integrate_apertures = False
  self.apertures = self.getApertureMask(self.bin_apertures_in, self.bin_apertures_out, method=method, phot=phot, right=right, both=both, central=central, r_units=r_units, integrate_apertures=integrate_apertures, integrated=integrated)
  self.apertures_Area_pix = np.array([ap.sum() for ap in self.apertures])
  self.apertures_Area_pc2 = self.apertures_Area_pix * self.parsecPerPixel**2
  mdict = {'set_bin_apertures': dict(method=method, phot=phot, right=right, both=both, r_units=r_units, central=central, integrate_apertures=integrate_apertures, integrated=integrated, inclusive=inclusive)}
  self.updateMetadict(mdict)

 def _set_label_apertures(self, re='R$_{e}$', sign='$\leq$', fmt_re='%.2f', string_central='Nucleus', string_integrated='Total', strip_zero=True, bin_apertures_in=None, bin_apertures_out=None, min_ap_in=0.0, replace_min_ap=None, label_in=None):
  mdict = {'set_label_apertures': dict(re=re, sign=sign, fmt_re=fmt_re, string_central=string_central, string_integrated=string_integrated, strip_zero=strip_zero)}
  self.updateMetadict(mdict)
  self.label_apertures = []
  if bin_apertures_in is None:
   bin_apertures_in = self.bin_apertures_in
  if bin_apertures_out is None:
   bin_apertures_out = self.bin_apertures_out
  if replace_min_ap is not None:
   bin_aperture_out[bin_aperture_out == min(bin_aperture_out)] = replace_min_ap
  if label_in is None:
   label_in = False if np.all(bin_apertures_in <= min_ap_in) else True
  for r_in, r_out in zip(bin_apertures_in, bin_apertures_out):
   if label_in:
    label_ap = r'%s %s %s %s %s' % (self.format_float(r_in, fmt=fmt_re, strip_zero=strip_zero), sign, re, sign, self.format_float(r_out, fmt=fmt_re, strip_zero=strip_zero))
   else:
    label_ap = r'%s %s %s' % (sign, self.format_float(r_out, strip_zero=strip_zero, fmt=fmt_re), re)
   self.label_apertures.append(label_ap)
  if self.metadict.get('set_bin_apertures').get('central'):
   self.label_apertures.insert(0, string_central)
  if self.metadict.get('set_bin_apertures').get('integrate_apertures') and not self.metadict.get('set_bin_apertures').get('inclusive'):
   self.label_apertures.append(r'%s %s %s' % (sign, self.format_float(max(bin_apertures_out), strip_zero=strip_zero, fmt=fmt_re), re))
  if self.metadict.get('set_bin_apertures').get('integrated'):
   self.label_apertures.append(string_integrated)

 def format_float(self, value, fmt='%.2f', strip_zero=True):
  fmt_value = fmt % value
  if strip_zero:
   fmt_value = fmt_value.rstrip('0').rstrip('.')
  return fmt_value

 def reset_apertures(self, **kwargs):
  dbinr = getDictFunctionArgs(self._set_bin_r, remove_self=True)
  if np.any([key in dbinr for key in kwargs]):
   self._set_bin_r(**self.updateFunctionArgs(self._set_bin_r, **kwargs))
  self._set_bin_radial_ap(**self.updateFunctionArgs(self._set_bin_radial_ap, **kwargs))
  self._set_bin_r_ap(**self.updateFunctionArgs(self._set_bin_r_ap, **kwargs))
  self._set_bin_apertures(**self.updateFunctionArgs(self._set_bin_apertures, **kwargs))
  self._set_label_apertures(**self.updateFunctionArgs(self._set_label_apertures, **kwargs))

 def reset_bin_apertures(self, **kwargs):
  self._set_bin_apertures(**self.updateFunctionArgs(self._set_bin_apertures, **kwargs))
  self._set_label_apertures(**self.updateFunctionArgs(self._set_label_apertures, **kwargs))

 def maskDistanceApertures(self, r_in, r_out, mask=None, apply_qmask=False, r_units='rad_scale', single=False, area=False, both=False, right=False):
  if mask is None:
   if apply_qmask:
    mask = self.qMask
   else:
    mask = np.ones(self.qMask.shape, dtype=np.bool)

  # Image distance is in Pixels
  r__yx = self.imagePixelDistance

  # Scale in PIXEL / UNITS
  scale = self.getDistanceScaleUnits(r_units) 

  # Convert input radius to pixel distance
  r_in  = np.atleast_1d(r_in).astype(np.float32)  * scale
  r_out = np.atleast_1d(r_out).astype(np.float32) * scale

  rmask = []
  for br_in, br_out in zip(r_in, r_out):
   if both:
    bmask = (r__yx >= br_in) & (r__yx <= br_out) & mask
   else:
    if right:
     bmask = (r__yx > br_in) & (r__yx <= br_out) & mask
    else:
     bmask = (r__yx >= br_in) & (r__yx < br_out) & mask
   rmask.append(bmask)

  if area:
   area = np.array([ap.sum() for ap in rmask])

  if single and len(rmask) == 1:
   rmask = rmask[0]
  else:
   rmask = np.array(rmask)

  if area:
   return rmask, area
  else:
   return rmask

 def maskDistanceAperturesPhot(self, r_in, r_out, method='exact', r_units='rad_scale', single=False, area=False):
  from pycasso.photutils import EllipticalAnnulus

  N_y, N_x = self.qSignal.shape
 
  x_min = -self.x0
  x_max = N_x - self.x0
  y_min = -self.y0
  y_max = N_y - self.y0

  # Scale in PIXEL / UNITS
  scale = self.getDistanceScaleUnits(r_units)

  # r_in and r_out must be in pixels for calculation
  r_in  = np.atleast_1d(r_in).astype(np.float32)  * scale
  r_out = np.atleast_1d(r_out).astype(np.float32) * scale

  rmask = []
  mask_area = []
  for a_in, a_out in zip(r_in, r_out):
   an = EllipticalAnnulus(a_in, a_out, a_out * self.ba, self.pa)
   rmask.append(an.encloses(x_min, x_max, y_min, y_max, N_x, N_y, method=method).astype(np.bool))
   mask_area.append(an.area())

  mask_area = np.array(mask_area)

  if single and len(rmask) == 1:
   rmask = rmask[0]
  else:
   rmask = np.array(rmask)

  if area:
   return rmask, mask_area
  else:
   return rmask

 def getMaskDistanceApertures(self, r_in, r_out, method='exact', phot=False, right=False, 
	both=False, single=False, r_units='rad_scale', mask=None, central=False):
  if phot:
   return self.maskDistanceAperturesPhot(r_in, r_out, method=method, single=single, r_units=r_units)
  else:
   return self.maskDistanceApertures(r_in, r_out, single=single, r_units=r_units, mask=mask, apply_qmask=False)

 def getApertureMask(self, r_in, r_out, method='exact', phot=True, right=False, both=False, mask=None,
	r_units='rad_scale', central=False, integrate_apertures=False, integrated=False):

  r_in  = np.atleast_1d(r_in).astype(np.float32)
  r_out = np.atleast_1d(r_out).astype(np.float32)
 
  amask = self.getMaskDistanceApertures(r_in, r_out, method=method, phot=phot, right=right, both=both, 
	r_units=r_units, mask=mask, single=False)
 
  if central:
   mcentral = np.zeros_like(self.qSignal, dtype=np.bool)
   # Using a non-integer number instead of an integer will result in an error in the future
   mcentral[int(self.y0), int(self.x0)] = True
   amask = np.insert(amask, 0, mcentral, axis=0)
 
  if integrate_apertures:
   a_in  = min(r_in)
   a_out = max(r_out)
   imask = self.getMaskDistanceApertures(a_in, a_out, method=method, phot=phot, right=right, both=both, 
	r_units=r_units, mask=mask, single=False)
   amask = np.append(amask, imask, axis=0)

  if integrated:
   imask = self.qMask > 0.0
   amask = np.append(amask, imask[np.newaxis, ...], axis=0)
 
  return amask

 def _set_age_interp(self, iageBase=None, iage_min=None, iage_max=None, dt=None, der=1, s=0, k=3, bspline=True):
  self.dt = dt
  self.s = s
  self.k = k
  self.bspline = bspline
  self.iageBase = self.ageBase.copy() if iageBase is None else iageBase
  if self.dt is not None and iageBase is None:
   iage_min = self.ageBase.min() if iage_min is None else iage_min
   iage_max = self.ageBase.max() if iage_max is None else iage_max
   self.iageBase = np.arange(iage_min, iage_max + self.dt/2., self.dt)
  
 def getMaskRegion(self, region=None, invert_mask=False):
  mask_region = None
  if isinstance(region, str):
   import pyregion
   if os.path.exists(region):
    region = pyregion.open(region)
   else:
    try:
     region = pyregion.parse(region)
    except:
     print ('WARNING: DS9 file or region NOT valid ["%s"]' % region)
     region = None
  if region is not None:
   try:
    mask_region = region.get_mask(shape=self.qSignal.shape, header=self.header)
   except:
    hwcs = WCS(self.header)
    mask_region = region.get_mask(shape=self.qSignal.shape, header=hwcs.celestial.to_header())
   if not invert_mask:
    mask_region = ~mask_region
  return mask_region

 def _setMaskRegion(self, region, invert_mask_region=False, include_mask_region=False):
  # Run it before Ellipse and HLR/HMR calculations mask regions IF include_mask_region = True
  # HLR/HMR calculations take as mask qMask (NOT maskXY)
  mdict = {'setMaskRegion': {'invert_mask_region': invert_mask_region, 'include_mask_region': include_mask_region}}
  self.updateMetadict(mdict)
  self.mask_region = self.getMaskRegion(region, invert_mask=invert_mask_region)
  if self.mask_region is not None and include_mask_region:
   self.qMask = np.logical_and(self.qMask, self.mask_region)

 def getDistanceScaleUnits(self, units):
  # Returns a scale in PIXEL / UNITS --> imagePixelDistance / scale is in UNITS | distance_units * scale is in PIXEL
  units = units.lower().strip()
  if units == 'hlr':
   scale = self.HLR_pix
  elif units == 'hlr_fill':
   scale = self.HLR_pix_fill
  elif units == 'deredhlr':
   scale = self.DeredHLR_pix
  elif units == 'deredhlr_fill':
   scale = self.DeredHLR_pix_fill
  elif units == 'hmr':
   scale = self.HMR_pix
  elif units == 'hmr_fill':
   scale = self.HMR_pix_fill
  elif units == 'rad_scale':
   scale = self.rad_scale
  elif units == 'pc':
   scale = 1. / self.parsecPerPixel
  elif units == 'kpc':
   scale = 1.0e3 / self.parsecPerPixel
  elif units == 'pixel':
   scale = 1.0
  else:
   print ('WARNING: Scale "%s" NOT found! Setting scale to 1.0' % (units))
   scale = 1.0
  return scale

 def imagePixelDistanceScaleUnits(self, units):
  units = units.lower().strip()
  scale = self.getDistanceScaleUnits(units)
  mdict = {'imagePixelDistanceScaleUnits': {'units': units, 'scale': scale}}
  self.updateMetadict(mdict)
  return scale

 def _setMaskXY(self, maskXY=None, qMask=True, residual=True, r_in=None, r_out=None, mask_region=True, 
	include_maskXY=False, invert_mask_distance=False, units='hlr'):
  # maskXY is a mask for radial and aperture calculations 
  # It is not taken into account in HLR/HMR estimation unless set_rad_scale is run again after setMaskXY
  mdict = {'setMaskXY': {'qMask': qMask, 'residual': residual, 'r_in': r_in, 'r_out': r_out, 'mask_region': mask_region, 'include_maskXY': include_maskXY, 'invert_mask_distance': invert_mask_distance}}
  self.updateMetadict(mdict)
  self.mask_distance = None
  if maskXY is None:
   maskXY = self.qMask if qMask else self.fillImage(self.qMask, mode='hollow')[0].astype(np.bool)
  else:
   if qMask:
    maskXY = np.logical_and(self.qMask, maskXY)
  if residual and self.filter_residual__z is not None:
   maskXY = np.logical_and(maskXY, self.filter_residual__yx)
  scale = self.imagePixelDistanceScaleUnits(units)
  if r_in is not None or r_out is not None:
   imageDistanceInUnits = self.imagePixelDistance / scale
   mask_distance = np.ones_like(maskXY, dtype=np.bool)
   if r_in is not None:
    mask_distance = np.logical_and(mask_distance, imageDistanceInUnits >= r_in)
   if r_out is not None:
    mask_distance = np.logical_and(mask_distance, imageDistanceInUnits <= r_out)
   if not invert_mask_distance:
    mask_distance = ~mask_distance
   self.mask_distance = mask_distance
   maskXY = np.logical_and(maskXY, mask_distance)
  if mask_region and self.mask_region is not None:
   maskXY = np.logical_and(self.mask_region, maskXY)
  if include_maskXY and maskXY is not None:
   self.qMask = np.logical_and(self.qMask, maskXY)
  self.maskXY = maskXY

 def set_qMask(self, maskXY=False, mask_region=False):
  mdict = {'setMaskXY': {'include_maskXY': maskXY}, 'setMaskRegion': {'include_mask_region': mask_region}}
  self.updateMetadict(mdict)
  self.qMask = self.orig_qMask.copy()
  if maskXY and self.maskXY is not None:
   self.qMask = np.logical_and(self.qMask, self.maskXY)
  if mask_region and self.mask_region is not None:
   self.qMask = np.logical_and(self.qMask, self.mask_region)

 def _set_rad_scale(self, rad_scale=None, fill=True, set_hmr=False, set_dered=False, set_image_filter=True):
  set_image_filter = True if (self.imageFilter is not None and set_image_filter) else False
  mdict = {'set_rad_scale': {'rad_scale': rad_scale, 'fill': fill, 'set_hmr': set_hmr, 'set_dered': set_dered, 'set_image_filter': set_image_filter}}
  self.updateMetadict(mdict)
  if self.passband_filter is not None:
   self.metadict['set_rad_scale']['passband_filter'] = self.passband_filter
  if rad_scale is None:
   if set_hmr:
    if fill:
     self.rad_scale = self.HMR_pix_fill
    else:
     self.rad_scale = self.HMR_pix
   else:
    if self.imageFilter is not None and set_image_filter:
     if fill:
      self.rad_scale = self.imageFilterDeRedHLR_pix_fill if set_dered else self.imageFilterHLR_pix_fill
     else:
      self.rad_scale = self.imageFilterDeRedHLR_pix if set_dered else self.imageFilterHLR_pix
    else:
     if fill:
      self.rad_scale = self.DeRedHLR_pix_fill if set_dered else self.HLR_pix_fill
     else:
      self.rad_scale = self.DeRedHLR_pix if set_dered else self.HLR_pix_nofill
  else:
   self.rad_scale = rad_scale

 def _setZoneDistance(self, x0=None, y0=None, ba=None, pa=None):
  if ba is None:
   ba = self.ba
  if pa is None:
   pa = self.pa
  if x0 is None:
   x0 = self.x0
  if y0 is None:
   y0 = self.y0
  self.zonePixelDistance = getDistance(self.zonePos['x'], self.zonePos['y'], x0, y0, pa, ba)
  self.zoneDistance = self.zonePixelDistance / self.rad_scale

 def _setImageDistance(self, x0=None, y0=None, ba=None, pa=None):
  if ba is None:
   ba = self.ba
  if pa is None:
   pa = self.pa
  if x0 is None:
   x0 = self.x0
  if y0 is None:
   y0 = self.y0
  self.imagePixelDistance = getImageDistance(self.qSignal.shape, x0, y0, pa, ba)
  self.imageDistance = self.imagePixelDistance / self.rad_scale
  
 def _setMask__z(self, mask__z=None, residual=True, w1=None, w2=4600., fnorm_max=5.0, percentage=60., method='percentage', r_in=None, r_out=None, invert_mask_distance=False):
  mdict = {'setMask__z': dict(residual=residual, method=method, percentage=percentage, invert_mask_distance=invert_mask_distance, w1=w1, w2=w2)}
  self.updateMetadict(mdict)
  self.mask__z = np.ones(self.N_zone, dtype=np.bool)
  if mask__z is not None:
   self.mask__z *= mask__z
  if residual:
   self.filter_residual__z = self.filterResidual(w1=w1, w2=w2, fnorm_max=fnorm_max, percentage=percentage, method=method)
   self.mask__z *= self.filter_residual__z
  else:
   self.filter_residual__z = None
  if r_in is not None or r_out is not None:
   mask_distance = np.ones_like(self.mask__z, dtype=np.bool)
   if r_in is not None:
    mask_distance = np.logical_and(mask_distance, self.zoneDistance >= r_in)
   if r_out is not None:
    mask_distance = np.logical_and(mask_distance, self.zoneDistance <= r_out)
   if not invert_mask_distance:
    mask_distance = ~mask_distance
   self.mask__z = np.logical_and(self.mask__z, mask_distance)
  
 def _setMask__lz(self, mask=None, flag=True, error=True, weight=True):
  mdict = {'setMask__lz': {'flag': flag, 'error': error, 'weight': weight}}
  self.updateMetadict(mdict)
  self.mask__lz = np.ones_like(self.f_obs, dtype=np.bool)
  self.mask__lz *= self.mask__z
  if mask is not None:
   self.mask__lz *= mask 
  if flag:
   self.mask__lz *= self.maskFlag__lz
  if error:
   self.mask__lz *= self.maskErr__lz
  if weight:
   self.mask__lz *= self.maskWei__lz

 def _setMstars(self, mask_value=0.0):
  # Mstars
  if self.Mstars.ndim == 3:
   self.Mstars__tZ  = self.fillMstars(self.Mstars[...,0])
  else:
   self.Mstars__tZ  = self.fillMstars(self.Mstars)
  self.Mstars__ttZ = getMstarsForTime(self.Mstars__tZ, self.ageBase, self.metBase)
  self.Mstars__t   = self.Mstars__tZ.mean(axis=1)
  self.Mstars__tt  = self.Mstars__ttZ.mean(axis=2) 
  self.cumsum__tt = np.triu(np.ones((self.ageBase.size, self.ageBase.size)))  # For cumsum
  self.mass_growth__tz = np.tensordot(self.Mstars__ttZ, self.Mini__tZz, ((1,2),(0,1)))

 def _setPassband(self, passband=None):
  self.passband = passband
  self.passband_filter = None
  if isinstance(passband, str):
   from pyfu.passband import PassBand
   self.passband = PassBand()
   self.passband.loadTxtFile(passband)
  if self.passband is not None:
   self.passband_filter = self.passband._name_filter

 def _setImageFilter(self, image_filter=None, rest_frame=False, **kwargs):
  self.imageFilter = None if isinstance(image_filter, str) else image_filter
  self.image_filter_file = os.path.basename(image_filter) if isinstance(image_filter, str) else None
  if isinstance(image_filter, str) and self.passband is not None:
   fits = LoadFits(image_filter)
   wave = fits.wave_rest if rest_frame and fits.wave_rest is not None else fits.wave
   photo = self.passband.getFluxPass(wave, fits.data, error=fits.error, mask=fits.flag, **kwargs)
   self.imageFilter = photo[0]

 def _setImageFilterReddening(self, qnorm=None, R_V=3.1):
  self.image_filter_qnorm = qnorm
  self.imageFilterDeRed = None
  if self.passband is not None and qnorm is None:
   from pystarlight.util.redenninglaws import Cardelli_RedLaw
   self.image_filter_qnorm = Cardelli_RedLaw(np.atleast_1d(self.passband.effectiveWave), R_V=R_V)

 def _setImageFilterDeRed(self):
  self.imageFilterDeRed = None
  if self.imageFilter is not None and self.image_filter_qnorm is not None:
   av_corr = self.zoneToYX(np.power(10., 0.4 * self.image_filter_qnorm * self.A_V), extensive=False, surface_density=False)
   self.imageFilterDeRed = self.imageFilter * av_corr

 def _setPassbandImageFilter(self, passband=None, image_filter=None, qnorm=None, R_V=3.1, rest_frame=False, **kwargs):
  self._setPassband(passband=passband)
  self._setImageFilter(image_filter=image_filter, rest_frame=rest_frame, **kwargs)
  self._setImageFilterReddening(qnorm=qnorm, R_V=R_V)
  self._setImageFilterDeRed()

 def _setSignalDeRed(self, qnorm=None, qSignal_lambda=None, R_V=3.1):
  self.qSignal_qnorm = qnorm
  self.qSignal_lambda = qSignal_lambda
  self.qSignalDeRed = None
  if self.qSignal_qnorm is None:
   window = self.header.get('WINDOWSN', None)
   try:
    window = ast.literal_eval(window)
   except:
    pass
   if window is None:
    self.qSignal_qnorm = self.q_norm
   elif isinstance(window, (list, tuple, np.ndarray)) and self.qSignal_lambda is None:
    self.qSignal_lambda = (max(window) + min(window)) / 2.
   if self.qSignal_qnorm is None and self.qSignal_lambda is not None:
    from pystarlight.util.redenninglaws import Cardelli_RedLaw
    self.qSignal_qnorm = Cardelli_RedLaw(np.atleast_1d(self.qSignal_lambda), R_V=R_V)
  if self.qSignal_qnorm is not None:
   self.qSignalDeRed = self.zoneToYX(np.power(10., 0.4 * self.qSignal_qnorm * self.A_V), extensive=False, surface_density=False) * self.qSignal

 def get_age_selection(self, ages=None, min_age=None, max_age=None, fill_None=False):
  if ages is None:
   return
  ages_min = []
  ages_max = []
  ages = checkList(ages)
  if np.all([not isinstance(age, (list, tuple, np.ndarray)) for age in ages]):
   nages = [(None, ages[0])]
   for i in range(len(ages) - 1):
    nages.append((ages[i], ages[i+1]))
   nages.append((ages[-1], None))
  else:
   nages = ages
  lage_min = []
  lage_max = []
  for age in nages:
   if not isinstance(age, (list, tuple)):
    age = (None, age)
   age_min, age_max = age
   if age_min is None:
    age_min = min_age
   if age_max is None:
    age_max = max_age
   if age_min is None and fill_None:
    age_min = self.ageBase.min()
   if age_max is None and fill_None:
    age_max = self.ageBase.max()
   lage_min.append(age_min)
   lage_max.append(age_max)
  return lage_min, lage_max

 def _set_pop_age(self, ages=None, min_age=None, max_age=None, both=False, right=True, fill_None=False):
  mdict = {'set_pop_age': dict(ages=ages, min_age=min_age, max_age=max_age, both=both, right=right, fill_None=fill_None)}
  self.updateMetadict(mdict)
  self.pop_age_mask = None
  self.pop_age_min = None
  self.pop_age_max = None
  if ages is None:
   return
  self.pop_age_min, self.pop_age_max = self.get_age_selection(ages=ages, min_age=min_age, max_age=max_age, fill_None=fill_None)
  self.pop_age_mask = []
  for age_min, age_max in zip(self.pop_age_min, self.pop_age_max):
   if age_min is None and age_max is None:
    mask = np.ones(self.ageBase.shape, dtype=np.bool)
   elif age_min is None:
    mask = self.ageBase <= age_max if right else self.ageBase < age_max
   elif age_max is None:
    mask = self.ageBase > age_min if right else self.ageBase >= age_min
   else:
    if both:
     mask = (self.ageBase >= age_min) & (self.ageBase <= age_max)
    else:
     if right:
      mask = (self.ageBase > age_min) & (self.ageBase <= age_max)
     else:
      mask = (self.ageBase >= age_min) & (self.ageBase < age_max)
   self.pop_age_mask.append(mask)
  self.pop_age_mask = np.array(self.pop_age_mask)

 def get_pop_age(self, pop, lmask=None):
  if lmask is None:
   lmask = self.pop_age_mask
  if lmask is not None:
   return array_mask1D_operation(pop, lmask, axis=0, func=np.sum)

 def _set_sfr_mask_age(self, t_sf=None, popx_min=None):
  mdict = {'set_sfr_mask_age': dict(t_sf=t_sf, popx_min=popx_min)}
  self.updateMetadict(mdict)
  self.t_sf = t_sf
  self.sfr_mask_age = None
  self.popx_mask__yx = None
  if t_sf is not None:
   self.sfr_mask_age = self.ageBase <= t_sf
   if popx_min is not None:
    self.popx_mask__yx = self.popx__tyx[self.sfr_mask_age,...].sum(axis=0) < popx_min

 def prop_popx_mask__yx(self, prop):
  if self.popx_mask__yx is not None:
   prop = np.ma.array(prop, mask=self.popx_mask__yx)
  return prop

 def updateMetaDictArgs(self, keys, **kwargs):
  keys = checkList(keys)
  ldict = []
  for key in keys:
   if not key in self.metadict:
    print ('WARNING: Key "%s" NOT in metadict [%s]' % (key, ' | '.join(self.metadict.keys())))
   else:
    ldict.append(copy.deepcopy(self.metadict.get(key, {})))
  if len(ldict) < 1:
   return
  ref_dict = ldict[0]
  for kdict in ldict[1:]:
   ref_dict.update(kdict)
  ref_dict.update(kwargs)
  return ref_dict

 def fillMstars(self, mstars, mask_value=0.0):
  ''' Substitutes empty values by the mean of the age row, averaging 
      values of Mstars of different metallicities but same age'''
  #from sklearn.preprocessing import Imputer
  #imp = Imputer(strategy="mean", axis=1, missing_values=0.0)
  #Mstars = imp.fit_transform(Mstars)
  mstars = mstars.copy()
  mstars[mstars == mask_value] = np.nan
  axis_mean = np.nanmean(mstars, axis=1)
  inds = np.where(np.isnan(mstars))
  mstars[inds] = np.take(axis_mean, inds[0])
  return mstars

 def interp(self, x, y, dx, kind='linear', bounds_error=False, fill_value=np.nan, masked=True, inverse=False):
  if x is None or y is None or dx is None:
   return
  return interpMaskedArray(x, y, dx, kind=kind, bounds_error=bounds_error, fill_value=fill_value, masked=masked, inverse=inverse)

 def getAperturesPropLXY(self, prop, apertures, get_nzones=False, isError=False, area=True, mask_value=None, transpose=True, relative=False):
  if apertures is None:
   return
  prop_apertures = np.ma.zeros((apertures.shape[0], prop.shape[0]), dtype=prop.dtype)
  prop_apertures.mask = False
  nzones = np.zeros(apertures.shape[0], dtype=np.int)
  for i, aperture in enumerate(apertures):
   if aperture.sum() < 1:
    prop_apertures.mask[i, :] = True
    continue
   nzones[i] = aperture.sum()
   area_norm = float(nzones[i]) if area else 1.
   if not isError:
    prop_apertures[i] = prop[:, aperture].sum(axis=1) / area_norm
   else:
    prop_apertures[i] = np.sqrt(np.power(prop[:, aperture], 2.).sum(axis=1) / area_norm)
   if relative:
    prop_apertures[i] /= prop_apertures[i][0]
  if mask_value is not None:
   prop_apertures.data[prop_apertures.mask] = mask_value
  if transpose:
   prop_apertures = prop_apertures.T
  if get_nzones:
   return prop_apertures, nzones
  else:
   return prop_apertures

 def growthAgeAperturesPropXY(self, prop, apertures, get_nzones=False, relative=False, mask_value=None, tensor=None):
  if apertures is None:
   return
  prop_apertures = np.ma.zeros((apertures.shape[0], prop.shape[0]), dtype=prop.dtype)
  # Initialize mask to False for ALL values. If not, mask attribute is not available
  prop_apertures.mask = False
  nzones = np.zeros(apertures.shape[0], dtype=np.int)
  for i, aperture in enumerate(apertures):
   if aperture.sum() < 1:
    prop_apertures.mask[i, :] = True
    continue
   nzones[i] = aperture.sum()
   aprop = prop[..., aperture].sum(axis=-1)
   if tensor is None:
    prop_apertures[i] = np.ma.cumsum(aprop[::-1], axis=0)[::-1]
   else:
    prop_apertures[i] = np.tensordot(tensor, aprop, (1,0))
   if relative:
    prop_apertures[i] /= prop_apertures[i][0]
  if mask_value is not None:
   prop_apertures.data[prop_apertures.mask] = mask_value
  if get_nzones:
   return prop_apertures, nzones
  else:
   return prop_apertures

 def growthAgeAperturesDerivative(self, prop, sign=1.):
  if prop is None:
   return None
  else:
   iprop_growth__Apt = sci.interp1d(self.ageBase, prop, axis=1)(self.iageBase)
   prop_growth__Apdt = [sign * derivative(self.iageBase, iprop_growth__Apt[i,:], s=self.s, k=self.k, bspline=self.bspline) for i in range(iprop_growth__Apt.shape[0])]
   prop_growth__Apdt = np.ma.vstack(prop_growth__Apdt)
   prop_growth__Apdt.fill_value = self.fill_value
   return prop_growth__Apdt

 def getApertures(self, bin_r=None, bin_r_dr=None, bin_r_in=None, bin_r_out=None, r_in=0.0, r_out=3.0, dr=0.1, default=True, 
	mask_negative=True, method='exact', phot=True, right=False, both=False, mask=None, r_units='rad_scale', 
	central=False, integrate_apertures=False, integrated=False):
  rbin = self._get_bin_r(bin_r=bin_r, bin_r_dr=bin_r_dr, bin_r_in=bin_r_in, bin_r_out=bin_r_out, r_in=r_in, r_out=r_out, dr=dr, default=default, mask_negative=mask_negative)
  if rbin[1] is None and rbin[2] is None:
   return
  return self.getApertureMask(rbin[1], rbin[2], method=method, phot=phot, right=right, both=both, r_units=r_units, mask=mask,
		central=central, integrate_apertures=integrate_apertures, integrated=integrated)

 def radialApertures(self, bin_r=None, bin_r_dr=None, bin_r_in=None, bin_r_out=None, r_in=0.0, r_out=3.0, dr=0.1, 
	default=True, mask_negative=True, method='exact', phot=True, right=False, both=False, mask=None, 
	r_units='rad_scale'):
  rbin = self._get_bin_r(bin_r=bin_r, bin_r_dr=bin_r_dr, bin_r_in=bin_r_in, bin_r_out=bin_r_out, r_in=r_in, r_out=r_out, dr=dr, default=default, mask_negative=mask_negative)
  if rbin[1] is None and rbin[2] is None:
   return
  return self.getApertureMask(rbin[1], rbin[2], method=method, phot=phot, right=right, both=both, r_units=r_units, 
		central=False, integrate_apertures=False, integrated=False)

 def propAperturesGen(self, prop, apertures, transpose=False, relative=False, **kwargs):
  if apertures is None:
   return
  prop_apertures = propApertures(prop, apertures, **kwargs)
  if relative:
   prop_apertures /= prop_apertures[0]
  if transpose:
   prop_apertures = prop_apertures.T
  return prop_apertures

 def propApertures(self, prop, apertures, mode=None, mask=None, vmask=None, filled=True, area=False, transpose=False, relative=False):
  if mode is None:
   mode = self.radial_mode
  if mask is None:
   mask = self.maskXY
  if vmask is None:
   vmask = self.fill_value
  return self.propAperturesGen(prop, apertures, mode=mode, mask=mask, vmask=vmask, area=area, filled=filled, transpose=transpose, relative=relative)

 def propAp(self, prop, apertures=None, mode=None, mask=None, vmask=None, area=False, filled=True, relative=False, transpose=True):
  if apertures is None:
   apertures = self.apertures
  if mode is None:
   mode = self.aperture_mode
  if mask is None:
   mask = self.maskXY
  if vmask is None:
   vmask = self.fill_value
  # transpose = True --> For apertures we defined the shape [N_Apertures, ...], while propApertures gives [..., N_Apertures]
  return self.propAperturesGen(prop, apertures, mode=mode, mask=mask, vmask=vmask, area=area, filled=filled, transpose=transpose, relative=relative)

 def propApIntensive(self, prop, apertures=None, mode=None, mask=None, vmask=None, area=False, filled=True, relative=False, transpose=True):
  if apertures is None:
   apertures = self.apertures
  if mode is None:
   mode = self.aperture_intensive_mode
  if mask is None:
   mask = self.maskXY
  if vmask is None:
   vmask = self.fill_value
  if area is None:
   area = self.aperture_intensive_area
  # transpose = True --> For apertures we defined the shape [N_Apertures, ...], while propApertures gives [..., N_Apertures]
  return self.propAperturesGen(prop, apertures, mode=mode, mask=mask, vmask=vmask, area=area, filled=filled, transpose=transpose, relative=relative)

 def propApSD(self, prop, apertures=None, mode=None, mask=None, vmask=None, area=None, filled=True, relative=False, transpose=True):
  if apertures is None:
   apertures = self.apertures
  if mode is None:
   mode = self.aperture_mode
  if mask is None:
   mask = self.maskXY
  if vmask is None:
   vmask = self.fill_value
  if area is None:
   area = self.aperture_areaSD
  # transpose = True --> For apertures we defined the shape [N_Apertures, ...], while propApertures gives [..., N_Apertures]
  return self.propAperturesGen(prop, apertures, mode=mode, mask=mask, vmask=vmask, area=area, filled=filled, transpose=transpose, relative=relative)

 def propApL(self, prop, apertures=None, mode=None, mask=None, vmask=None, area=False, filled=True, relative=False, transpose=False):
  if apertures is None:
   apertures = self.apertures
  if mode is None:
   mode = self.spectral_mode
  if mask is None:
   mask = self.maskXY
  if vmask is None:
   vmask = self.fill_value
  # transpose = False --> For spectral mode we defined the shape [L, N_Apertures] and propApertures gives [..., N_Apertures]
  return self.propAperturesGen(prop, apertures, mode=mode, mask=mask, vmask=vmask, area=area, filled=filled, transpose=transpose, relative=relative)

 def prop_apr(self, prop, apertures=None, mode=None, mask=None, vmask=None, filled=True, area=False, relative=False, transpose=False):
  if apertures is None:
   apertures = self.bin_apr
  if mode is None:
   mode = self.apr_mode
  if mask is None:
   mask = self.maskXY
  if vmask is None:
   vmask = self.fill_value
  # For spectral mode we defined the shape [L, N_Apertures] and propApertures gives [..., N_Apertures]
  return self.propAperturesGen(prop, apertures, mode=mode, mask=mask, vmask=vmask, area=area, filled=filled, transpose=transpose, relative=relative)

 def prop_aprSD(self, prop, apertures=None, mode=None, mask=None, vmask=None, filled=True, area=None, relative=False, transpose=False):
  if apertures is None:
   apertures = self.bin_apr
  if mode is None:
   mode = self.apr_mode
  if mask is None:
   mask = self.maskXY
  if vmask is None:
   vmask = self.fill_value
  if area is None:
   area = self.apr_areaSD
  # For spectral mode we defined the shape [L, N_Apertures] and propApertures gives [..., N_Apertures]
  return self.propAperturesGen(prop, apertures, mode=mode, mask=mask, vmask=vmask, area=area, filled=filled, transpose=transpose, relative=relative)

 def radialProfileBin(self, prop, bin_r=None, rad_scale=None, mask=None, mode=None):
  if bin_r is None:
   bin_r = self.bin_r_profile
  if rad_scale is None:
   rad_scale = self.rad_scale
  if mask is None:
   mask = self.maskXY
  if mode is None:
   mode = self.radial_mode
  return self.radialProfile(prop, bin_r, rad_scale=rad_scale, mask=mask, mode=mode)

 def radialProfileApertures(self, prop, mode=None, mask=None, vmask=None, area=False, filled=True, **kwargs):
  mkeys = ['set_bin_radial_ap', 'set_bin_r']
  radial_dict = self.updateMetaDictArgs(mkeys, **kwargs)
  apertures = self.radialApertures(**radial_dict)
  return self.propApertures(prop, apertures, mode=mode, mask=mask, vmask=vmask, area=area, filled=filled)

 def radialProfileAp(self, prop):
  return propApertures(prop, self.bin_radial_ap, mode=self.radial_mode, mask=self.maskXY, vmask=self.fill_value)

 def radialProfileGen(self, prop, **kwargs):
  if self.radial_profile_basic:
   return self.radialProfileBin(prop, **kwargs)
  else:
   return self.radialProfileAp(prop)

 def getGalProp(self, **kwargs):
  self.gprop = getGalProp(self, **kwargs)

 def getHalfRadiusProp(self, prop, fill=False, mask=None):
  if prop is None:
   return
  if fill:
   if isinstance(prop, np.ma.MaskedArray):
    prop = prop.data
   prop, mask = self.fillImage(prop)
  if mask is None:
   mask = self.qMask
  return getGenHalfRadius(prop[mask], self.pixelDistance__yx[mask])

 def plotProp2D(self, prop=None, figname=None, show=False, plot_mask=False, dim_mask=None, 
	ellip_filled=False, ellip=False, mask=None, x0=None, y0=None, ba=None, pa=None, 
	radius=None, radius_filled=None, ba_filled=None, pa_filled=None, ls='dashed', 
	ls_filled='dashed', color='y', color_filled='y', ax=None, fig=None, 
	colorbar=True, set_clabel=True, rad_scale=False, **kwargs):
  from matplotlib.patches import Ellipse
  import matplotlib.pyplot as plt
  if fig is None:
   fig = plt.figure()
  if ax is None:
   ax = fig.add_subplot(111)
  if prop is None:
   prop = 'qSignal'
  if isinstance(prop, str):
   image = getattr(self, prop, None)
   if image is None:
    print ('WARNING: Property "%s" NOT available!' % prop)
    return
   image = image.copy()
  else:
   image = prop.copy()
  if image.ndim != 2:
   if isinstance(prop, str):
    print ('WARNING: Property "%s" has ndim=%i (ndim=2 required)' % (prop, image.ndim))
   else:
    print ('WARNING: Property has ndim=%i (ndim=2 required)' % (image.ndim))
   return
  if dim_mask is not None:
   for key in dim_mask:
    image[image == key] = dim_mask[key]
  if mask is not None and plot_mask:
   image[mask] = np.nan
  if colorbar and fig is not None:
   cax = ax.imshow(image, **kwargs)
   cbar = fig.colorbar(cax)
  if isinstance(prop, str) and set_clabel:
   cbar.set_label(prop, fontsize=15)
  # Ellipse
  if x0 is None:
   x0 = self.x0
  if y0 is None:
   y0 = self.y0
  if ellip:
   if ba is None:
    ba = self.ba
   if pa is None:
    pa = np.rad2deg(self.pa)
   if radius is None:
    radius = self.rad_scale if rad_scale else self.getHLR_pix(fill=False)
   semi_major = radius
   semi_minor = semi_major * ba
   major = 2. * semi_major
   minor = 2. * semi_minor
   ellip = Ellipse(xy=(x0, y0), width=major, height=minor, angle=pa, fill=False, linewidth=3, ls=ls, color=color)
   ax.add_artist(ellip)
  if ellip_filled:
   if ba_filled is None:
    ba_filled = self.ba
   if pa_filled is None:
    pa_filled = np.rad2deg(self.pa)
   if radius_filled is None:
    radius_filled = self.HLR_pix_fill
   semi_major = radius_filled
   semi_minor = semi_major * ba_filled
   major = 2. * semi_major
   minor = 2. * semi_minor
   ellip = Ellipse(xy=(x0, y0), width=major, height=minor, angle=pa_filled, fill=False, linewidth=3, ls=ls_filled, color=color_filled)
   ax.add_artist(ellip)
  ax.set_title('%s [%s]' % (self.galaxyName, self.califaID), fontsize=20)
  if figname is not None:
   fig.savefig(figname)
  if show:
   plt.show()
  return fig, ax
  
 @property
 def maskFlag__lz(self):
  return self.f_flag == 0

 @property
 def maskErr__lz(self):
  return self.f_err > 0

 @property
 def maskWei__lz(self):
  return self.f_wei > 0

 @property
 def f_err_median(self):
  return replaceMaskMedian(self.f_err, big=self.big_errors)

 @property
 def filter_residual__yx(self):
  return self.zoneToYX(self.filter_residual__z, extensive=False, surface_density=False)

 @property
 def masked_residuals__z(self):
  residuals = (self.f_obs - self.f_syn) / self.fobs_norm
  return np.ma.array(residuals, mask=~self.mask__lz)

 @property
 def masked_residuals__lyx(self):
  return self.zoneToYX(self.masked_residuals__z, extensive=True, surface_density=False)

 @property
 def masked_residuals__ayx(self):
  return self.zoneToYX(self.masked_residuals__z / self.zoneArea_pix, extensive=False, surface_density=False)

 @property
 def masked_residuals(self):
  return self.masked_residuals__z.sum(axis=1)

 @property
 def masked_residuals_n(self):
  return self.mask__lz.sum(axis=1)

 @property
 def masked_residuals_norm(self):
  return self.masked_residuals / self.masked_residuals_n

 @property
 def residuals__z(self):
  return (self.f_obs - self.f_syn) / self.fobs_norm

 @property
 def residuals__lyx(self):
  return self.zoneToYX(self.residuals__z, extensive=True, surface_density=False)

 @property
 def residuals__ayx(self):
  return self.zoneToYX(self.residuals__z / self.zoneArea_pix, extensive=False, surface_density=False)

 @property
 def residuals(self):
  return self.residuals__z.sum(axis=1)

 @property
 def residuals_norm(self):
  return self.residuals / self.N_zone

 @property
 def fobs_norm__lyx(self):
  return self.zoneToYX(self.fobs_norm, extensive=True, surface_density=False)

 @property
 def f_obs__ayx(self):
  return self.zoneToYX(self.f_obs, extensive=False, surface_density=False) * self.zoneWeight

 @property
 def f_syn__ayx(self):
  return self.zoneToYX(self.f_syn, extensive=False, surface_density=False) * self.zoneWeight

 @property
 def fobs_norm__ayx(self):
  return self.zoneToYX(self.fobs_norm, extensive=False, surface_density=False) * self.zoneWeight

 @property
 def f_err__ayx(self):
  return self.zoneToYX(self.f_err, extensive=False, surface_density=False) * self.zoneWeight

 @property
 def f_obs__lApSD(self):
  return self.propApL(self.f_obs__lyx, area=True)

 @property
 def f_obs__lAp(self):
  return self.propApL(self.f_obs__lyx)

 @property
 def f_obs__aApSD(self):
  return self.propApL(self.f_obs__ayx, area=True)

 @property
 def f_obs__aAp(self):
  return self.propApL(self.f_obs__ayx)

 @property
 def f_syn__lApSD(self):
  return self.propApL(self.f_syn__lyx, area=True)

 @property
 def f_syn__lAp(self):
  return self.propApL(self.f_syn__lyx)

 @property
 def f_syn__aApSD(self):
  return self.propApL(self.f_syn__ayx, area=True)

 @property
 def f_syn__aAp(self):
  return self.propApL(self.f_syn__ayx)

 @property
 def fobs_norm__lApSD(self):
  return self.propApL(self.fobs_norm__lyx, area=True)

 @property
 def fobs_norm__lAp(self):
  return self.propApL(self.fobs_norm__lyx)

 @property
 def fobs_norm__aApSD(self):
  return self.propApL(self.fobs_norm__ayx, area=True)

 @property
 def fobs_norm__aAp(self):
  return self.propApL(self.fobs_norm__ayx)

 @property
 def f_err__lApSD(self):
  return self.getAperturesPropLXY(self.f_err__lyx, self.apertures, area=True, isError=True, mask_value=self.fill_value)

 @property
 def f_err__lAp(self):
  return self.getAperturesPropLXY(self.f_err__lyx, self.apertures, area=False, isError=True, mask_value=self.fill_value)

 @property
 def f_err__aApSD(self):
  return self.getAperturesPropLXY(self.f_err__ayx, self.apertures, area=True, isError=True, mask_value=self.fill_value)

 @property
 def f_err__aAp(self):
  return self.getAperturesPropLXY(self.f_err__ayx, self.apertures, area=False, isError=True, mask_value=self.fill_value)

 @property
 def masked_residuals__lApSD(self):
  return self.propApL(self.masked_residuals__lyx, area=True)

 @property
 def masked_residuals__lAp(self):
  return self.propApL(self.masked_residuals__lyx)

 @property
 def masked_residuals__aApSD(self):
  return self.propApL(self.masked_residuals__ayx, area=True)

 @property
 def masked_residuals__aAp(self):
  return self.propApL(self.masked_residuals__ayx)

 @property
 def residuals__lApSD(self):
  return self.propApL(self.residuals__lyx, area=True)

 @property
 def residuals__lAp(self):
  return self.propApL(self.residuals__lyx)

 @property
 def residuals__aApSD(self):
  return self.propApL(self.residuals__ayx, area=True)

 @property
 def residuals__aAp(self):
  return self.propApL(self.residuals__ayx)

 @property
 def logMetBase(self):
  return np.log10(self.metBase/self.zsun)

 @property
 def popx__tyx(self):
  return self.popx__tZyx.sum(axis=1)

 @property
 def popx_sum__yx(self):
  return self.popx__tyx.sum(axis=0)

 @property
 def popx_norm__tyx(self):
  return self.popx__tyx / self.popx_sum__yx[np.newaxis,...]

 @property
 def popx__Ag_yx(self):
  return self.get_pop_age(self.popx__tyx)

 @property
 def popx_norm__Ag_yx(self):
  return self.get_pop_age(self.popx_norm__tyx)

 @property
 def popx_norm_global__Ag(self):
  return self.popx_norm__Ag_yx.sum(axis=-1).sum(axis=-1) / self.zoneArea_pix.sum().astype(np.float)

 @property
 def popx_norm__Ag_Ap(self):
  return self.propAp(self.popx_norm__Ag_yx, area=True)

 @property
 def popmu_cor__tyx(self):
  return self.popmu_cor__tZyx.sum(axis=1)

 @property
 def popmu_cor_sum__yx(self):
  return self.popmu_cor__tyx.sum(axis=0)

 @property
 def popmu_cor_norm__tyx(self):
  return self.popmu_cor__tyx / self.popmu_cor_sum__yx[np.newaxis,...]

 @property
 def popmu_cor__Ag_yx(self):
  return self.get_pop_age(self.popmu_cor__tyx)

 @property
 def popmu_cor_norm__Ag_yx(self):
  return self.get_pop_age(self.popmu_cor_norm__tyx)

 @property
 def popmu_cor_norm_global__Ag(self):
  return self.popmu_cor_norm__Ag_yx.sum(axis=-1).sum(axis=-1) / self.zoneArea_pix.sum().astype(np.float)

 @property
 def popmu_cor_norm__Ag_Ap(self):
  return self.propAp(self.popmu_cor_norm__Ag_yx, area=True)

 @property
 def Mcor_tot_sum(self):
  return self.Mcor_tot.sum()

 @property
 def Mini_tot_sum(self):
  return self.Mini_tot.sum()

 @property
 def alogZ_flux__z(self):
  popx_sum   = self.popx.sum(axis=1).sum(axis=0)
  popxZ_norm = self.popx.sum(axis=0) / popx_sum 
  return np.tensordot(popxZ_norm, self.logMetBase, (0, 0))

 @property
 def alogZ_mass__z(self):
  popmu_cor_sum  = self.popmu_cor.sum(axis=1).sum(axis=0)
  popmu_corZ_norm = self.popmu_cor.sum(axis=0) / popmu_cor_sum 
  return np.tensordot(popmu_corZ_norm, self.logMetBase, (0, 0))

 @property
 def alogZ_flux__yx(self):
  return self.zoneToYX(self.alogZ_flux__z, extensive=False)

 @property
 def alogZ_mass__yx(self):
  return self.zoneToYX(self.alogZ_mass__z, extensive=False)

 @property
 def age_flux__z(self):
  '''
  Flux-weighted average age, in voronoi zones.

      * Units: :math:`[\log Gyr]`
      * Shape: ``(N_zone)``
  '''
  if self.hasAlphaEnhancement:
      popx_sumZ = self.popx.sum(axis=2).sum(axis=1)
  else:
      popx_sumZ = self.popx.sum(axis=1)
  popx_sum = popx_sumZ.sum(axis=0)
  popx_sumZ /= popx_sum
  return np.tensordot(popx_sumZ, self.ageBase, (0, 0))

 @property
 def age_flux__yx(self):
  '''
  Spatially resolved, flux-weighted average log. age.

      * Units: :math:`[\log Gyr]`
      * Shape: ``(N_y, N_x)``
  '''
  return self.zoneToYX(self.age_flux__z, extensive=False)

 @property
 def age_mass__z(self):
  '''
  Mass-weighted average log. age, in voronoi zones.

      * Units: :math:`[\log Gyr]`
      * Shape: ``(N_zone)``
  '''
  if self.hasAlphaEnhancement:
      popmu_cor_sumZ = self.popmu_cor.sum(axis=2).sum(axis=1)
  else:
      popmu_cor_sumZ = self.popmu_cor.sum(axis=1)
  popmu_cor_sum = popmu_cor_sumZ.sum(axis=0)
  popmu_cor_sumZ /= popmu_cor_sum
  return np.tensordot(popmu_cor_sumZ, self.ageBase, (0, 0))

 @property
 def age_mass__yx(self):
  '''
  Spatially resolved, mass-weighted average log. age.

      * Units: :math:`[\log Gyr]`
      * Shape: ``(N_y, N_x)``
  '''
  return self.zoneToYX(self.age_mass__z, extensive=False)

 @property
 def HLR_pix_nofill(self):
  return self.getHLR_pix(fill=False)

 @property
 def HLR_parsec(self):
  return self.getHLR_pix_nofill * self.parsecPerPixel

 @property
 def HLR_pix_fill(self):
  return self.getHLR_pix(fill=True)

 @property
 def HLR_parsec_fill(self):
  return self.HLR_pix_fill * self.parsecPerPixel

 @property
 def DeRedHLR_LobnSD_pix(self):
  return self.getHalfRadiusProp(self.DeRed_LobnSD__yx, fill=False)

 @property
 def DeRedHLR_pix(self):
  return self.getHalfRadiusProp(self.qSignalDeRed, fill=False)

 @property
 def DeRedHLR_parsec(self):
  return self.DeRedHLR_pix * self.parsecPerPixel

 @property
 def DeRedHLR_LobnSD_pix_fill(self):
  return self.getHalfRadiusProp(self.DeRed_LobnSD__yx, fill=True)

 @property
 def DeRedHLR_pix_fill(self):
  return self.getHalfRadiusProp(self.qSignalDeRed, fill=True)

 @property
 def DeRedHLR_parsec_fill(self):
  return self.DeRedHLR_pix_fill * self.parsecPerPixel

 @property
 def imageFilterHLR_pix(self):
  return self.getHalfRadiusProp(self.imageFilter, fill=False)

 @property
 def imageFilterHLR_pix_fill(self):
  return self.getHalfRadiusProp(self.imageFilter, fill=True)

 @property
 def imageFilterHLR_parsec(self):
  if self.imageFilter is None:
   return
  return self.imageFilterHLR_pix * self.parsecPerPixel

 @property
 def imageFilterHLR_parsec_fill(self):
  if self.imageFilter is None:
   return
  return self.imageFilterHLR_pix_fill * self.parsecPerPixel

 @property
 def imageFilterDeRedHLR_pix(self):
  return self.getHalfRadiusProp(self.imageFilterDeRed, fill=False)

 @property
 def imageFilterDeRedHLR_pix_fill(self):
  return self.getHalfRadiusProp(self.imageFilterDeRed, fill=True)

 @property
 def imageFilterDeRedHLR_parsec(self):
  if self.imageFilterDeRed is None:
   return
  return self.imageFilterDeRedHLR_pix * self.parsecPerPixel

 @property
 def imageFilterDeRedHLR_parsec_fill(self):
  if self.imageFilterDeRed is None:
   return
  return self.imageFilterDeRedHLR_pix_fill * self.parsecPerPixel
  
 @property
 def HMR_pix(self):
  return self.getHalfRadiusProp(self.McorSD__yx, fill=False)

 @property
 def HMR_parsec(self):
  return self.HMR_pix * self.parsecPerPixel

 @property
 def HMR_pix_fill(self):
  return self.getHalfRadiusProp(self.McorSD__yx, fill=True)

 @property
 def HMR_parsec_fill(self):
  return self.HMR_pix_fill * self.parsecPerPixel

 @property
 def Mcor_tot_fill(self):
  return self.fillImage(self.McorSD__yx, mode=self.fill_mode).sum() * self.parsecPerPixel**2

 @property
 def Mini_tot_fill(self):
  return self.fillImage(self.MiniSD__yx, mode=self.fill_mode).sum() * self.parsecPerPixel**2

 @property
 def v_d__r(self):
  return self.radialProfileGen(self.v_d__yx)

 @property
 def at_flux__r(self):
  return self.radialProfileGen(self.at_flux__yx)

 @property
 def at_flux__Ap(self):
  return self.propApIntensive(self.at_flux__yx)

 @property
 def at_mass__r(self):
  return self.radialProfileGen(self.at_mass__yx)

 @property
 def at_mass__Ap(self):
  return self.propApIntensive(self.at_mass__yx)

 @property
 def aZ_flux__r(self):
  return self.radialProfileGen(self.aZ_flux__yx)

 @property
 def aZ_flux__Ap(self):
  return self.propApIntensive(self.aZ_flux__yx)

 @property
 def aZ_mass__r(self):
  return self.radialProfileGen(self.aZ_mass__yx)

 @property
 def aZ_mass__Ap(self):
  return self.propApIntensive(self.aZ_mass__yx)

 @property
 def alogZ_flux__r(self):
  return self.radialProfileGen(self.alogZ_flux__yx)

 @property
 def alogZ_flux__Ap(self):
  return self.propApIntensive(self.alogZ_flux__yx)

 @property
 def alogZ_mass__r(self):
  return self.radialProfileGen(self.alogZ_mass__yx)

 @property
 def alogZ_mass__Ap(self):
  return self.propApIntensive(self.alogZ_mass__yx)

 @property
 def A_V__r(self):
  return self.radialProfileGen(self.A_V__yx)

 @property
 def A_V__Ap(self):
  return self.propApIntensive(self.A_V__yx)

 @property
 def Mcor__r(self):
  return self.radialProfileGen(self.Mcor__yx)

 @property
 def Mini__r(self):
  return self.radialProfileGen(self.Mini__yx)

 @property
 def McorSD__r(self):
  return self.radialProfileGen(self.McorSD__yx)

 @property
 def MiniSD__r(self):
  return self.radialProfileGen(self.MiniSD__yx)

 @property
 def LobnSD__r(self):
  return self.radialProfileGen(self.LobnSD__yx)

 @property
 def DeRed_LobnSD__r(self):
  return self.radialProfileGen(self.DeRed_LobnSD__yx)

 @property
 def ML__r(self):
  return self.radialProfileGen(self.ML__yx)

 @property
 def DeRed_ML__r(self):
  return self.radialProfileGen(self.DeRed_ML__yx)

 @property
 def ML__yx(self):
  return self.zoneToYX(self.Mcor__z / self.Lobn__z, extensive=False, surface_density=False) * self.zoneWeight

 @property
 def DeRed_ML__yx(self):
  return self.zoneToYX(self.Mcor__z / self.DeRed_Lobn__z, extensive=False, surface_density=False) * self.zoneWeight

 @property
 def Mcor__tz(self):
  return self.Mcor__tZz.sum(axis=1)

 @property
 def Mcor__t(self):
  return self.Mcor__tZz.sum(axis=1).sum(axis=1)

 @property
 def norm_Mcor__t(self):
  return self.Mcor__t / self.Mcor_tot_sum
  
 @property
 def Mcor__tyx(self):
  return self.zoneToYX(self.Mcor__tZz.sum(axis=1), extensive=False, surface_density=False) * self.zoneWeight

 @property
 def Mcor__yx(self):
  return self.zoneToYX(self.Mcor__z, extensive=False, surface_density=False) * self.zoneWeight

 @property
 def masked_Mcor__yx(self):
  return np.ma.array(self.Mcor__yx, mask=~self.maskXY)

 @property
 def masked_Mcor__tyx(self):
  mask = np.ones_like(self.Mcor__tyx, dtype=np.bool) * self.maskXY
  return np.ma.array(self.Mcor__tyx, mask=~mask)

 @property
 def masked_Mcor__t(self):
  return self.masked_Mcor__tyx.sum(axis=1).sum(axis=1)

 @property
 def Mcor__tr(self):
  return self.radialProfileGen(self.Mcor__tyx)

 @property
 def masked_Mcor__tr(self):
  return self.radialProfileGen(self.masked_Mcor__tyx)

 @property
 def Mini__tz(self):
  return self.Mini__tZz.sum(axis=1)

 @property
 def Mini__t(self):
  return self.Mini__tZz.sum(axis=1).sum(axis=1)

 @property
 def norm_Mini__t(self):
  return self.Mini__t / self.Mini_tot_sum

 @property
 def Mini__tyx(self):
  return self.zoneToYX(self.Mini__tZz.sum(axis=1), extensive=False, surface_density=False) * self.zoneWeight

 @property
 def Mini__yx(self):
  return self.zoneToYX(self.Mini__z, extensive=False, surface_density=False) * self.zoneWeight

 @property
 def masked_Mini__tyx(self):
  mask = np.ones_like(self.Mini__tyx, dtype=np.bool) * self.maskXY
  return np.ma.array(self.Mini__tyx, mask=~mask)

 @property
 def masked_Mini__t(self):
  return self.masked_Mini__tyx.sum(axis=1).sum(axis=1)

 @property
 def Mini__tr(self):
  return self.radialProfileGen(self.Mini__tyx)

 @property
 def masked_Mini__tr(self):
  return self.radialProfileGen(self.masked_Mini__tyx)

 @property
 def McorSD__tyx(self):
  return self.McorSD__tZyx.sum(axis=1)

 @property
 def MiniSD__tyx(self):
  return self.MiniSD__tZyx.sum(axis=1)

 @property
 def McorSD__yx(self):
  return self.McorSD__tyx.sum(axis=0)

 @property
 def McorSD_fill__yx(self):
  return self.fillImage(self.McorSD__yx, mode=self.fill_mode)

 @property
 def Mcor__apr(self):
  return self.prop_apr(self.McorSD__yx * self.parsecPerPixel**2)

 @property
 def McorSD__apr(self):
  # Equivalent calculation
  #Mcor_rap = self.prop_apr(self.McorSD__yx * self.parsecPerPixel**2)
  #McorSD__apr = Mcor_rap / self.bin_r_ap_Area_pix / self.parsecPerPixel**2
  #McorSD__apr = Mcor_rap / self.bin_r_ap_Area_pc2
  return self.prop_apr(self.McorSD__yx, area=True)

 @property
 def McorSD_fill__apr(self):
  return self.prop_apr(self.McorSD_fill__yx, area=True)

 @property
 def McorSD_sum(self):
  return self.McorSD__yx.sum(axis=0).sum(axis=0)

 @property
 def McorSD_tot(self):
  '''As defined by equation 2 of Gonzalez Delgado et al (2014)'''
  # Equivalent to McorSD_tot_fill if there is nothing to fill
  return self.Mcor_tot_sum / self.zoneArea_pc2.sum()

 @property
 def McorSD_tot_fill(self):
  total_mass = self.McorSD_fill__yx.sum() * self.parsecPerPixel**2
  total_pix =  np.invert(self.McorSD_fill__yx.mask).sum()
  return total_mass / total_pix / self.parsecPerPixel**2

 @property
 def MiniSD__yx(self):
  return self.MiniSD__tyx.sum(axis=0)

 @property
 def MiniSD__apr(self):
  return self.prop_apr(self.MiniSD__yx, area=True)

 @property
 def Mini__apr(self):
  return self.prop_apr(self.MiniSD__yx * self.parsecPerPixel**2)

 @property
 def MiniSD_fill__yx(self):
  return self.fillImage(self.MiniSD__yx, mode=self.fill_mode)

 @property
 def MiniSD_sum(self):
  return self.MiniSD__yx.sum(axis=0).sum(axis=0)

 @property
 def MiniSD_tot(self):
  return self.Mini_tot.sum() / self.zoneArea_pc2.sum()

 @property
 def MiniSD_tot_fill(self):
  total_mass = self.MiniSD_fill__yx.sum() * self.parsecPerPixel**2
  total_pix =  np.invert(self.MiniSD_fill__yx.mask).sum()
  return total_mass / total_pix / self.parsecPerPixel**2

 @property
 def sfr_total__yx(self):
  prop = self.Mini__yx / self.ageBase.max()
  return self.prop_popx_mask__yx(prop)

 @property
 def sfr_total__r(self):
  return self.radialProfileGen(self.sfr_total__yx)

 @property
 def sfr_total__Ap(self):
  return self.propAp(self.sfr_total__yx)

 @property
 def sfrSD_total__yx(self):
  prop = self.MiniSD__yx / self.ageBase.max()
  return self.prop_popx_mask__yx(prop)

 @property
 def sfrSD_total__r(self):
  return self.radialProfileGen(self.sfrSD_total__yx)

 @property
 def sfrSD_total__Ap(self):
  # The following calculations are equivalent (with are=True in propAp):
  # sfrSD_total__yx.sum() * parsecPerPixel**2
  # sfr_total__yx.sum()
  # sfrSD_total__Ap[integrated] * zoneArea_pc2.sum()
  return self.propApSD(self.sfrSD_total__yx)

 @property
 def ssfr_total__yx(self):
  prop = self.sfr_total__yx / self.Mcor__yx
  return self.prop_popx_mask__yx(prop)

 @property
 def ssfr_total__r(self):
  return self.radialProfileGen(self.ssfr_total__yx)

 @property
 def ssfr_total__Ap(self):
  return self.propAp(self.ssfr_total__yx)

 @property
 def ssfrSD_total__yx(self):
  prop = self.sfrSD_total__yx / self.Mcor__yx
  return self.prop_popx_mask__yx(prop)

 @property
 def ssfrSD_total__r(self):
  return self.radialProfileGen(self.ssfrSD_total__yx)

 @property
 def ssfrSD_total__Ap(self):
  return self.propApSD(self.ssfrSD_total__yx)

 @property
 def sfr_tsf__yx(self):
  prop = self.Mini__tyx[self.sfr_mask_age].sum(axis=0) / self.t_sf
  return self.prop_popx_mask__yx(prop)

 @property
 def sfr_tsf__r(self):
  return self.radialProfileGen(self.sfr_tsf__yx)

 @property
 def sfr_tsf__Ap(self):
  return self.propAp(self.sfr_tsf__yx)

 @property
 def sfrSD_tsf__yx(self):
  prop = self.MiniSD__tyx[self.sfr_mask_age].sum(axis=0) / self.t_sf
  return self.prop_popx_mask__yx(prop)

 @property
 def sfrSD_tsf__r(self):
  return self.radialProfileGen(self.sfrSD_tsf__yx)

 @property
 def sfrSD_tsf__Ap(self):
  return self.propApSD(self.sfrSD_tsf__yx)

 @property
 def ssfr_tsf__yx(self):
  prop = self.sfr_tsf__yx / self.Mcor__yx
  return self.prop_popx_mask__yx(prop)

 @property
 def ssfr_tsf__r(self):
  return self.radialProfileGen(self.ssfr_tsf__yx)

 @property
 def ssfr_tsf__Ap(self):
  return self.propAp(self.ssfr_tsf__yx)

 @property
 def ssfrSD_tsf__yx(self):
  prop = self.sfrSD_tsf__yx / self.Mcor__yx
  return self.prop_popx_mask__yx(prop)

 @property
 def ssfrSD_tsf__r(self):
  return self.radialProfileGen(self.ssfrSD_tsf__yx)

 @property
 def ssfrSD_tsf__Ap(self):
  return self.propApSD(self.ssfrSD_tsf__yx)

 @property
 def mass_growth__tyx(self):
  return self.zoneToYX(self.mass_growth__tz, extensive=False, surface_density=False) * self.zoneWeight
  # The following gives different values, due to Mstars__tt
  #self.mass_growth__tyx = np.tensordot(self.Mstars__tt, self.Mini__tyx, (1,0)) 

 @property
 def mass_growth__t(self):
  return self.mass_growth__tyx.sum(axis=1).sum(axis=1)

 @property
 def mass_growth__tr(self):
  return self.radialProfileGen(self.mass_growth__tyx)

 @property
 def norm_mass_growth__tr(self):
  return self.mass_growth__tr / self.mass_growth__tr[0]

 @property
 def mass_growth__r_tg(self):
  return self.interp(self.norm_mass_growth__tr.T, self.ageBase, self.t_mass_growth, inverse=True)

 @property
 def mass_growth__r_tgmass(self):
  return self.interp(self.ageBase, self.mass_growth__tr.T, self.mass_growth__r_tg)

 @property
 def masked_mass_growth__tyx(self):
  mask = np.ones_like(self.mass_growth__tyx, dtype=np.bool) * self.maskXY
  return np.ma.array(self.mass_growth__tyx, mask=~mask)

 @property
 def masked_mass_growth__t(self):
  return self.masked_mass_growth__tyx.sum(axis=1).sum(axis=1)

 @property
 def norm_mass_growth__t(self):
  return self.masked_mass_growth__t / self.masked_mass_growth__t[0]

 @property
 def mass_growth__tg(self):
  return self.interp(self.norm_mass_growth__t, self.ageBase, self.t_mass_growth)

 @property
 def mass_growth__tgmass(self):
  return self.interp(self.ageBase, self.mass_growth__t, self.mass_growth__tg)

 @property
 def Mcor_growth__t(self):
  return np.ma.cumsum(self.masked_Mcor__t[::-1], axis=0)[::-1]
  #return np.tensordot(self.cumsum__tt, self.masked_Mcor__t, (1,0)) # Equivalent but does not treat masked values

 @property
 def norm_Mcor_growth__t(self):
  return self.Mcor_growth__t / self.Mcor_growth__t[0]

 @property
 def Mcor_growth__tg(self):
  return self.interp(self.norm_Mcor_growth__t, self.ageBase, self.t_mass_growth)

 @property
 def Mcor_growth__tgmass(self):
  return self.interp(self.ageBase, self.Mcor_growth__t, self.Mcor_growth__tg)

 @property
 def Mcor_growth__tr(self):
  return np.ma.cumsum(self.masked_Mcor__tr[::-1], axis=0)[::-1]
  #return np.tensordot(self.cumsum__tt, self.masked_Mcor__tr, (1,0)) # Does not mask masked values

 @property
 def norm_Mcor_growth__tr(self):
  return self.Mcor_growth__tr / self.Mcor_growth__tr[0]

 @property
 def Mcor_growth__r_tg(self):
  return self.interp(self.norm_Mcor_growth__tr.T, self.ageBase, self.t_mass_growth, inverse=True)

 @property
 def Mcor_growth__r_tgmass(self):
  return self.interp(self.ageBase, self.Mcor_growth__tr.T, self.Mcor_growth__r_tg)

 @property
 def Mcor_growth__Apt(self):
  return self.growthAgeAperturesPropXY(self.Mcor__tyx, self.apertures, relative=False, mask_value=self.fill_value)

 @property
 def norm_Mcor_growth__Apt(self):
  return self.growthAgeAperturesPropXY(self.Mcor__tyx, self.apertures, relative=True, mask_value=self.fill_value)

 @property
 def Mcor_growth__Ap_tg(self):
  return self.interp(self.norm_Mcor_growth__Apt, self.ageBase, self.t_mass_growth, inverse=True)

 @property
 def Mcor_growth__Ap_tgmass(self):
  return self.interp(self.ageBase, self.Mcor_growth__Apt, self.Mcor_growth__Ap_tg)

 @property
 def Mcor__Ap(self):
  return self.propAp(self.McorSD__yx * self.parsecPerPixel**2)

 @property
 def McorSD__Ap(self):
  return self.propApSD(self.McorSD__yx)

 @property
 def Mcor__Apt(self):
  return self.propAp(self.McorSD__tyx * self.parsecPerPixel**2)

 @property
 def norm_Mcor__Apt(self):
  return self.Mcor__Apt / self.Mcor__Ap[..., np.newaxis]

 @property
 def Mini_growth__t(self):
  return np.ma.cumsum(self.masked_Mini__t[::-1], axis=0)[::-1]
  # np.tensordot(self.cumsum__tt, self.masked_Mini__t, (1,0)) # Equivalent but does not treat masked values

 @property
 def norm_Mini_growth__t(self):
  return self.Mini_growth__t / self.Mini_growth__t[0]

 @property
 def Mini_growth__tg(self):
  return self.interp(self.norm_Mini_growth__t, self.ageBase, self.t_mass_growth)

 @property
 def Mini_growth__tgmass(self):
  return self.interp(self.ageBase, self.Mini_growth__t, self.Mini_growth__tg)

 @property
 def Mini_growth__tr(self):
  return np.ma.cumsum(self.masked_Mini__tr[::-1], axis=0)[::-1]
  #return np.tensordot(self.cumsum__tt, self.masked_Mini__tr, (1,0)) # Equivalent but does not treat masked values

 @property
 def norm_Mini_growth__tr(self):
  return self.Mini_growth__tr / self.Mini_growth__tr[0]

 @property
 def Mini_growth__r_tg(self):
  return self.interp(self.norm_Mini_growth__tr.T, self.ageBase, self.t_mass_growth, inverse=True)

 @property
 def Mini_growth__r_tgmass(self):
  return self.interp(self.ageBase, self.Mini_growth__tr.T, self.Mini_growth__r_tg)

 @property
 def Mini_growth__Apt(self):
  return self.growthAgeAperturesPropXY(self.Mini__tyx, self.apertures, relative=False, mask_value=self.fill_value)

 @property
 def norm_Mini_growth__Apt(self):
  return self.growthAgeAperturesPropXY(self.Mini__tyx, self.apertures, relative=True, mask_value=self.fill_value)

 @property
 def Mini_growth__Ap_tg(self):
  return self.interp(self.norm_Mini_growth__Apt, self.ageBase, self.t_mass_growth, inverse=True)

 @property
 def Mini_growth__Ap_tgmass(self):
  return self.interp(self.ageBase, self.Mini_growth__Apt, self.Mini_growth__Ap_tg)

 @property
 def Mini__Ap(self):
  return self.propAp(self.MiniSD__yx * self.parsecPerPixel**2)

 @property
 def MiniSD__Ap(self):
  return self.propApSD(self.MiniSD__yx)

 @property
 def Mini__Apt(self):
  return self.propAp(self.MiniSD__tyx * self.parsecPerPixel**2)

 @property
 def norm_Mini__Apt(self):
  return self.Mini__Apt / self.Mini__Ap[..., np.newaxis]

 @property
 def mass_growth__Apt(self):
  # The following gives similar results but slightly lower values due to the use of Mstars__tt
  #self.growthAgeAperturesPropXY(self.Mini__tyx, self.apertures, relative=False, mask_value=self.fill_value, tensor=self.Mstars__tt)
  return self.propAp(self.mass_growth__tyx)

 @property
 def mass_growth__Apt2(self):
  return self.getAperturesPropLXY(self.mass_growth__tyx, self.apertures, area=False, mask_value=self.fill_value, transpose=False)

 @property
 def norm_mass_growth__Apt(self):
  return self.propAp(self.mass_growth__tyx, relative=True)

 @property
 def mass_growth__Ap_tg(self):
  return self.interp(self.norm_mass_growth__Apt, self.ageBase, self.t_mass_growth, inverse=True)

 @property
 def mass_growth__Ap_tgmass(self):
  return self.interp(self.ageBase, self.mass_growth__Apt, self.mass_growth__Ap_tg)

 @property
 def Mcor_growth__dt(self):
  '''
        First derivative of the mass growth (stellar mass growth rate)

            * Units: :math:`[M_\odot / yr]`
  '''
  iMcor_growth__t = np.interp(self.iageBase, self.ageBase, self.Mcor_growth__t)
  return -derivative(self.iageBase, iMcor_growth__t, s=self.s, k=self.k, bspline=self.bspline)

 @property
 def Mcor_growth__dt_tot(self):
  return np.trapz(self.Mcor_growth__dt, self.iageBase)

 @property
 def Mcor_growth__dt_norm(self):
  return self.Mcor_growth__dt * self.Mcor_tot_sum / self.Mcor_growth__dt_tot

 @property
 def Mini_growth__dt(self):
  '''
        First derivative of the Mini growth (star formation growth rate)

            * Units: :math:`[M_\odot / yr]`
  '''
  iMini_growth__t = np.interp(self.iageBase, self.ageBase, self.Mini_growth__t)
  return -derivative(self.iageBase, iMini_growth__t, s=self.s, k=self.k, bspline=self.bspline)

 @property
 def Mini_growth__dt_tot(self):
  return np.trapz(self.Mini_growth__dt, self.iageBase)

 @property
 def Mini_growth__dt_norm(self):
  return self.Mini_growth__dt * self.Mini_tot_sum / self.Mini_growth__dt_tot

 @property
 def mass_growth__dt(self):
  '''
        First derivative of the mass growth (stellar mass growth rate)

            * Units: :math:`[M_\odot / yr]`
  '''
  imass_growth__t = np.interp(self.iageBase, self.ageBase, self.mass_growth__t)
  return -derivative(self.iageBase, imass_growth__t, s=self.s, k=self.k, bspline=self.bspline)

 @property
 def mass_growth__dt_tot(self):
  return np.trapz(self.mass_growth__dt, self.iageBase)

 @property
 def mass_growth__dt_norm(self):
  return self.mass_growth__dt * self.Mcor_tot_sum / self.mass_growth__dt_tot # ?

 @property
 def Mcor_growth__Apdt(self):
  return self.growthAgeAperturesDerivative(self.Mcor_growth__Apt, sign=-1.)

 @property
 def Mini_growth__Apdt(self):
  return self.growthAgeAperturesDerivative(self.Mini_growth__Apt, sign=-1.)

 @property
 def mass_growth__Apdt(self):
  return self.growthAgeAperturesDerivative(self.mass_growth__Apt, sign=-1.)

 @property
 def mass_growth__Apdt2(self):
  return self.growthAgeAperturesDerivative(self.mass_growth__Apt2, sign=-1.)

 @property
 def Mcor_growth__Apdt_tot(self):
  return np.trapz(self.Mcor_growth__Apdt, self.iageBase, axis=1)

 @property
 def Mini_growth__Apdt_tot(self):
  return np.trapz(self.Mini_growth__Apdt, self.iageBase, axis=1)

 @property
 def mass_growth__Apdt_tot(self):
  return np.trapz(self.mass_growth__Apdt, self.iageBase, axis=1)

 @property
 def Mcor_growth__Apdt_norm(self):
  return self.Mcor_growth__Apdt * self.Mcor__Ap[..., np.newaxis] / self.Mcor_growth__Apdt_tot[..., np.newaxis]

 @property
 def Mini_growth__Apdt_norm(self):
  return self.Mini_growth__Apdt * self.Mini__Ap[..., np.newaxis] / self.Mini_growth__Apdt_tot[..., np.newaxis]

 @property
 def mass_growth__Apdt_norm(self):
  return self.mass_growth__Apdt * self.Mcor__Ap[..., np.newaxis] / self.mass_growth__Apdt_tot[..., np.newaxis] # ?
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def readStarlightSpectra(lsynth,lspec=None,residual=True,error_units=1.0,slfig=False,
	slfig_dout='',flag_eval=None,median_error=500.,v0_max=125.,same_wave=False,
	filters=None,calc=None,sunfile=None,prop=None,name_sl=True,order_mag=False,
	table_mag=False,table_mag_fits=None,ikey='Name',addNone=False,order_keys2=[0,1],
	order_keys3=[0,2,1],replace_error=True,serror='e_',get_object=True,
	sep=None,isep=0,name_object='readStarlightSpectra',fmt='png',
	plot=False,fewhb=False,show=False):
 '''
 flag_eval: None or Dictionary
	flag_eval = {0.0: "(iflag != 9.0) & (iflag > 0.0)"}  --> iflag[eval(flag_eval[val])] = val
 '''

 result = {'dist':[],'filename':[],'bad':[],'wave':[],'dict_mag':None,'table_mag':None}
 magnitudes = OrderedDict()
 outfig = None

 lsynth = checkList(lsynth, include_array=True)

 if lspec is None:
  lspec = [None]*len(lsynth)
 else:
  lspec = checkList(lspec, include_array=True)

 v0 = []
 if not same_wave:
  flux  = []
  error = []
  synt  = []
  error = []
  flag  = []
  cont  = []
  
 for i,spec,synth in izip(count(),lspec,lsynth):
  try:
   if spec is not None:
    spec_wave,spec_error,spec_flag = np.loadtxt(spec,unpack=True,usecols=(0,2,3))
    spec_error *= error_units
   else:
    spec_wave = None
    spec_flag = None
    spec_error = None
   if slfig:
    outfig = '%s.%s' % (os.path.basename(synth),fmt)
    outfig = os.path.join(slfig_dout,outfig)
   K = ReadStarlightOutput(synth,error=spec_error,flag=spec_flag,ewave=spec_wave,plot=plot,show=show,
       savefig=slfig,outfile=outfig,fewhb=fewhb,filters=filters,calc=calc,sunfile=sunfile,prop=prop)
  except:
   result['bad'].append(spec)
   continue
  if residual:
   iflux  = K.f_res
   ierror = K.error
   icont  = K.f_obs
   isynt  = K.f_syn
   iv0    = K.v0
  else:
   iflux  = K.f_obs
   ierror = K.error
   isynt  = K.f_syn
   icont  = None
   iv0    = K.v0
  iflag = K.flag

  # Evaluate flag and error conditions
  if flag_eval is not None and iflag is not None:
   for val in flag_eval:
    iflag[eval(flag_eval[val])] = val
  if median_error is not None and ierror is not None:
   ierror[ierror/np.median(ierror) > median_error] = median_error*np.median(ierror)

  # Concatenate
  if same_wave:
   if i == 0:
    flux,synt,error,flag,cont = [np.reshape(item,(-1,1)) for item in [iflux,isynt,ierror,iflag,icont]]
   else:
    flux,synt,error,flag,cont = [np.hstack((a2d,np.reshape(a1d,(-1,1)))) for (a1d,a2d) in zip([iflux,isynt,ierror,iflag,icont],[flux,synt,error,flag,cont])]
  else:
   flux.append(iflux)
   synt.append(isynt)
   error.append(ierror)
   flag.append(iflag)
   cont.append(icont)
   result['wave'].append(K.wave)

  v0.append(iv0)
  result['dist'].append(K.distance_Mpc)
  if not name_sl and spec is not None:
   filename = os.path.basename(spec)
  else:
   filename = K.filename
  if sep is not None:
   filename = filename.split(sep)[isep]
  result['filename'].append(filename)

  # Magnitudes
  if K.magnitudes is not None:
   magnitudes[filename] = K.magnitudes

 if same_wave:
  result['wave'] = K.wave

 # Velocity corrections
 v0 = np.atleast_1d(v0)
 if v0_max is not None:
  v0[np.abs(v0) > v0_max] = 0.0

 result['dist']  = np.array(result['dist'])
 result['error'] = error if lspec[0] is not None else None
 result['flag']  = flag if lspec[0] is not None else None
 result['cont']  = cont if residual else None
 result['flux']  = flux
 result['synt']  = synt
 result['v0']    = v0

 result['dict_data'] = {'V0': v0}

 if len(magnitudes) == 0:
  magnitudes = None
 result['magnitudes'] = magnitudes
 if magnitudes is not None:
  if order_mag or table_mag or table_mag_fits is not None:
   result['dict_mag'] = orderDictTable(magnitudes,ikey=ikey,addNone=addNone,order_keys2=order_keys2,
        order_keys3=order_keys3,replace_error=replace_error,serror=serror)
   if table_mag or table_mag_fits is not None:
    try:
     from astropy.table import Table
     result['table_mag'] = Table(result['dict_mag'])
     if table_mag_fits is not None:
      result['table_mag'].write(table_mag_fits,overwrite=True)
    except:
     pass

 if get_object:
  result = dict2object(result,name=name_object)

 return result
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def readPycassoFits(fits,residual=True,integrated=True,flag_eval=None,
	median_error=None,v0_max=None,get_object=True,dict_head=None,
	name_object='readPycassoFits'):
 '''
 flag_eval: None or Dictionary
	flag_eval = {0.0: "(flag != 9.0) & (flag > 0.0)"}  --> flag[eval(flag_eval[val])] = val
 '''

 result = {'filename':os.path.basename(fits)}

 K = fitsDataCube(fits)

 # Integrated
 iflux  = K.integrated_f_obs
 isynt  = K.integrated_f_syn
 ires   = K.integrated_f_obs - K.integrated_f_syn
 ierror = K.integrated_f_err
 iflag  = K.integrated_f_flag
 iv0    = np.asarray(K.header['SYN INTEG V_0'],dtype=np.float)

 # Zones
 zflux  = K.f_obs
 zsynt  = K.f_syn
 zres   = K.f_obs - K.f_syn
 zerror = K.f_err
 zflag  = K.f_flag
 zv0    = K.v_0

 # All: first "zone" is the integrated spectrum
 if integrated:
  flux  = np.hstack((iflux.reshape((iflux.size,1)),zflux))
  synt  = np.hstack((isynt.reshape((isynt.size,1)),zsynt))
  res   = flux - synt
  error = np.hstack((ierror.reshape((ierror.size,1)),zerror))
  flag  = np.hstack((iflag.reshape((iflag.size,1)),zflag))
  v0    = np.hstack((iv0,zv0))
 else:
  flux  = zflux
  synt  = zsynt
  res   = zres
  error = zerror
  flag  = zflag
  v0    = zv0

 # Evaluate flag, velocity and error conditions
 if flag_eval is not None and flag is not None:
  for val in flag_eval:
   flag[eval(flag_eval[val])] = val

 if v0_max is not None:
  v0[np.abs(v0) > v0_max] = 0.0

 if median_error is not None and error is not None:
  error[error/np.median(error) > median_error] = median_error*np.median(error)

 # Fill dictionary
 result['wave']  = K.l_obs
 result['flux']  = res if residual else flux
 result['dist']  = K.distance_Mpc
 result['error'] = error
 result['flag']  = flag
 result['synt']  = synt
 result['cont']  = flux if residual else None
 result['res']   = res
 result['v0']    = v0
 result['K']     = K

 result['dict_head'] = {'D_MPC': K.distance_Mpc, 'REDSHIFT': K.header['REDSHIFT'], 'NED_NAME': K.header['NED_NAME']}
 result['dict_data'] = {'V0': v0}

 if integrated:
  result['dict_head']['ATTENT']    = '0 index zone IS the INTEGRATED spectrum'
  result['dict_head']['INTESPEC']  = True
 else:
  result['dict_head']['ATTENT']    = 'INTEGRATED spectrum NOT present'
  result['dict_head']['INTESPEC']  = False

 if dict_head is not None:
  result['dict_head'].update(dict_head)

 if get_object:
  result = dict2object(result,name=name_object)

 return result
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class getMagSpectrum():
 def __init__(self,filters,wave,flux,error=None,flag=None,sunfile=None,
  RV=3.1,AV=None,EBV=None,redshift=None,distance_Mpc=None,finite=True,
  prop=None,mask=0.0,units=1.0):

  from Py3D.core.spectrum1d import Spectrum1D
  from pyfu.passband import PassBand

  self.filters = filters
  self.wave = wave
  self.flux = flux
  self.error = error
  self.flag = flag
  self.RV = RV
  self.AV = AV
  self.EBV = EBV
  self.redshift = redshift
  self.finite = finite
  self.mask = mask
  self.distance_Mpc = distance_Mpc
  self.sunfile = sunfile
  self.prop = prop
  self.units = units
  self.PassBand = PassBand
  self.Spectrum1D = Spectrum1D
  self.sundict = OrderedDict()
  self.magnitudes = OrderedDict()
  self.rl = None

  # Estimate magnitudes
  self.setSpectrum()
  self.getMag()

 def setSpectrum(self):
  self.setDefaults()
  self.redshiftWave()
  self.LumDistFac()
  self.reddening()
  self.getSun()
  self.setProp()

 def setDefaults(self):
  self.flux *= self.units
  if self.error is not None:
   self.error *= self.units
  self._wave  = self.wave
  self._flux  = self.flux
  self._error = self.error
  self._flag  = self.flag

 def setProp(self):
  if self.prop is None:
   self.prop = ['Flux','Mag','AB','Lum','L']

 def redshiftWave(self):
  if self.redshift is not None and self.wave is not None:
   self.wave = self.wave/(1.0 + self.redshift)

 def LumDistFac(self,norm=1000.): # PC in cm
  if self.distance_Mpc is not None:
   self.lumdist   = 4*np.pi*np.power(self.distance_Mpc*1.0e6*CST.PC,2.0)
   self.nlumdist  = self.lumdist/(4*np.pi*np.power(norm*CST.PC,2.0))

 def reddening(self):
  if self.EBV is not None and self.AV is None:
   self.AV = self.RV * self.EBV
  if self.AV is None:
   return
  from pystarlight.util.redenninglaws import calc_redlaw
  # Reddening law
  self.rl = calc_redlaw(self.wave, self.RV, redlaw='CCM')
  self.flux *= np.power(10., 0.4 * self.rl * self.AV)
  if self.error is not None:
   self.error *= np.power(10., 0.4 * self.rl * self.AV)

 def getSun(self):
  if self.sunfile is not None:
   if isinstance(self.sunfile,str):
    fsun = pyfits.getdata(self.sunfile)
    self.sun = self.Spectrum1D(fsun['WAVELENGTH'],fsun['FLUX'])

 def LsunBand(self,passband,**kwargs):
  lsunband = None
  if self.sunfile is not None:
   flux,eflux = passband.getFluxPass(self.sun._wave,self.sun._data,**kwargs)
   lsunband = flux*4.*np.pi*np.power(CST.AU,2)
   if self.finite:
    lsunband = lsunband if np.isfinite(lsunband) else self.mask
  return lsunband

 def readPassBand(self,pfile):
  w,f = np.loadtxt(pfile,usecols=(0,1),unpack=True)
  return self.PassBand(wave=w,data=f)

 def getPassBandProp(self,passband,lsunband=None,units=1.,**kwargs):
  bprop  = OrderedDict()
  ebprop = OrderedDict()
  flux, eflux = passband.getFluxPass(self.wave,self.flux,error=self.error,mask=self.flag,**kwargs)
  for key in self.prop:
   if 'Flux' == key:
    bprop[key]  = flux
    ebprop[key] = eflux
   if 'Mag' == key:
    vmag,evmag  = passband.fluxToMag(flux,error=eflux,system='Vega',units=units)
    bprop[key]  = vmag
    ebprop[key] = evmag
   if 'AB' == key:
    mab,emab    = passband.fluxToMag(flux,error=eflux,system='AB',units=units)
    bprop[key]  = mab
    ebprop[key] = emab
   if ('Lum' == key or 'L' == key) and self.distance_Mpc is not None:
    lum,elum  = flux2Lum(flux,self.distance_Mpc*CST.PC*1.0e6,eflux)
    if 'Lum' == key:
     bprop[key]  = lum
     ebprop[key] = elum
    if 'L' == key and lsunband is not None:
     bprop[key]  = lum/lsunband
     ebprop[key] = elum/lsunband if elum is not None else None
   if self.finite and key in bprop:
    if isinstance(bprop[key], np.ndarray):
     bprop[key][~np.isfinite(bprop[key])] = self.mask
    else:
     bprop[key]  = bprop[key]  if np.isfinite(bprop[key])  else self.mask
    if ebprop[key] is not None:
     if isinstance(bprop[key], np.ndarray):
      ebprop[key][~np.isfinite(ebprop[key])] = self.mask
     else:
      ebprop[key] = ebprop[key] if np.isfinite(ebprop[key]) else self.mask
  return bprop, ebprop

 def getMag(self):
  for fkey in self.filters:
   passb = self.readPassBand(self.filters[fkey])
   lsunband = self.LsunBand(passb)
   if lsunband is not None:
    self.sundict[fkey+'_Lsun'] = lsunband
   bprop, ebprop = self.getPassBandProp(passb,lsunband=lsunband)
   self.magnitudes[fkey] = OrderedDict()
   for key in bprop:
    ekey = 'e_%s' % key
    self.magnitudes[fkey][key]  = bprop[key]
    self.magnitudes[fkey][ekey] = ebprop[key]
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def readTables(tables,readfits=True,readtable=True,selcols=None,rename=None,prefix=True,sep='_',get_list=False):
 from astropy.io.ascii import read
 import astropy

 ntables = []
 tables  = checkList(tables)
 if selcols is not None:
  if isinstance(selcols,str):
   selcols = [[selcols]]*len(tables)
  if isinstance(selcols,list) and not isinstance(selcols[0],list):
   selcols = [selcols]*len(tables)
  if len(selcols) != len(tables):
   print ('Number of keys (%i) != number of tables (%i)' % (len(selcols),len(tables)))
   selcols = None
 if rename is not None:
  if isinstance(rename,dict):
   rename = [rename]*len(tables)
  if isinstance(rename,list) and isinstance(rename[0],str) and len(rename) != len(tables):
   print ('Number of prefix (%i) != number of tables (%i)' % (len(rename),len(tables)))
   rename = None
  if rename is not None and isinstance(rename,dict) and len(rename) != len(tables):
   print ('Number of dicts (%i) != number of tables (%i)' % (len(rename),len(tables)))
   rename = None
 for i,table in enumerate(tables):
  if isinstance(table,astropy.table.table.Table):
   table = table
  elif isinstance(table,(pyfits.fitsrec.FITS_rec,astropy.io.fits.fitsrec.FITS_rec,dict)):
   table = Table(table)
  elif isinstance(table,str):
   if readfits:
    try:
     table = Table(pyfits.getdata(table))
    except:
     pass
   if readtable:
    try:
     table = read(table)
    except:
     pass
  else:
   print ('>>> Unknown format for table: %s' % table)
   continue
  if selcols is not None:
   if selcols[i] is not None:
    tcols = [col for col in selcols[i] if col in table.colnames]
    bcols = [col for col in selcols[i] if not col in table.colnames]
    table = table[tcols]
    if len(bcols) > 0:
     print ('Columns "%s" NOT available in table %i' % (' | '.join(bcols), i+1))
  if rename is not None:
   drename = rename[i]
   if drename is not None:
    renameColumns(table,drename,prefix=prefix,sep=sep)
  ntables.append(table)

 if len(tables) != len(ntables):
  print ('# INPUT tables (%i) NOT equal to # OUTPUT tables (%i)' % (len(tables,),len(ntables)))

 if len(ntables) == 1 and not get_list:
  ntables = ntables[0]

 return ntables
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def joinTables(tables,join_type='outer',readfits=True,readtable=True,keys=None,
	checkcols=True,order=None,id_order=[0,1],prefix=True,sep='_',dcol=None,
	include=None,exclude=None,mask=False,float_mask=np.nan,integer_mask=-999,
	string_mask='NAN',include_mask=None,exclude_mask=None,filled=False,
	mask_inf=True,order_before=True,mask_before=False,mask_values=False,
	string_masks=['NAN','--','-'],verbose=True,**kwargs):
 from astropy.io.ascii import read
 lkeys = keys
 tables = readTables(tables,readfits=readfits,readtable=readtable)
 if mask_before:
  ltables = []
  for table in tables:
   ltables.append(maskTableColumns(table,float_mask=float_mask,integer_mask=integer_mask,
        string_mask=string_mask,include=include_mask,exclude=exclude_mask,
	mask_inf=mask_inf,filled=True))
  tables = ltables
 if lkeys is not None and checkcols:
  lkeys = [key for key in checkList(keys) if key in tables[0].colnames and key in tables[1].colnames]
 ftable = join(tables[id_order[0]],tables[id_order[1]],join_type=join_type,keys=lkeys,**kwargs)
 for table in tables[2:]:
  if lkeys is not None and checkcols:
   lkeys = [key for key in checkList(keys) if key in ftable.colnames and key in table.colnames]
  ftable = join(ftable,table,join_type=join_type,keys=lkeys,**kwargs)
 if order is not None and order_before:
  ftable = orderTableColumns(ftable,order,get_list=False,verbose=verbose)
 if dcol is not None:
  renameColumns(ftable,dcol,include=include,exclude=exclude,sep=sep,prefix=prefix)
 if mask or mask_values:
  ftable = maskTableColumns(ftable,float_mask=float_mask,integer_mask=integer_mask,mask_inf=mask_inf,
	string_mask=string_mask,include=include_mask,exclude=exclude_mask,filled=filled)
 if mask_values:
  ftable = maskTableValues(ftable,float_mask=float_mask,integer_mask=integer_mask,mask_inf=mask_inf,
        string_mask=string_masks,include=include_mask,exclude=exclude_mask,filled=False)
 if order is not None and not order_before:
  ftable = orderTableColumns(ftable,order,get_list=False,verbose=verbose)
 return ftable
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def joinToMasterTable(table, dtables, fmt='%%s_%s', join_keys=None, key_col='columns', 
	key_fmt='fmt', key_table='table', key_keys='keys'):
 join_keys = checkList(join_keys)
 if not isinstance(dtables, (list, tuple, dict)):
  dtables = checkList(dtables)
 if isinstance(dtables, (list, tuple)):
  dtables = OrderedDict((fmt % i, t) for (i, t) in enumerate(dtables, 1))
 for fmt in dtables:
  dt = dtables[fmt]
  if not isinstance(dt, dict):
   dt = {key_col: dt.colnames, key_fmt: fmt, key_table: dt}
  if not key_col in dt:
   dt[key_col] = dt[key_table].colnames
  dt[key_col] = checkList(dt[key_col])
  if not key_fmt in dt:
   dt[key_fmt] = fmt
  if not key_keys in dt:
   dt[key_keys] = join_keys
  colnames = [colname for colname in dt[key_col] if colname in dt[key_table].colnames]
  if dt[key_keys] is not None:
   colnames = dt[key_keys] + [colname for colname in colnames if not colname in dt[key_keys]]
  dt[key_table] = dt[key_table][colnames]
  for col in dt[key_col]:
   if dt[key_keys] is not None and col in dt[key_keys]:
    continue
   dt[key_table].rename_column(col, fmt % col)
  if dt[key_keys] is None:
   table = hstack([table, dt[key_table]])
  else:
   table = joinTables((table, dt[key_table]), keys=dt[key_keys])
 return table
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def maskTable(table, float_mask=None, integer_mask=-999, string_mask=None,
        mask_invalid=True, fill_float=np.nan, fill_integer=-999, fill_string='NAN', 
	include=None, exclude=None, filled=False, mask_table=False):
 colnames = checkList(include) if include is not None else table.colnames
 colnames = includeExclude(colnames, exclude)
 float_mask = checkList(float_mask)
 integer_mask = checkList(integer_mask)
 string_mask = checkList(string_mask)
 if mask_table:
  table = Table(table, masked=True)
 for col in colnames:
  # Floats
  if np.issubdtype(table[col].data.dtype, np.floating):
   if mask_invalid:
    table[col].mask[~np.isfinite(table[col].data)] = True
   if float_mask is not None:
    for fmask in float_mask:
     table[col].mask[table[col].data == fmask] = True
   if fill_float is not None:
    table[col].fill_value = fill_float
  # Integers
  if np.issubdtype(table[col].data.dtype, np.integer):
   if integer_mask is not None:
    for imask in integer_mask:
     table[col].mask[table[col].data == imask] = True
   if fill_integer is not None:
    table[col].fill_value = fill_integer
  # Strings
  if np.issubdtype(table[col].data.dtype, str): 
   if string_mask is not None:
    for smask in string_mask:
     table[col].mask[table[col].data == smask] = True
   if fill_string is not None:
    table[col].fill_value = fill_string
 table._meta['MFLOAT'] = float_mask if float_mask is not None and np.isfinite(float_mask) else str(float_mask)
 table._meta['MINT']   = integer_mask if integer_mask is not None else str(integer_mask)
 table._meta['MSTR']   = string_mask if string_mask is not None else str(string_mask)
 table._meta['MVFLOAT'] = fill_float if fill_float is not None and np.isfinite(fill_float) else str(fill_float)
 table._meta['MVINT']   = fill_integer if fill_integer is not None else str(fill_integer)
 table._meta['MVSTR']   = fill_string if fill_string is not None else str(fill_string)
 if filled:
  table = table.filled()
 return table
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def maskTableColumns(table,float_mask=np.nan,integer_mask=-999,string_mask='NAN',
	mask_inf=True,include=None,exclude=None,filled=False):
 colnames = checkList(include) if include is not None else table.colnames
 colnames = includeExclude(colnames,exclude)
 if mask_inf:
  table = Table(table, masked=True)
 for col in colnames:
  if np.issubdtype(table[col].data.dtype, np.floating) and float_mask is not None:
   if mask_inf:
    table[col].mask[~np.isfinite(table[col].data)] = True
   table[col].fill_value = float_mask
  if np.issubdtype(table[col].data.dtype, np.integer) and integer_mask is not None:
   table[col].fill_value = integer_mask
  if np.issubdtype(table[col].data.dtype, str) and string_mask is not None:
   table[col].fill_value = string_mask
 table._meta['MFLOAT'] = float_mask if float_mask is not None and np.isfinite(float_mask) else str(float_mask)
 table._meta['MINT']   = integer_mask
 table._meta['MSTR']   = string_mask
 if filled:
  table = table.filled()
 return table
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def maskTableValues(table,float_mask=np.nan,integer_mask=-999,string_mask=['NAN','--','-'],
	mask_inf=True,include=None,exclude=None,filled=False):
 colnames = checkList(include) if include is not None else table.colnames
 colnames = includeExclude(colnames,exclude)
 float_mask   = checkList(float_mask)
 integer_mask = checkList(integer_mask)
 string_mask  = checkList(string_mask)
 if mask_inf:
  table = Table(table, masked=True)
 for col in colnames:
  if np.issubdtype(table[col].data.dtype, np.floating) and float_mask is not None:
   if mask_inf:
    table[col].mask[~np.isfinite(table[col].data)] = True
   for mvalue in float_mask:
    table[col].mask[table[col].data == mvalue] = True
  if np.issubdtype(table[col].data.dtype, np.integer) and integer_mask is not None:
   for mvalue in integer_mask:
    table[col].mask[table[col].data == mvalue] = True
  if np.issubdtype(table[col].data.dtype, str) and string_mask is not None:
   for mvalue in string_mask:
    table[col].mask[table[col].data == mvalue] = True
 table._meta['MVFLOAT'] = str(float_mask)
 table._meta['MVINT']   = str(integer_mask)
 table._meta['MVSTR']   = str(string_mask)
 if filled:
  table = table.filled()
 return table
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def applyTableMaskColumns(table, mask, columns=None, verbose=True, include=None, exclude=None, exact=False, ignore_case=False, defaults_include=None):
 table = Table(table, masked=True)
 columns = getListDefaultNames(table.colnames, names=columns, exact=exact, exclude=exclude, include=include, ignore_case=ignore_case, defaults_include=defaults_include)
 for column in columns:
  if column in table.columns:
   try:
    table[column].mask[mask] = True
   except:
    if verbose:
     print (' >>> Column "%s" does NOT accept mask' % column)
  else:
   if verbose:
    print (' >>> Column "%s" NOT in Table' % column)
 return table
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def renameColumns(tables,dcol,prefix=True,sep='_',include=None,exclude=None):
 tables = checkList(tables)
 for table in tables:
  dcolumns = dcol
  if isinstance(dcolumns,str):
   columns = checkList(include) if include is not None else table.colnames
   dcolumns = renameList(columns,dcolumns,prefix=prefix,exclude=exclude,sep=sep,get_dict=True)
  for key in dcolumns:
   if key in table.colnames:
    table.rename_column(key,dcolumns[key])
 return table
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def removeColumns(tables,columns):
 columns = checkList(columns)
 tables  = checkList(tables)
 for table in tables:
  ncolumns = [column for column in columns if column in table.colnames]
  table.remove_columns(ncolumns)
 return table
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def addTableColumn(table, name, data, expand=False, axis=0, fmt_rename='%s_orig', overwrite=False):
 if expand:
  data = np.expand_dims(data, axis=axis)
 if not isinstance(data, Column):
  column = Column(name=name, data=data)
 else:
  column = data.copy()
  column.name = name
 if name in table.colnames:
  print ('>>> Column "%s" already exists in table!' % name)
  if overwrite:
   print ('>>> Overwriting existing column "%s"' % name)
   table.remove_column(name)
  else:
   if fmt_rename is not None:
    print ('>>> Renaming column "%s" to %s' % (name, fmt_rename % name))
    table.rename_column(name, fmt_rename % name)
   else:
    return table
 table.add_column(column)
 return table
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def addTableColumns(table, dcolumns, expand=False, axis=0, fmt_rename='%s_orig', overwrite=False):
 for key in dcolumns:
  data = dcolumns[key]
  table = addTableColumn(table, key, data, expand=expand, axis=axis, fmt_rename=fmt_rename, overwrite=overwrite)
 return table
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def orderTableColumns(tables,columns,get_list=False,verbose=True):
 tables  = checkList(tables)
 columns = checkList(columns)
 ntables = []
 for table in tables:
  ncols = [col for col in columns if col in table.colnames] + [col for col in table.colnames if not col in columns]
  if verbose:
   non_cols = [col for col in columns if not col in table.colnames]
   if len(non_cols) > 0:
    print (' >>> Some columns names NOT in table: %s' % ' | '.join(non_cols))
  ntables.append(table[ncols])
 if len(ntables) == 1 and not get_list:
  ntables = ntables[0]
 return ntables
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def replaceTableColumns(table, rtable, keys, tmp='tmp_%s', exclude=None, remove=True):
 table =  table.copy()
 rtable = rtable.copy()
 keys = checkList(keys)
 exclude = checkList(exclude)
 miss_tkeys  = [key for key in keys if not key in table.columns]
 miss_rtkeys = [key for key in keys if not key in rtable.columns]
 if len(miss_tkeys) > 0:
  print ('Missing joinining keys in left table [%s]' % ' | '.join(miss_tkeys))
  return
 if len(miss_rtkeys) > 0:
  print ('Missing joinining keys in right table [%s]' % ' | '.join(miss_rtkeys))
  return
 if exclude is not None:
  for key in exclude:
   rtable.remove_column(key)
 non_tablecols = [key for key in rtable.columns if not key in table.columns]
 if len(non_tablecols) > 0 and remove:
  print ('Removing columns from the right table NOT present in main table [%s]' % ' | '.join(non_tablecols))
  for key in non_tablecols:
   rtable.remove_column(key)
 nokeys =  [col for col in rtable.columns if not col in keys and col in table.columns]
 tnokeys = [tmp % col for col in rtable.columns if not col in keys and col in table.columns]
 if not remove:
  new_cols = [col for col in rtable.columns if not col in keys and not col in table.columns]
  if len(new_cols) > 0:
   print ('Adding new columns from right table to the main table [%s]' % ' | '.join(new_cols))
 for nokey, tnokey in zip(nokeys, tnokeys):
  rtable.rename_column(nokey, tnokey)
 # Empty meta dict so there are no conflict in writting the output: 
 # TypeError: unsupported operand type(s) for +: 'NoneType' and 'str'
 rtable._meta = {}
 nt = join(table, rtable, join_type='left', keys=keys)
 for nokey, tnokey in zip(nokeys, tnokeys):
  mask = ~nt[tnokey].mask
  if not np.any(mask):
   continue
  narray = np.ma.array(nt[nokey].data, mask=nt[nokey].mask)
  narray.data[mask] = nt[tnokey].data[mask].astype(narray.dtype)
  nt.replace_column(nokey, narray)
 for tnokey in tnokeys:
  nt.remove_column(tnokey)
 for key in table.columns:
  nt[key].fill_value = table[key].fill_value
 return nt
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def replaceColumnsValues(table,drename,columns=None,verbose=2):
 if drename is None:
  return table
 columns = checkList(columns)
 if not isinstance(drename[drename.keys()[0]],dict):
  if columns is None:
   if verbose > 0:
    print ('>>> If replace dictionary "drename" does not contain column names {COLUMN: {KEY: REPLACE}}, "columns" should be filled')
   return table
  ndrename = {column: drename for column in columns}
  drename = ndrename
 else:
  if columns is None:
   columns = drename.keys()
 new_columns = []
 for column in columns:
  if column in table.colnames:
   new_columns.append(column)
  else:
   if verbose > 1:
    print ('>>> Key "%s" NOT in table (removed)' % column)
 columns = new_columns
 if len(columns) < 1:
  if verbose > 0:
   print ('>>> None of the columns found in table')
  return table
 for column in columns:
  for key in drename[column]:
   if table[column].data.dtype.type is np.string_ and isinstance(key,str):
    if len(drename[column][key]) > table[column].data.dtype.itemsize:
     narray = np.ma.array(table[column].data,dtype='S%i'%len(drename[column][key]))
     narray[table[column] == key] = drename[column][key]
     table.replace_column(column, narray)
    else:
     table[column][table[column] == key] = drename[column][key]
   else:
    table[column][table[column] == key] = drename[column][key]
 return table
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def patchMaskedTable(tbl,tbr,key,columns=None,both=True,float_mask=np.nan,integer_mask=-999,
	string_mask=['NAN','--','-'],mask_inf=True,include=None,exclude=None,
	filled=False,mask=False,drmask=None,verbose=2):
 tbl = Table(tbl, masked=True)
 tbr = Table(tbr, masked=True)
 if mask:
  default_lmask = {'float_mask':float_mask,'integer_mask':integer_mask,'string_mask':string_mask,'mask_inf':mask_inf,'include':include,'exclude':exclude,'filled':filled}
  default_rmask = {key: value for (key,value) in default_lmask.items()}
  if drmask is not None and isinstance(drmask,dict):
   default_rmask.update(drmask)
  tbl = maskTableValues(tbl,**default_lmask)
  tbr = maskTableValues(tbr,**default_rmask)
 if not key in tbl.columns or not key in tbr.columns:
  if not key in tbl.columns and not key in tbr.columns:
   print ('>>> Merge key "%s" NOT present in any of the tables' % key)
   return
  else:
   table_miss = 'Left' if not key in tbl.colnames else 'Right'
   print ('>>> Merge key "%s" NOT present in %s table' % (key,table_miss))
   return
 if columns is None:
  #columns = list(set(tbl.colnames) | set(tbr.colnames))
  columns = tbr.colnames
 if key in columns:
  columns.remove(key)
 new_columns = []
 for tkey in columns:
  if tkey in tbr.colnames:
   if tkey in tbl.colnames:
    new_columns.append(tkey)
   else:
    if both:
     if verbose > 0:
      print('>>> Key "%s" not present in both tables. Removed from the operation' % tkey)
    else:
     tbl.add_column(Column(name=tkey,data=np.zeros_like(tbl[key].data,dtype=tbr[tkey].dtype)))
     tbl[tkey].mask = True
     new_columns.append(tkey)
     if verbose > 0:
      print('>>> Key "%s" not present in Left table. Added and masked (both=False)' % tkey)
  else:
   if verbose > 0:
    print('>>> Key "%s" not present in Right table. Removed from the operation' % tkey)
 columns = new_columns
 if len(columns) < 1:
  print ('>>> Tables do not share any column in common')
  return
 if np.any(tbl[key].mask):
  print ('>>> Left table has masked values in "%s" key' % key)
  return
 if np.any(tbr[key].mask):
  print ('>>> Right table has masked values in "%s" key' % key)
  return
 # Check key column does not have repeated values
 dtbl = getDuplicates(tbl[key].data)
 if len(dtbl) > 0:
  print ('>>> Left table has duplicates in key "%s": %s' % (key, ' | '.join(dtbl.astype(str))))
  return
 dtbr = getDuplicates(tbr[key].data)
 if len(dtbr) > 0:
  print ('>>> Right table has duplicates in key "%s": %s' % (key, ' | '.join(dtbr.astype(str))))
  return
 # Sort tables according to key to have the same order and assign values at the same position
 tbl.sort(key)
 tbr.sort(key)
 idc = np.in1d(tbl[key].data,tbr[key].data) 
 if not np.any(idc):
  print ('>>> There are no coincident values in "%s" key between the two tables' % key)
  return
 idr = np.in1d(tbr[key].data,tbl[key].data)
 if not np.all(idr):
  len_tbr = len(tbr)
  info = ('%s:  %s' % (key, ' | '.join(tbr[key][~idr].data.astype(str))))
  tbr = tbr[idr]
  if verbose > 0:
   print ('>>> Not all items in Right table are in Left table. Right table reduced from %i to %i items' % (len_tbr,len(tbr)))
  if verbose > 1:
   print ('\n    >>> %s\n' % info)
 patch_cols = []
 for col in columns:
  idm = tbl[col].mask[idc] & ~tbr[col].mask 
  if np.any(idm):
   acol = np.ma.array(tbl[col].data,mask=tbl[col].mask)
   acol_idc = acol[idc]
   acol_idc[idm] = np.ma.array(tbr[col].data[idm],mask=tbr[col].mask[idm])
   acol[idc] = acol_idc
   # When a non masked array is set to a column, the columns has a False masked in all items. Set then a masked array
   tbl[col] = acol
   patch_cols.append(col)
 if len(patch_cols) < 1 and verbose > 0:
  print('>>> No columns have been patched!!')
 return tbl
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def operationTables(table1, table2, func, columns=None, exclude=None, exclude_dtype=None, idx=None, exclude_cols=None):
 exclude = checkList(exclude) if exclude is not None else []
 exclude_cols = checkList(exclude_cols) if exclude_cols is not None else []
 exclude_dtype = checkList(exclude_dtype) if exclude_dtype is not None else []
 exclude_dtype.append(np.string_)

 if columns is None:
  columns = table1.colnames + [col for col in table2.colnames if not col in table1.colnames]

 if idx is not None:
  if idx in table1.colnames and idx in table2.colnames:
   id1 = np.in1d(table1[idx], table2[idx])
   id2 = np.in1d(table2[idx], table1[idx])
   table1 = table1[id1]
   table2 = table2[id2]
  else:
   print ('>>> Selection by column "%s" NOT possible! (Table 1: %s | Table 2: %s)' % (idx, idx in table1.colnames, idx in table2.colnames))

 if len(table1) != len(table2):
  print ('>>> Table 1 (%i) and Table 2 (%i) have DIFFERENT length!!!' % (len(table1), len(table2)))
  return

 new_table = table1.copy()
 for col in columns:
  if col in table1.colnames and col in table2.colnames and not col in exclude and not table1[col].data.dtype.type in exclude_dtype:
   new_table[col] = func(table1[col], table2[col])
  else:
   if not col in exclude_cols:
    new_table[col] = table1[col] if col in table1.colnames else table2[col]
   else:
    if col in new_table.colnames:
     new_table.remove_column(col)

 return new_table
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def maskDistanceApertures(imshape, bin_r, x0, y0, pa=0.0, ba=1.0, rad_scale=1.0, 
	mask=None, filled=True, right=False, both=False, dr=None):
 if isinstance(imshape, np.ndarray):
  imshape = imshape.shape
 if dr is not None and filled:
  bin_r_ap = []
  for rbin in bin_r:
   bin_r_ap.extend([rbin - dr, rbin + dr])
  bin_r = bin_r_ap
  filled = False
 if mask is None:
  mask = np.ones(imshape, dtype=bool)
 r__yx = getImageDistance(imshape, x0, y0, pa, ba) / rad_scale
 if mask.any():
  if filled:
   bin_r_in  = bin_r[:-1]
   bin_r_out = bin_r[1:]
  else:
   bin_r_in  = bin_r[0:-1:2]
   bin_r_out = bin_r[1::2]
  bin_mask = []
  for br_in, br_out in zip(bin_r_in, bin_r_out):
   if both:
    bmask = (r__yx >= br_in) & (r__yx <= br_out) & mask
   else:
    if right:
     bmask = (r__yx > br_in) & (r__yx <= br_out) & mask
    else:
     bmask = (r__yx >= br_in) & (r__yx < br_out) & mask
   bin_mask.append(bmask)
  return np.array(bin_mask)
 else:
  return
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def radialProfile(prop, bin_r, x0, y0, pa=0.0, ba=1.0, rad_scale=1.0,
	mask=None, mode='mean', return_npts=False, reduce_func=None, 
	filled=True, right=False, both=False, return_mask=False, 
	dr=None):

 def red(func, x, fill_value):
  if x.size == 0: return fill_value, fill_value
  if x.ndim == 1: return func(x), len(x)
  return func(x, axis=-1), x.shape[-1]

 imshape = prop.shape[-2:]
 bin_mask = maskDistanceApertures(imshape, bin_r, x0, y0, pa=pa, ba=ba, rad_scale=rad_scale, mask=mask, filled=filled, right=right, both=both, dr=dr)
 nbins = len(bin_mask)
 new_shape = prop.shape[:-2] + (nbins,)

 if mask is None:
  mask = np.ones(imshape, dtype=bool)

 if reduce_func is None:
  if mode == 'mean':
   reduce_func = np.nanmean
  elif mode == 'median':
   reduce_func = np.nanmedian
  elif mode == 'sum':
   reduce_func = np.nansum
  elif mode == 'var':
   reduce_func = np.nanstd
  elif mode == 'std':
   reduce_func = np.nanstd
  else:
   raise ValueError('Invalid mode: %s' % mode)

 if isinstance(prop, np.ma.MaskedArray):
  n_bad = prop.mask.astype('int')
  max_bad = 1.0
  while n_bad.ndim > 2:
   max_bad *= n_bad.shape[0]
   n_bad = n_bad.sum(axis=0)
  mask = mask & (n_bad / max_bad < 0.5)
  prop_profile = np.ma.masked_all(new_shape)
  npts = np.ma.masked_all((nbins,))
  prop_profile.fill_value = prop.fill_value
  reduce_fill_value = np.ma.masked
 else:
  prop_profile = np.empty(new_shape)
  npts = np.empty((nbins,))
  reduce_fill_value = np.nan
 for i, bmask in enumerate(bin_mask):
  prop_profile[..., i], npts[i] = red(reduce_func, prop[..., bmask], reduce_fill_value)

 result = (prop_profile, )
 if return_npts:
  result = result + (npts, )
 if return_mask:
  result = result + (bin_mask, )
 if len(result) == 1:
  result = result[0]
 return result
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def getApertureMask(shape, x0, y0, pa, ba, apertures, method='exact', exclusive=True, 
	central=False, integrate_apertures=False, get_apertures=False, dr=None,
	update_apertures=False, filled=True, mask_negative=True, 
	phot=True):
 from pycasso.photutils import EllipticalAnnulus

 N_y, N_x = shape
 apertures = np.asarray(apertures)

 if dr is not None and filled:
  bin_r_ap = []
  for rbin in apertures:
   bin_r_ap.extend([rbin - dr, rbin + dr])
  apertures = np.asarray(bin_r_ap)
  filled = False

 if mask_negative:
  apertures[apertures < 0.0] = 0.0

 if filled:
  apertures_in = apertures[:-1]
  apertures_out = apertures[1:]
 else:
  apertures_in = apertures[0:-1:2]
  apertures_out = apertures[1::2]

 x_min = -x0
 x_max = N_x - x0
 y_min = -y0
 y_max = N_y - y0
 N_r = len(apertures_in)
 mask = np.zeros((N_r, N_y, N_x))
 area = np.zeros((N_r))

 if phot:
  for i in xrange(apertures_in.size):
   a_in = apertures_in[i] if exclusive else apertures_in[0]
   a_out = apertures_out[i]
   b_out = a_out * ba
   an = EllipticalAnnulus(a_in, a_out, b_out, pa)
   mask[i] = an.encloses(x_min, x_max, y_min, y_max, N_x, N_y, method=method)
   area[i] = an.area()
 else:
  mask = maskDistanceApertures(shape, apertures, x0, y0, pa=pa, ba=ba, filled=filled)
  area = np.array([ap.sum() for ap in mask])

 if central:
  mcentral = np.zeros((N_y, N_x), dtype=np.bool)
  mcentral[y0,x0] = True
  mask = np.insert(mask, 0, mcentral, axis=0)
  area = np.insert(area, 0, 1, axis=0)

 if integrate_apertures and exclusive:
  a_in  = min(apertures)
  a_out = max(apertures)
  b_out = a_out * ba
  if phot:
   an = EllipticalAnnulus(a_in, a_out, b_out, pa)
   mask = np.append(mask, np.expand_dims(an.encloses(x_min, x_max, y_min, y_max, N_x, N_y, method=method), axis=0), axis=0)
   area = np.append(area, an.area())
  else:
   ap = maskDistanceApertures(shape, [a_in, a_out], x0, y0, pa=pa, ba=ba, filled=True)
   mask = np.append(mask, ap, axis=0)
   area = np.append(area, ap[0].sum())

 if get_apertures:
  if update_apertures:
   if integrate_apertures and exclusive:
    apertures = np.append(apertures, a_out)
   if central:
    apertures = np.insert(apertures, 0, 0.0)
  return mask, area, apertures
 else:
  return mask, area
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def getGalApertureMask(K, bin_r=None, rad_scale=None, mask=None, exclusive=True, pa=None, ba=None, x0=None, y0=None, 
	method='center', central=False, integrated=False, integrate_apertures=False, apply_qmask=True, 
	filled=True, dr=None, phot=True):
 qMask = K.qMask if apply_qmask else np.ones(K.qMask.shape, dtype=np.bool)
 if bin_r is not None:
  kpa, kba = K.getEllipseParams()
  pa = kpa if pa is None else pa
  ba = kba if ba is None else ba
  x0 = K.x0 if x0 is None else x0
  y0 = K.y0 if y0 is None else y0
  if rad_scale is None:
   rad_scale = K.HLR_pix
  shape = K.N_y, K.N_x
  ryx, _ = getApertureMask(shape, x0, y0, pa, ba, np.asarray(bin_r) * rad_scale, method=method, exclusive=exclusive, central=central, integrate_apertures=integrate_apertures, filled=filled, phot=phot, dr=dr, get_apertures=False)
  ryx = ryx > 0.0
  ryx[:, ~qMask] = False
  if integrated:
   iryx = qMask > 0.0
   ryx = np.append(ryx, iryx[np.newaxis,...], axis=0)
 else:
  ryx = qMask > 0.0
  ryx = ryx[np.newaxis,...]
 if mask is not None:
  ryx[:, ~mask] = False
 return ryx
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def propApertures(prop, apertures, mode='mean', mask=None, vmask=np.nan, filled=True, area=False):

 def red(func, x, fill_value):
  if x.size == 0: return fill_value
  if x.ndim == 1: return func(x)
  return func(x, axis=-1)

 if isinstance(mode, str):
  if mode == 'mean':
   reduce_func = np.nanmean
  elif mode == 'median':
   reduce_func = np.nanmedian
  elif mode == 'sum':
   reduce_func = np.nansum
  elif mode == 'var':
   reduce_func = np.nanstd
  elif mode == 'std':
   reduce_func = np.nanstd
  else:
   raise ValueError('Invalid mode: %s' % mode)
 else:
  reduce_func = mode

 new_shape = prop.shape[:-2] + (len(apertures),)

 if isinstance(prop, np.ma.MaskedArray):
  reduce_fill_value = np.ma.masked
  ap_prop = np.ma.masked_all(new_shape)
  ap_prop.fill_value = prop.fill_value
 else:
  reduce_fill_value = vmask
  ap_prop = np.empty(new_shape)

 for i, ap in enumerate(apertures):
  ap_mask = np.logical_and(ap, mask) if mask is not None else ap
  ap_prop[..., i] = red(reduce_func, prop[..., ap_mask], reduce_fill_value)
  if area:
   area_norm = float(ap.sum())
   if area_norm > 0:
    ap_prop[..., i] /= area_norm
 
 if isinstance(ap_prop, np.ma.MaskedArray) and filled:
  ap_prop.data[ap_prop.mask] = ap_prop.fill_value

 return ap_prop
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def getApertureSpectra(K, bin_r=None, signal=True, rad_scale=None, mask_residual=True, cid_sum=True,
        residual=True, mask=None, exclusive=True, central=False, integrated=False, integrate_apertures=False, 
	error=False, big=1e3, w1=3650., w2=4600., fnorm_max=5.0, percentage=60., method='percentage', 
	update_bin_r=True, phot=True):
 dmask = {}
 if mask is not None:
  if isinstance(mask, np.ma.MaskedArray):
   mask.data[mask.mask] = False
   mask = mask.data
 if mask_residual:
  mask_res = K.filterResidual(w1=w1, w2=w2, fnorm_max=fnorm_max, percentage=percentage, method=method)
  mask_res = K.zoneToYX(mask_res, extensive=False, surface_density=False)
  mask_res.data[mask_res.mask] = False
  mask_res = mask_res.data
  if mask is not None:
   mask = np.logical_and(mask_res, mask)
  else:
   mask = mask_res
 apertures = getGalApertureMask(K, bin_r, rad_scale=rad_scale, mask=mask, exclusive=exclusive, central=central, integrated=integrated, integrate_apertures=integrate_apertures, phot=phot)
 if error:
  ferr = replaceMaskMedian(K.f_err, big=big)
 if signal:
  fobs = K.zoneToYX(K.f_obs, extensive=True, surface_density=False)
  fsyn = K.zoneToYX(K.f_syn, extensive=True, surface_density=False)
  norm = K.zoneToYX(K.fobs_norm, extensive=True, surface_density=False)
  if error:
   ferr = K.zoneToYX(ferr, extensive=True, surface_density=False)
 else:
  fobs = K.zoneToYX(K.f_obs, extensive=False, surface_density=False) * K.zoneWeight
  fsyn = K.zoneToYX(K.f_syn, extensive=False, surface_density=False) * K.zoneWeight
  norm = K.zoneToYX(K.fobs_norm, extensive=False, surface_density=False) * K.zoneWeight
  if error:
   ferr = K.zoneToYX(ferr, extensive=False, surface_density=False) * K.zoneWeight
 spec_aps = np.zeros((apertures.shape[0], fobs.shape[0]))
 nzones = np.zeros(apertures.shape[0])
 if error:
  spec_err = np.zeros((apertures.shape[0], fobs.shape[0]))
 for i,ap in enumerate(apertures):
  if ap.sum() < 1:
   continue
  nzones[i] = ap.sum()
  if residual:
   if cid_sum:
    spec_ap = ((fobs[:,ap] - fsyn[:,ap]) / norm[ap]).sum(axis=1) / float(ap.sum())
   else:
    spec_ap = (fobs[:,ap] - fsyn[:,ap]).sum(axis=1) / norm[ap].sum()
  else:
   spec_ap = fobs[:,ap].sum(axis=1) / float(ap.sum())
  spec_aps[i] = spec_ap
 if error:
  spec_err[i] = np.sqrt(np.power(ferr[:,ap],2.).sum(axis=1) / float(ap.sum()))

 dmask['spec_aps']   = spec_aps
 dmask['nzones']     = nzones
 dmask['apertures']  = apertures
 dmask['spec_err']   = spec_err if error else None
 dmask['bin_r']      = bin_r
 dmask['orig_bin_r'] = None

 if (central or (integrate_apertures and exclusive) or integrated) and update_bin_r:
  dmask['orig_bin_r'] = dmask['bin_r']
  if central:
   dmask['bin_r'] = np.insert(dmask['bin_r'], 0 , 0.0)
  if integrate_apertures and exclusive:
   dmask['bin_r'] = np.append(dmask['bin_r'], max(dmask['bin_r']))
  if integrated:
   # Nan means here the integration of all non masked pixels
   dmask['bin_r'] = np.append(dmask['bin_r'], np.nan) 

 return dmask
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class getPycassoTable(object):
 def __init__(self, namefits, dprop_key='DICTPROP', props=None, get0D=True, get1D=True, 
		get2D=True, exact=True, exclude='PHO', include=None, ignore_case=True, 
		defaults_include=None, dict_keys=None, dict_exact=True, dict_exclude='PHO', 
		dict_include=None, dict_ignore_case=True, dict_def_include=None, get_dictprop=True, 
		mask=np.nan, masked=True, get_header_keys=False, header_keys=None, header_exact=True, 
		header_exclude='COMMENT', header_include=None, header_def_include=None, dict_meta=None, 
		header_ignore_case=False, convert_to_string=False, dtype=None, key_mdict='MDICT',
		keys_meta_prop=None, default_keys=True, key_metadict='METADICT', default=None, 
		kfile='FILE', dict_prop=None, mag_exact=True, mag_exclude=None, mag_include=None, 
		mag_ignore_case=True, mag_defaults_include=None, radial_prop=None, props_ap=None, 
		dict_ap=None, get_mag_int=True, warning=True, warning_prop=True, **kwargs):

  self.K = fitsDataCube(namefits, **kwargs)
  self.dprop_key = dprop_key
  self.dprop = None
  self.table = Table()

  if get_dictprop:
   self.getDictPropTable(dprop_key=dprop_key, keys=dict_keys, exact=dict_exact, exclude=dict_exclude, include=dict_include, ignore_case=dict_ignore_case, mask=mask, dtype=dtype, default=default, defaults_include=dict_def_include)
  if get_header_keys:
   self.getDictHeaderTable(keys=header_keys, exact=header_exact, exclude=header_exclude, include=header_include, ignore_case=header_ignore_case, mask=mask, dtype=dtype, default=default, defaults_include=header_def_include)
  self.getPycassoProp(props=props, get0D=get0D, get1D=get1D, get2D=get2D, exact=exact, exclude=exclude, include=include, ignore_case=ignore_case, defaults_include=defaults_include, warning=warning, warning_prop=warning_prop)
  if masked:
   self.table = Table(self.table, masked=True)
  self.dictMeta(dict_meta, convert_to_string=convert_to_string)
  self.saveMeta(kfile=kfile, key_mdict=key_mdict, keys_meta_prop=keys_meta_prop, default_keys=default_keys, key_metadict=key_metadict)
  if dict_prop is not None:
   self.getMag(mag_exact=mag_exact, mag_exclude=mag_exclude, mag_include=mag_include, mag_ignore_case=mag_ignore_case, get_mag_int=get_mag_int, **dict_prop)
  if radial_prop is not None:
   self.radialProfileGen(radial_prop)
  if props_ap is not None:
   dict_ap = dict_ap if dict_ap is not None else {}
   self.propApertures(props_ap, **dict_ap)

 def getDictPropTable(self, keys=None, exact=False, exclude=None, include=None, ignore_case=False, dprop_key='DICTPROP', mask=np.nan, dtype=None, default=None, defaults_include=None):
  dkeys, keys = self.getDictKeys(keys)
  if dprop_key in self.K.header:
   self.dprop = eval(self.K.header[dprop_key])
  if self.dprop is None:
   return
  dtype = self.set_dtype(dtype)
  keys = getListDefaultNames(self.dprop.keys(), names=keys, exact=exact, exclude=exclude, include=include, ignore_case=ignore_case, defaults_include=defaults_include)
  for key in keys:
   ktype = self.get_dtype(dtype=dtype, key=key, default=default)
   prop = self.evalValue(self.dprop[key], mask=mask, dtype=ktype)
   if dkeys is not None and key in dkeys:
    key = dkeys[key]
   self.addColumn(key, prop)

 def getDictHeaderTable(self, keys=None, exact=False, exclude=None, include=None, ignore_case=False, mask=np.nan, dtype=None, default=None, defaults_include=None):
  dkeys, keys = self.getDictKeys(keys)
  keys = getListDefaultNames(self.K.header.keys(), names=keys, exact=exact, exclude=exclude, include=include, ignore_case=ignore_case, defaults_include=defaults_include)
  for key in keys:
   ktype = self.get_dtype(dtype=dtype, key=key, default=default)
   prop = self.evalValue(self.K.header[key], mask=mask, dtype=ktype)
   if dkeys is not None and key in dkeys:
    key = dkeys[key]
   self.addColumn(key, prop)

 def set_dtype(self, dtype=None):
  if dtype is None:
   return
  ndtype = {}
  for key in dtype:
   if isinstance(key, str):
    ndtype[key] = dtype[key]
   else:
    names = checkList(dtype[key])
    for name in names:
     ndtype[name] = key
  return ndtype

 def get_dtype(self, dtype=None, key=None, default=None):
  ntype = None
  if dtype is not None:
   if isinstance(dtype, dict):
    if key is None:
     ntype = default
    else:
     if key in dtype:
      ntype = dtype[key]
     else:
      ntype = default
   else:
    ntype = dtype
  return ntype
  
 def evalValue(self, prop, evaluate=True, dict2none=True, mask_None=True, mask=np.nan, to_array=True, dtype=None):
  if evaluate:
   try:
    prop = ast.literal_eval(prop)
   except:
    pass
  if isinstance(prop, (dict)) and dict2none:
   prop = None
  if prop is None and mask_None:
   prop = mask
  if to_array:
   prop = np.atleast_1d(prop)
  if dtype is not None and isinstance(prop, np.ndarray):
   prop = prop.astype(dtype)
  return prop

 def getPycassoProp(self, props=None, get0D=True, get1D=True, get2D=True, exact=False, exclude=None, include=None, ignore_case=False, defaults_include=None, warning=False, warning_prop=False):
  dprops, props = self.getDictKeys(props)
  props = self.getPycassoPropNames(props=props, exact=exact, exclude=exclude, include=include, ignore_case=ignore_case, defaults_include=defaults_include, warning=warning)
  for key in props:
   prop = self.getPycassoAttribute(key)
   if prop is None:
    if not exact and warning_prop:
     print ('WARNING: Property "%s" NOT found in object!' % key)
    continue
   if dprops is not None and key in dprops:
    key = dprops[key]
   if exact:
    try:
     self.addColumn(key, prop)
    except:
     print ('>>> WARNING: Property "%s" could not be added' % key)
   else:
    if prop.ndim == 1:
     if get0D and prop.size == 1:
      self.addColumn(key, prop)
     if get1D and prop.size > 1:
      self.addColumn(key, prop)
    if prop.ndim == 2 and get2D:
     self.addColumn(key, prop)

 def getPycassoPropNames(self, props=None, exact=False, exclude=None, include=None, ignore_case=False, defaults_include=None, warning=False):
  defaults = [item for item in dir(self.K) if not item.startswith('_')]
  defaults = getListDefaultNames(defaults, names=props, exact=exact, exclude=exclude, include=include, ignore_case=ignore_case, defaults_include=defaults_include)
  if warning and props is not None:
   missing_props = np.setdiff1d(props, defaults)
   if missing_props.size > 0:
    print ('WARNING: Properties [%s] NOT found in object!' % (' | '.join(missing_props)))
  nprops = []
  for key in defaults:
   try:
    prop = getattr(self.K, key, None)
   except:
    continue
   if inspect.ismethod(prop) or isinstance(prop, dict):
    continue
   nprops.append(key)
  return nprops

 def getPycassoAttribute(self, key):
  try:
   prop = getattr(self.K, key, None)
  except:
   prop = None
  if prop is not None:
   if inspect.ismethod(prop) or isinstance(prop, dict):
    prop = None
   else:
    if not isinstance(prop, np.ndarray):
     prop = np.atleast_1d(prop)
    # Gives None for float values in python 2.7.10
    #prop = np.ma.atleast_1d(prop)
    else:
     if len(prop.shape) < 1 and prop.size > 0:
      prop = np.atleast_1d(prop)
   if len(prop.shape) < 1:
    print ('>>> WARNING: Property "%s" has NO shape' % key)
    prop = None
  return prop

 def addColumn(self, key, data, axis=0, verbose=True):
  nkey = key
  kid  = 0
  if data.size > 1:
   data = np.ma.expand_dims(data, axis=axis)
  while nkey in self.table.columns:
   kid = getInteger(nkey,sep='_',default=0)
   if kid > 0:
    nkey = '_'.join(nkey.split('_')[:-1])
   nkey = '%s_%i' % (nkey, kid+1)
  if key != nkey:
   self.table._meta[key] = nkey
   if verbose:
    print (' >>> Key "%s" already in table!! Changed from "%s" --> "%s"' % (key, key, nkey))
  self.table.add_column(MaskedColumn(name=nkey, data=data))

 def getMag(self, mag_exact=False, mag_exclude=None, mag_include=None, mag_ignore_case=True, mag_defaults_include=None, get_mag_int=True, **kwargs):
  if kwargs is None or len(kwargs) < 1:
   return
  self.K.getGalProp(**kwargs)
  columns = getListDefaultNames(self.K.gprop.KT.colnames, exact=mag_exact, exclude=mag_exclude, include=mag_include, ignore_case=mag_ignore_case, defaults_include=mag_defaults_include)
  for key in columns:
   self.addColumn(key, np.ma.squeeze(self.K.gprop.KT[key]))
  if 'INTDICT' in self.K.gprop.KT._meta and get_mag_int:
   intdict = eval(self.K.gprop.KT._meta['INTDICT'])
   icolumns = getListDefaultNames(intdict.keys(), exact=mag_exact, exclude=mag_exclude, include=mag_include, ignore_case=mag_ignore_case, defaults_include=mag_defaults_include)
   for key in icolumns:
    self.addColumn(key, np.atleast_1d(intdict[key]))
  for key in ['SUNDICT', 'FILTERS', 'PROP']:
   if key in self.K.gprop.KT._meta:
    self.table._meta[key] = self.K.gprop.KT._meta[key]

 def radialProfileGen(self, radial):
  radial = checkList(radial)
  for prop in radial:
   vprop = getattr(self.K, prop, None)
   if vprop is None and prop in self.K.gprop.KT.columns:
    vprop = self.K.gprop.KT[prop].data[0]
   if vprop is None:
    print ('WARNING: Attribute "%s" of fitsDataCube does NOT exists!' % prop)
    continue
   if vprop.shape != self.K.qSignal.shape:
    print ('WARNING: Attribute "%s" of fitsDataCube has different 2D shape %s than the image %s!' % (prop, vprop.shape, self.K.qSignal.shape))
    continue
   prop__r = self.K.radialProfileGen(vprop)
   name = '%s__r' % prop.replace('__yx', '')
   self.addColumn(name, prop__r)
 
 def propApertures(self, props, apertures=None, **kwargs):
  props = checkList(props)
  if apertures is None:
   apertures = self.K.apertures
  if apertures is None:
   return
  for prop in props:
   vprop = getattr(self.K, prop, None)
   if vprop is None and prop in self.K.gprop.KT.columns:
    vprop = self.K.gprop.KT[prop].data[0]
   if vprop is None:
    print ('WARNING: Attribute "%s" of fitsDataCube does NOT exists!' % prop)
    continue
   if vprop.shape != self.K.qSignal.shape:
    print ('WARNING: Attribute "%s" of fitsDataCube has different 2D shape %s than the image %s!' % (prop, vprop.shape, self.K.qSignal.shape))
    continue
   prop__Ap = propApertures(vprop, apertures, **kwargs)
   name = '%s__Ap' % prop.replace('__yx', '')
   self.addColumn(name, prop__Ap)

 def dictMeta(self, dict_meta, convert_to_string=False):
  if dict_meta is None:
   return
  for key in dict_meta:
   value = dict_meta[key]
   if convert_to_string:
    if isinstance(value, np.ndarray):
     value = value.tolist()
    # Change "'" to allow evaluation for lists later on
    if isinstance(value, list):
     value = str(value).replace("'",'"') 
    else:
     value = str(value)
   self.table._meta[key] = value

 def saveMeta(self, kfile='FILE', key_mdict='MDICT', keys_meta_prop=None, default_keys=True, key_metadict='METADICT'):
  default_keys_meta_prop = ['bin_r', 'bin_r_in', 'bin_r_out', 'bin_r_profile', 'bin_r_ap', 'bin_r_ap_in', 'bin_r_ap_out', 
		'bin_apertures', 'bin_apertures_in', 'bin_apertures_out', 'label_apertures']
  keys_meta_prop = checkList(keys_meta_prop)
  if keys_meta_prop is not None and default_keys:
   keys_meta_prop = default_keys_meta_prop + [key for key in keys_meta_prop if not key in default_keys_meta_prop]
  if keys_meta_prop is None and default_keys:
   keys_meta_prop = default_keys_meta_prop
  if keys_meta_prop is None:
   return
  if kfile is not None:
   self.table._meta[kfile] = self.K.namefits
  if keys_meta_prop is not None and key_mdict is not None:
   dprop = {}
   for prop in keys_meta_prop:
    vprop = getattr(self.K, prop, None)
    if isinstance(vprop, np.ndarray):
     vprop = vprop.tolist()
    dprop[prop] = vprop
   self.table._meta[key_mdict] = dprop
  if key_metadict is not None and self.K.metadict is not None and len(self.K.metadict) > 0:
   self.table._meta[key_metadict] = self.K.metadict
   
 def getDictKeys(self, keys):
  dkeys = None
  if isinstance(keys, dict):
   dkeys = keys
   keys  = dkeys.keys()
  return dkeys, keys
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class getPycassoTables(object):
 def __init__(self, lfits, masked=False, dict_meta=None, convert_to_string=True, kfile='FILE', kfiles='FILES', 
	key_mdict='MDICT', key_metadict='METADICT', ngal='NGAL', verbose=False, dfits=None, dnew=None, 
	fmt='%%s_%s', join_keys=None, key_col='columns', key_fmt='fmt', key_table='table', key_keys='keys', 
	key_args='args', merge_dict2string=True, dict2string=True, dict_columns=None, outfile=None, 
	dict_outfile=None, ifmt=None, ikey=None, idict_file=None, idirout=None, idict_columns=None, 
	ipath=None, multi_outfile=None, multi_key=None, multi_dict_columns=None, multi_dirout=None, 
	fmt_path=None, cpus=None, path='PATH', mask_columns=None, mask_values=None, filled=True, 
	write_verbose=False, join_type='inner', **kwargs):
  self.dtables   = None
  self.table     = None
  self.meta_keys = [key_mdict, key_metadict]

  self.getDictTables(lfits, cpus=cpus, dfits=dfits, dnew=dnew, key_mdict=key_mdict, key_metadict=key_metadict, key_args=key_args, 
	key_col=key_col, key_fmt=key_fmt, key_table=key_table, key_keys=key_keys, kfile=kfile, 
	merge_dict2string=merge_dict2string, verbose=verbose, **kwargs)
  self.writeIndividualTables(ifmt, ikey, dirout=idirout, dict_file=idict_file, dict_columns=idict_columns, path=ipath, 
	mask_columns=mask_columns, mask_values=mask_values, filled=filled, verbose=write_verbose)
  self.mergeTables(dict_meta=dict_meta, convert_to_string=convert_to_string, dict2string=dict2string, masked=masked, 
	kfiles=kfiles, ngal=ngal, mask_values=mask_values, mask_columns=mask_columns, outfile=outfile, path=path,
	dict_outfile=dict_outfile, dict_columns=dict_columns, filled=filled, join_type=join_type)
  self.writeMultiFile(multi_outfile, multi_key, dict_columns=multi_dict_columns, dirout=multi_dirout, mask_values=mask_values, 
	mask_columns=mask_columns, filled=filled, verbose=write_verbose, fmt_path=fmt_path)

 def getTable(self, namefits, dfits=None, dnew=None, key_mdict='MDICT', key_metadict='METADICT', key_args='args', key_col='columns', 
	key_fmt='fmt', key_table='table', key_keys='keys', merge_dict2string=True, kfile='FILE', **kwargs):
  bnamefits = os.path.basename(namefits)
  dfits = {} if dfits is None else dfits
  dkwargs = updateDictDefault(kwargs, updateDictDefault(dfits.get(bnamefits, {}), dict(key_mdict=key_mdict, key_metadict=key_metadict)))
  gal = getPycassoTable(namefits, **dkwargs)
  # Remove kfile entry to avoid MergeConflictWarning Warning
  gal.table._meta.pop(kfile, None)
  table = gal.table
  if isinstance(dnew, dict):
   dmeta = {key_mdict: {}, key_metadict: {}}
   for fmt in dnew:
    new_kwargs = updateDictDefault(dnew[fmt][key_args], dkwargs)
    new_gal = getPycassoTable(namefits, **new_kwargs)
    for mkey in self.meta_keys:
     if mkey in new_gal.table._meta and mkey in dmeta:
      dmeta[mkey][fmt % mkey] = new_gal.table._meta[mkey]
    new_gal.table._meta = OrderedDict()
    dnew[fmt][key_table] = new_gal.table
   table = joinToMasterTable(table, dnew, fmt=fmt, join_keys=join_keys, key_col=key_col, key_fmt=key_fmt, key_table=key_table, key_keys=key_keys)
   for mkey in dmeta:
    if mkey in table._meta:
     table._meta[mkey] = updateNestedDict(table.meta[mkey], dmeta[mkey])
  if merge_dict2string:
   for mkey in table._meta:
    if isinstance(table._meta[mkey], dict):
     table._meta[mkey] = str(table._meta[mkey])
  return table

 def getDictTables(self, lfits, cpus=False, verbose=True, **kwargs):
  lfits = checkList(lfits)
  if lfits is None or len(lfits) < 1:
   return
  self.dtables = OrderedDict()
  cpus = getCPUS(cpus)
  if cpus is not None:
   import multiprocess
   dtables = OrderedDict()
   pool = multiprocess.Pool(processes=cpus)
   for fits in lfits:
    dtables[os.path.basename(fits)] = pool.apply_async(self.getTable, (fits,), kwargs)
   pool.close()
   pool.join()
   for i, fits in enumerate(dtables, 1):
    if verbose:
     print ('>>> %i/%i (Superfits: %s)' % (i, len(lfits), os.path.basename(fits)))
    table = dtables[fits].get()
    if table is not None:
     self.dtables[fits] = table
   #self.dtables = OrderedDict([(fits, result.get()) for fits, result in dtables.items()])
  else:
   for i, fits in enumerate(lfits, 1):
    if verbose:
     print ('>>> %i/%i (Superfits: %s)' % (i, len(lfits), os.path.basename(fits)))
    self.dtables[fits] = self.getTable(fits, **kwargs)
  self.sortDictTables()

 def sortDictTables(self):
  self.dtables = OrderedDict(sorted(self.dtables.items()))

 def maskTable(self, table, mask_values=None, mask_columns=None, copy=True, filled=False):
  if copy:
   table = table.copy()
  if isinstance(mask_values, dict):
   table = maskTableValues(table, **mask_values)
  if isinstance(mask_columns, dict):
   # Astropy looses masked values when written to HDF5, need to fill it before
   table = maskTableColumns(table, **mask_columns)
  # Need too fill with custom values the fill_value attribute for every column in order to save to FITS file
  # Astropy looses masked values when written to HDF5, need to fill it before
  if filled:
   table = table.filled()
  return table

 def writeIndividualTables(self, fmt, key, dirout=None, dict_file=None, dict_columns=None, 
	path=None, mask_values=None, mask_columns=None, filled=True, verbose=False):
  if fmt is None or key is None or len(self.dtables) < 1:
   return
  ltables = self.getListTables(dict_columns=dict_columns)
  if not key in ltables[0].colnames:
   print ('>>> WARNING: Key "%s" NOT present in individual table! Could NOT write individual files!' % key)
   return
  dict_file = {} if dict_file is None else dict_file
  for table in ltables:
   table = self.maskTable(table, mask_values=mask_values, mask_columns=mask_columns, filled=filled)
   name = str(table[key].data[0])
   ifile = joinPath(fmt % name, dirname=dirout)
   ipath = name if path is None else path
   default_dict = dict(overwrite=True, path=ipath, append=True) if fmt.endswith('hdf5') or fmt.endswith('h5') else dict(overwrite=True)
   try:
    if verbose:
     print ('>>> Saving individual table file "%s"' % name)
    table.write(ifile, **updateDictDefault(dict_file, default_dict))
   except:
    print ('>>> WARNING: Could NOT write table to file "%s"' % (ifile))
    print (format_exception(traceback.format_exc()))

 def getListTables(self, dict_columns=None):
  if not (isinstance(self.dtables, dict) and len(self.dtables) > 0):
   return
  if isinstance(dict_columns, dict) and len(dict_columns.keys()) > 0:
   ltables = []
   for table in self.dtables:
    columns = getListDefaultNames(self.dtables[table].colnames, **dict_columns)
    ltables.append(self.dtables[table][columns])
  else:
   ltables = [self.dtables[table] for table in self.dtables]
  return ltables
  
 def mergeTables(self, dict_meta=None, convert_to_string=True, dict2string=True, masked=False, 
	kfiles='FILES', ngal='NGAL', mask_values=None, mask_columns=None, outfile=None, 
	dict_outfile=None, dict_columns=None, path='PATH', filled=True, join_type='inner'):
  if not (isinstance(self.dtables, dict) and len(self.dtables) > 0):
   return
  ltables = self.getListTables(dict_columns=dict_columns)
  table = vstack(ltables, join_type=join_type)
  if masked:
   table = Table(table, masked=True)
  if dict_meta is not None:
   for key in dict_meta:
    value = dict_meta[key]
    if convert_to_string:
     if isinstance(value, np.ndarray):
      value = value.tolist()
     value = str(value)
    table._meta[key] = value
  if kfiles is not None:
   nfiles = self.dtables.keys()
   # Change "'" to allow evaluation for lists later on
   if convert_to_string:
    nfiles = str(nfiles).replace("'",'"')
   table._meta[kfiles] = nfiles
  if self.meta_keys is not None:
   for key in self.meta_keys:
    if key in ltables[-1]._meta and not key in table._meta:
     table._meta[key] = ltables[-1]._meta[key]
     if dict2string and isinstance(table._meta[key], dict):
      table._meta[key] = str(table._meta[key])
  if ngal is not None:
   table._meta[ngal] = len(self.dtables.keys())
  table = self.maskTable(table, mask_values=mask_values, mask_columns=mask_columns, copy=False, filled=filled)
  if outfile is not None:
   dict_outfile = {} if dict_outfile is None else dict_outfile
   default_dict = dict(overwrite=True, path=path) if outfile.lower().endswith('hdf5') or outfile.lower().endswith('h5') else dict(overwrite=True)
   try:
    table.write(outfile, **updateDictDefault(dict_outfile, default_dict))
   except:
    print ('>>> WARNING: Error writing merged table to file "%s"' % (os.path.basename(outfile)))
    print format_exception(traceback.format_exc())
  self.table = table
  return table

 def appendTableHDU(self, name, table, lhdu, force=True, mask_values=None, mask_columns=None, filled=True, verbose=False):
  # filled = True --> Need too fill with custom values the fill_value attribute for every column in order to save to FITS file
  table = self.maskTable(table, mask_values=mask_values, mask_columns=mask_columns, filled=filled)
  name = name.upper()
  names = [hdu.name for hdu in lhdu]
  if name in names and not force:
   print ('WARNING: Fits file "%s" has already an HDU with name "%s" [%s] (use "force" instead)' % (os.path.basename(lhdu.filename()), name, ' | '.join(names)))
   return lhdu
  if name in names and force:
   print ('OVERWRITING: Fits file "%s" has already an HDU with name "%s" [%s]' % (os.path.basename(lhdu.filename()), name, ' | '.join(names)))
  else:
   if verbose:
    print ('>>> Saving Multi FITS table "%s"' % name)
  bhdu = pyfits.BinTableHDU(pyfits.FITS_rec.from_columns(np.array(table.filled())), name=name)
  for key in table._meta:
   mdict = {}
   if isinstance(table._meta[key], dict):
    mdict = {mkey: dict2List(value) for (mkey, value) in table._meta[key].iteritems()}
    bhdu.header[key] = str(mdict)
   else:
    bhdu.header[key] = str(table._meta[key])
  if name in names:
   lhdu[names.index(name)] = bhdu
  else:
   lhdu.append(bhdu)
  return lhdu

 def writeMultiFITS(self, outfile, key, dict_columns=None, force=True, dirout=None, 
	mask_values=None, mask_columns=None, unique_header=False, verbose=False):
  if not (isinstance(self.dtables, dict) and len(self.dtables) > 0) or key is None or outfile is None:
   return
  ltables = self.getListTables(dict_columns=dict_columns)
  if not key in ltables[0].colnames:
   print ('>>> WARNING: Key "%s" NOT present in tables' % key)
   return
  lhdu = [pyfits.PrimaryHDU([])]
  for table in ltables:
   name = table[key].data
   if isinstance(name, np.ndarray) and name.ndim > 0:
    name = name[0]
   lhdu = self.appendTableHDU(str(name), table, lhdu, mask_values=mask_values, mask_columns=mask_columns, verbose=verbose)
  hdulist = pyfits.HDUList(lhdu)
  if unique_header and self.meta_keys is not None:
   for key in self.meta_keys:
    if key in hdulist[1].header:
     hdulist[0].header[key] = hdulist[1].header[key]
     for hdu in hdulist[1:]:
      hdu.header.remove(key)
  outfile = joinPath(outfile, dirname=dirout)
  try:
   print ('>>> Saving Multi Fits file "%s"' % os.path.basename(outfile))
   hdulist.writeto(outfile, clobber=True)
  except:
   print ('>>> WARNING: Error saving Multi Fits file "%s"' % os.path.basename(outfile))
   print format_exception(traceback.format_exc())

 def writeMultiHDF5(self, outfile, key, dict_columns=None, dirout=None, mask_values=None, 
	mask_columns=None, filled=True, verbose=False, fmt_path='%s', overwrite=True):
  if not (isinstance(self.dtables, dict) and len(self.dtables) > 0) or key is None or outfile is None:
   return
  ltables = self.getListTables(dict_columns=dict_columns)
  if not key in ltables[0].colnames:
   print ('>>> WARNING: Key "%s" NOT present in tables' % key)
   return
  outfile = joinPath(outfile, dirname=dirout)
  if os.path.exists(outfile) and overwrite:
   os.remove(outfile)
  fmt_path = '%s' if fmt_path is None else fmt_path
  for table in ltables:
   table = self.maskTable(table, mask_values=mask_values, mask_columns=mask_columns, filled=filled)
   name = table[key].data
   if isinstance(name, np.ndarray) and name.ndim > 0:
    name = name[0]
   try:
    path = fmt_path % name
   except:
    path = name
   if verbose:
    print ('>>> Saving Multi HDF5 table "%s"' % path)
   table.write(outfile, path=path, append=True, overwrite=True)

 def writeMultiFile(self, outfile, key, dict_columns=None, dirout=None, mask_values=None, mask_columns=None, 
	filled=True, verbose=False, unique_header=False, fmt_path=None, overwrite=True):
  if outfile is None or key is None:
   return
  if outfile.endswith('hdf5') or outfile.endswith('h5'):
   self.writeMultiHDF5(outfile, key, dict_columns=dict_columns, dirout=dirout, mask_values=mask_values, mask_columns=mask_columns, 
	filled=filled, verbose=verbose, fmt_path=fmt_path, overwrite=overwrite)
  elif outfile.lower().endswith('fits') or outfile.lower().endswith('fit'):
   self.writeMultiFITS(outfile, key, dict_columns=dict_columns, dirout=dirout, mask_values=mask_values, mask_columns=mask_columns, 
	verbose=verbose, unique_header=unique_header)
  else:
   print ('>>> WARNING: Extension file [%s] NOT understood!' % outfile)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class pycassoTable(object):
 def __init__(self, nfile, cfits=None, ftype=None, hpath=None, only_hdr=True, grid=True, info=None, dgal=None, 
	hb=None, dmask=None, mask_all_cols=False, agelim=(0, 14.2), lagelim=(6., 10.2), ignore_tables=False, 
	dmask_join=None, verbose_mask=False, serr='std_%s', serrn='stdn_%s', error_type='stdn', 
	smask='nmask_%s', shist='hist_%s', smean='mean_%s', smedian='median_%s', slab='%s_%s_label', 
        slabel='label_%s',stg='t%g%%', prop_type='median', ignore_primary=False, dict_gal=None, dlabels=None, 
	read_only=None, read_dprop=False, extra_columns=None, ignore_table=False, drmask=None, 
	min_ngal=3, rmask_columns=None, read_labels=False, desaturate=None, desaturate_tables=None, 
	desaturate_props=None, reset_table=False, tname='TABLE', prefix='orig_', hdu_labels='LABELS', 
	key_tname='TNAME', key_tables='TABLES', key_dtables='DTABLES', key_atables='ATABLES', 
	key_dprops='dprops', key_name='name', key_atname='ATNAME', fmt_atable='AV%s', 
	key_mdict='MDICT', key_metadict='METADICT', meta_keys='METAKEYS', key_obj='DOBJECT', 
	key_tmass='Mcor_tot_fill', key_hubtyp='Nhubtyp', key_ngal='Ngal', dunits=None, 
	tables_units=None, extra_mask_values=None, extra_mask_inf=True, key_path='MDATA', 
	bin_r=None, rtable=None, rtable_keys=None, rtable_exclude=None, 
	rtable_remove=True):
  if cfits is not None:
   self.readDataBaseTable(cfits, only_hdr=only_hdr)
  self.dtables = {}
  self.ltables = []
  self.tables = []
  self.atables = []
  self.dprops = {}
  self._set_info(hb=hb)
  self.grid = grid
  self.info = info
  self.agelim = agelim
  self.lagelim = lagelim
  self.min_ngal = min_ngal
  self.bin_r = bin_r
  self.hdu_labels = hdu_labels.upper()
  self.key_dtables = key_dtables.upper()
  self.key_tname = key_tname.upper()
  self.key_tables = key_tables.upper()
  self.key_atables = key_atables.upper()
  self.key_path = key_path
  self.key_dprops = key_dprops
  self.key_name = key_name
  self.key_atname = key_atname
  self.fmt_atable = fmt_atable
  self.key_tmass = key_tmass
  self.key_hubtyp = key_hubtyp
  self.key_mdict = key_mdict
  self.key_metadict = key_metadict
  self.meta_keys = meta_keys
  self.key_obj = key_obj
  self.key_ngal = key_ngal
  self.stg = stg
  self.t  = None
  self.tl = None
  self.ta = None
  self.ot = None
  self._ot = None
  self.bin_r_in  = None
  self.bin_r_out = None

  self.readTable(nfile, ftype=ftype, hpath=hpath, dgal=dgal, ignore_tables=ignore_tables, ignore_primary=ignore_primary, extra_columns=extra_columns, read_only=read_only, prefix=prefix, read_labels=read_labels, tname=tname, ignore_table=ignore_table, read_dprop=read_dprop, extra_mask_values=extra_mask_values, extra_mask_inf=extra_mask_inf, rtable=rtable, rtable_keys=rtable_keys, rtable_exclude=rtable_exclude, rtable_remove=rtable_remove)
  self.setMasks(dmask=dmask, dmask_join=dmask_join, mask_all_cols=mask_all_cols, drmask=drmask, rmask_columns=rmask_columns, verbose=verbose_mask, reset_table=reset_table)
  self._set_dlabels(dlabels)
  self.set_format_strings(serr, serrn, smask, shist, smean, smedian, slab, slabel)
  self.set_error_type(error_type)
  self.set_prop_type(prop_type)
  self.set_prop_colors(desaturate, tables=desaturate_tables, props=desaturate_props)
  self.update_dprops_units(dunits=dunits, tables=tables_units)

 def readDataBaseTable(self, cfits, only_hdr=False, hub='hubtyp', nhub='Nhubtyp'):
  if only_hdr:
   hdr = pyfits.getheader(cfits, ext=1)
  else:
   c , hdr = pyfits.getdata(cfits, header=True, ext=1)
   self.tc = Table(c)
  self.tch = Table(dict(zip([hub,nhub],zip(*eval(hdr['DTYPE']).items()))))
  self.tch.sort(nhub)
  self.chdr = hdr

 def getNamesTablesHDF5(self, nfile=None):
  nfile = self.nfile if nfile is None else nfile
  import h5py
  f = h5py.File(nfile, 'r')
  names = f.keys()
  f.close()
  return names

 def getAttributeKey(self, key, nfile=None):
  nfile = self.nfile if nfile is None else nfile
  import h5py
  value = None
  f = h5py.File(nfile, 'r')
  if key in f.attrs:
   value = f.attrs[key]
  f.close()
  return value

 def saveAttributeKey(self, key, value, path=None, nfile=None):
  nfile = self.nfile if nfile is None else nfile
  import h5py
  f = h5py.File(nfile, 'r+')
  if path is None:
   f.attrs[key] = value
  else:
   f[path].attrs[key] = value
  f.close()
  return value

 def readFile(self, nfile, ftype=None, hpath=None, rtable=None, rtable_keys=None, rtable_exclude=None, 
	rtable_remove=True, ignore_primary=False, **kwargs):
  self.nfile = nfile
  self.bfile = os.path.basename(nfile)
  self.ftype = ftype
  self.hpath = hpath if hpath is not None else None
  if ftype is None:
   suffix = self.bfile.split('.')[-1]
   if 'fit' in suffix.lower():
    self.ftype = 'fits'
   elif suffix.lower() in ['hdf5', 'h5']:
    self.ftype = 'h5'
  if self.ftype is None:
   return ('>>> WARNING: You need to specify a file type!')
  self.ftype = self.ftype.lower().strip()
  self.lhdu = None
  if self.ftype in 'fits':
   self.lhdu = pyfits.open(nfile)
   self.names = [hdu.name for hdu in self.lhdu[1:]]
   if not ignore_primary:
    self.t = maskTable(Table(self.lhdu[1].data, masked=True), **kwargs)
   self.thdr = self.lhdu[1].header
  if self.ftype in 'h5':
   self.names = self.getNamesTablesHDF5(nfile)
   self.hpath = hpath if hpath is not None else self.getAttributeKey(self.key_path)
   if len(self.names) == 1 and self.hpath is None:
    self.hpath = self.names[0]
    self.saveAttributeKey(self.key_path, self.names[0])
   if not ignore_primary:
    self.t = self.readTableHDF5(self.hpath, nfile=nfile, **kwargs)
    self.thdr = self.getDictAttributes(self.t._meta)
   else:
    self.thdr = getAttributeHDF5(nfile, path=self.hpath)
   if self.t is not None:
    self._ot = self.t.copy()
   if rtable is not None and self.t is not None:
    self.t = replaceTableColumns(self.t, rtable, keys=rtable_keys, exclude=rtable_exclude, remove=rtable_remove)

 def getHeaderKey(self, key):
  value = None
  if key in self.thdr:
   try:
    value = ast.literal_eval(self.thdr[key])
   except:
    value = self.thdr[key]
  else:
   print ('WARNING: Key "%s" NOT found in table header (thdr)' % key)
  return value

 def readTable(self, nfile, ftype=None, hpath=None, dgal=None, ignore_tables=False, ignore_primary=False, 
	extra_columns=None, read_only=None, prefix='orig_%s', read_labels=False, tname='TABLE', 
	fmt_rename=None, ignore_table=False, read_dprop=False, extra_mask_values=None, 
	extra_mask_inf=True, rtable=None, rtable_keys=None, rtable_exclude=None, 
	rtable_remove=True, **kwargs):
  self.readFile(nfile, ftype=ftype, hpath=hpath, ignore_primary=ignore_primary, rtable=rtable, rtable_keys=rtable_keys, exclude=rtable_exclude, rtable_remove=rtable_remove, **kwargs)
  if not ignore_primary:
   if dgal is not None:
    self.t = self.select_by_galaxy(self.t, dgal)
   if self.key_tmass in self.t.colnames:
    self.t = addTableColumn(self.t, 'lmass', np.log10(self.t[self.key_tmass]), fmt_rename=fmt_rename)
   if self.key_hubtyp in self.t.colnames:
    self.t = addTableColumn(self.t, 'hubtypbin', self.get_hbin_array(), fmt_rename=fmt_rename)
   self.t._meta[self.key_tname] = tname
   self.t._meta[self.key_name]  = tname
   if extra_columns is not None:
    self.addExtraColumns(extra_columns, prefix=prefix, mask_values=extra_mask_values, mask_inf=extra_mask_inf)
   self.ot = self.t.copy()
  self.mdict = self.getHeaderKey(self.key_mdict) 
  self.metadict = self.getHeaderKey(self.key_metadict) 
  if self.key_dtables in self.thdr:
   self.ltables = self.getHeaderKey(self.key_dtables) 
  if self.key_tables in self.thdr:
   self.tables = self.getHeaderKey(self.key_tables) 
  if self.key_atables in self.thdr:
   self.atables = self.getHeaderKey(self.key_atables) 
  if self.mdict is not None:
   if 'ageBase' in self.mdict:
    self.ageBase = np.array(self.mdict['ageBase'])
    self.lageBase = np.log10(self.ageBase)
    self.ageGyr = self.ageBase / 1e9
   if 'iageBase' in self.mdict:
    self.iageBase = np.array(self.mdict['iageBase'])
    self.liageBase = np.log10(self.iageBase)
    self.iageGyr = self.iageBase / 1e9
   if 'bin_r' in self.mdict: 
    self.orig_bin_r = np.array(self.mdict['bin_r'])
    if self.bin_r is None:
     self.bin_r = np.array(self.mdict['bin_r'])
   if 'bin_r_in' in self.mdict: 
    self.bin_r_in = np.array(self.mdict['bin_r_in'])
   if 'bin_r_out' in self.mdict: 
    self.bin_r_out = np.array(self.mdict['bin_r_out'])
   if 'label_apertures' in self.mdict:
    self.label_ap = self.mdict['label_apertures']
    self.n_ap = len(self.label_ap)
   self.t_mg = checkList(self.mdict['t_mass_growth']) if 't_mass_growth' in self.mdict else None
   self.label_t_mg = [self.stg % (i*100) for i in self.t_mg] if self.t_mg is not None else None

  # Read extra tables
  if not ignore_tables:
   self.readExtraTables(read_only=read_only, **kwargs)
  # Read labels table
  if read_labels:
   self.readLabelsTable(lhdu)
  if self.lhdu is not None:
   self.lhdu.close()
  # Read masked processed table
  if tname is not None and not ignore_table:
   self.readDataTable(tname, nfile, **kwargs)
  if tname is not None and ignore_table and read_dprop:
   self.readDataTableDprops(tname)

 def select_by_galaxy(self, table, dgal):
  dinclude = {}
  dexclude = {}
  if 'include' in dgal:
   dinclude = dgal['include']
  if 'exclude' in dgal:
   dexclude = dgal['exclude']
  if not 'include' in dgal and not 'exclude' in dgal:
   dinclude = dgal.copy()
  lmask = []
  for key in dinclude: 
   lmask.append(np.in1d(table[key].data, np.atleast_1d(dinclude[key])))
  for key in dexclude: 
   lmask.append(~np.in1d(table[key].data, np.atleast_1d(dexclude[key])))
  if len(lmask) > 1:
   mask = np.logical_and.reduce(lmask)
  else:
   mask = lmask[0]
  return table[mask]

 def setRadialMask(self, drmask=None, columns=None):
  if drmask is not None:
   if columns is None:
    columns = [col for col in self.t.columns if col.endswith('__r')]
   for key in drmask:
    if not key in self.t.columns:
     print ('WARNING: Key "%s" NOT in table' % key)
     continue
    dlim = drmask[key]
    for lim in dlim:
     r_min, r_max = lim
     if r_min is None:
      r_min = self.bin_r.min()
     if r_max is None:
      r_max = self.bin_r.max()
     idr = (self.bin_r >= r_min) & (self.bin_r <= r_max)
     lobj = checkList(dlim[lim])
     for obj in lobj:
      iobj = self.t[key] == obj
      if iobj.sum() > 0:
       for col in columns:
        self.t[col].mask[iobj, idr] = True

 def setMask(self, dmask=None, join=False, mask_all_cols=False, verbose=True):
  ldmask = checkList(dmask)
  if ldmask is not None:
   for dmask in ldmask:
    lmask = []
    mcols = []
    for col in dmask:
     if col in self.t.colnames:
      cmask = []
      cmin, cmax = dmask[col]
      if cmin is not None:
       cmask.append(self.t[col] >= cmin)
      if cmax is not None:
       cmask.append(self.t[col] <= cmax)
      if len(cmask) > 0:
       cmask = np.logical_and.reduce(cmask)
       if join:
        mcols.append(col)
        lmask.append(cmask)
       else:
        self.t = applyTableMaskColumns(self.t, ~cmask, columns=col, verbose=verbose)
     else:
      print ('WARNING: Column "%s" NOT present' % col)
    if len(lmask) > 0 and join:
     mask = np.logical_and.reduce(lmask)
     mcols = None if mask_all_cols else mcols
     self.t = applyTableMaskColumns(self.t, ~mask, columns=mcols, verbose=verbose)

 def setMasks(self, dmask=None, dmask_join=None, mask_all_cols=False, drmask=None, rmask_columns=None, verbose=True, reset_table=True):
  # Example: dmask_join = {'at_flux__r': (6., 10.25), 'McorSD__r': (1., 1e6), 'A_V__r': (None, 20.)}
  if self.ot is not None and reset_table:
   self.t = self.ot.copy()
  if self.t is not None:
   self.setMask(dmask=dmask, join=False, verbose=verbose)
   self.setMask(dmask=dmask_join, join=True, mask_all_cols=mask_all_cols, verbose=verbose)
   self.setRadialMask(drmask=drmask, columns=rmask_columns)

 def averageTable(self, tprop='lmass', tname=None, force=True, **kwargs):
  if self.t is not None:
   if tprop in self.t.columns:
    if self.ta is not None and not force:
     print ('WARNING (Average Table): Average table already created! Use "force=True" for overwriting')
    if self.ta is None or (self.ta is not None and force):
     if self.ta is not None:
      print ('WARNING (Average Table): Average table already created! OVERWRITING!!')
     tname = tname if tname is not None else self.fmt_atable % self.t._meta[self.key_tname]
     dict_atable = updateDictDefault(kwargs, dict(bin_props=1, dtable=False, get_labels=False, save_labels=False, force=True, units=False, add_units=False))
     self.ta = self.table_bin_props(tprop, table_key=tname, **dict_atable)
     self.ta._meta[self.key_name] = tname
    if self.ta is not None:
     self.t._meta[self.key_atname]  = self.ta._meta[self.key_name]
   else:
    print ('>>> WARNING (Average Table): Property "%s" NOT present in table "%s"' % (tprop, self.t._meta[self.key_tname]))

 def set_format_strings(self, serr, serrn, smask, shist, smean, smedian, slab, slabel):
  self.serr = serr
  self.serrn = serrn
  self.smask = smask
  self.shist = shist
  self.smean = smean
  self.smedian = smedian
  self.slab = slab
  self.slabel = slabel
  self.derror_type = {'std': serr, 'stdn': serrn}
  self.dprop_type = {'median': self.smedian, 'mean': self.smean}

 def set_error_type(self, error_type):
  if error_type in self.derror_type:
   self.serror = self.derror_type[error_type]
  else:
   print ('Error type "%s" NOT available! Choosing "stdn" from [%s]' % (error_type, ' | '.join(self.derror_type.keys())))
   self.serror = self.derror_type['stdn']

 def set_prop_type(self, prop_type):
  if prop_type in self.dprop_type:
   self.sprop = self.dprop_type[prop_type]
  else:
   print ('Prop type "%s" NOT available! Choosing "median" from [%s]' % (prop_type, ' | '.join(self.dprop_type.keys())))
   self.sprop = self.dprop_type['median']

 def set_prop_colors(self, desaturate, tables=None, props=None):
  if desaturate is None:
   return
  props = checkList(props)
  if not isinstance(desaturate, dict):
   desaturate = checkList(desaturate)
  for desval in desaturate:
   ltables = tables
   lprops = props
   if isinstance(desaturate, dict) and 'tables' in desaturate[desval]:
    ltables = desaturate[desval]['tables']
   if isinstance(desaturate, dict) and 'props' in desaturate[desval]:
    lprops = checkList(desaturate[desval]['props'])
   if ltables is None:
    ltables = self.dprops.keys()
   if not isinstance(ltables, dict):
    ltables = checkList(ltables)
   for table in ltables:
    if not table in self.dprops:
     print ('WARNING: Table "%s" NOT present in dprops [%s]' % (table, ' | '.join(self.dprops.keys())))
     continue
    if isinstance(ltables, dict):
     lprops = checkList(ltables[table])
    if lprops is None:
     lprops = self.dprops[table].keys()
    for prop in lprops:
     if not prop in self.dprops[table]:
      print ('WARNING: Table "%s" does NOT contain property "%s" [%s]' % (table, prop, ' | '.join(self.dprops[table].keys())))
      continue
     if 'color' in self.dprops[table][prop]:
      self.dprops[table][prop]['color'] = desaturate_colors(self.dprops[table][prop]['color'], desval)

 def _set_info(self, hb=None):
  if hb is None:
   self.hbin = OrderedDict([('E', (0,7)), ('S0', (8,9)), ('Sa', (10,11)), ('Sb', (12,12)), ('Sbc', (13,13)), ('Sc', (14,15)), ('Sd', (16,19))])
  else:
   self.hbin = hb
  self.dhc = {'E':'brown', 'S0':'red', 'Sa':'orange', 'Sb':'green', 'Sbc':'#00D0C9', 'Sc':'#0076C9', 'Sd':'blue', 'ALL': '#1f77b4'}
  self.lcolor = mpl.rcParams['axes.color_cycle']
  self.hnum = len(self.hbin)
  self.lsty = ['-','--','-.',':']
  self.kdprop = 'dprop'
  self.khubtyp = 'hubtyp'
  self.key_table_props = 'props'

 def get_hbin_array(self):
  thb = np.array(['NAN']*len(self.t))
  for key in self.hbin:
   if isinstance(self.hbin[key], (list, tuple)):
    hmin, hmax = self.hbin[key]
   else:
    hmin, hmax = self.hbin[key], self.hbin[key]
   idh = (self.t[self.key_hubtyp] >= hmin) & (self.t[self.key_hubtyp] <= hmax)
   thb[idh] = key
  return thb

 def new_array_selection(self, data, dbin, mask=None, dtype=None):
  if isinstance(data, str):
   data = self.t[data]
  if mask is None:
   dkey = list(dbin.keys())[0]
   if isinstance(dkey, np.str):
    mask = 'NAN'
    dtype = 'S%s' % len(max(list(dbin.keys()) + [mask], key=len))
   elif isinstance(dkey, np.float):
    mask = np.nan
   elif isinstance(dkey, np.int):
    mask = -999
  thb = np.array([mask] * len(self.t), dtype=dtype)
  for key in dbin:
   if isinstance(dbin[key], (list, tuple)):
    dmin, dmax = dbin[key]
   else:
    dmin, dmax = dbin[key], dbin[key]
   idh = (data >= dmin) & (data <= dmax)
   thb[idh] = key
  return thb

 def check_props(self, table, props):
  props = checkList(props)
  if props is None:
   return table.colnames
  nprops = []
  for prop in props:
   if prop in table.colnames:
    nprops.append(prop)
   else:
    print ('WARNING: Column "%s" NOT present in table' % prop)
  return nprops

 def funcExtraColumn(self, name, func, args, prefix='orig_%s', rename_orig_column=None, mask_values=None, mask_inf=True, overwrite=False):
  if self.t is not None:
   args = checkList(args)
   if not all([arg in self.t.colnames for arg in args]):
    bargs = [arg for arg in args if not arg in self.t.colnames]
    print ('>>> Columns "%s" NOT present in table!!' % ' | '.join(bargs))
    return
   args = [self.t[arg] for arg in args]
   if isinstance(func, dict) and len(args) == 1:
    data = self.new_array_selection(args[0], func)
   else:
    data = func(*args)
   self.addExtraColumn(name, data, prefix=prefix, rename_orig_column=rename_orig_column, mask_values=mask_values, mask_inf=mask_inf, overwrite=overwrite)

 def addExtraColumn(self, name, data, prefix='orig_%s', rename_orig_column=None, mask_values=None, mask_inf=True, overwrite=False):
  if name in self.t.colnames:
   if overwrite:
    print ('WARNING: Column "%s" already in table! OVERWRITING! ' % name)
    self.t.remove_column(name)
   else:
    rename_orig_column = prefix % name if rename_orig_column is None else rename_orig_column
    print ('WARNING: Column "%s" already in table! Changing original column to "%s"' % (name, rename_orig_column))
   self.t.rename_column(name, rename_orig_column)
  column = MaskedColumn(name=name, data=data)
  if mask_values is not None:
   for vmask in checkList(mask_values):
    column.mask[column.data == vmask] = True
  if mask_inf and column.dtype.kind == 'f':
   column.mask[~np.isfinite(column.data)] = True
  self.t.add_column(column)

 def addExtraColumns(self, dcols, prefix='orig_%s', mask_values=None, mask_inf=True, overwrite=False):
   for col in dcols:
    if isinstance(col, str):
     key = col
     value = dcols[col]
    elif isinstance(col, (tuple, list)):
     key, value = col
    if isinstance(value, dict):
     func    = value['func']   if 'func'   in value else None
     args    = value['args']   if 'args'   in value else None
     data    = value['data']   if 'data'   in value else None
     kprefix = value['prefix'] if 'prefix' in value else prefix
     rename  = value['rename'] if 'rename' in value else None
     vmask   = value['vmask']  if 'vmask'  in value else mask_values
     imask   = value['imask']  if 'imask'  in value else mask_inf
     if func is not None and args is not None and data is None:
      self.funcExtraColumn(key, func, args, prefix=kprefix, rename_orig_column=rename, mask_values=vmask, mask_inf=imask, overwrite=overwrite)
     else:
      self.addExtraColumn(key, data, prefix=kprefix, rename_orig_column=rename, mask_values=vmask, mask_inf=imask, overwrite=overwrite)
    else:
     self.addExtraColumn(key, value, prefix=prefix, mask_values=mask_values, mask_inf=mask_inf, overwrite=overwrite)

 def addColumnTable(self, table, name, data, prefix='orig_%s', rename_orig_column=None, overwrite=False, **kwargs):
  if not table in self.dtables:
   print ('>>> WARNING: Table "%s" NOT present [%s]' % (table, ' | '.join(self.dtables.keys())))
   return
  if name in self.dtables[table].columns:
   if overwrite:
    print ('WARNING: Column "%s" already in table! OVERWRITING! ' % name)
    self.dtables[table].remove_column(name)
   else:
    rename_orig_column = prefix % name if rename_orig_column is None else rename_orig_column
    print ('WARNING: Column "%s" already in table! Changing original column to "%s"' % (name, rename_orig_column))
    self.dtables[table].rename_column(name, rename_orig_column)
  self.dtables[table].add_column(MaskedColumn(name=name, data=data), **kwargs)

 def addColumnsTable(self, table, columns, data=None, prefix='orig_%s', drename=None, **kwargs):
  if not table in self.dtables:
   print ('>>> WARNING: Table "%s" NOT present [%s]' % (table, ' | '.join(self.dtables.keys())))
   return
  if not isinstance(columns, dict):
   columns = checkList(columns)
   data    = checkList(data)
   if data is None:
    print ('>>> WARNING: You need to provide the either a dictionary OR a list of names ("columns") AND a list of data ("data)')
    return
   columns = OrderedDict((key, val) for (key, val) in zip(columns, data))
  for key in columns:
   rename_orig_column = drename[key] if drename is not None and key in drename else None
   self.addColumnTable(table, key, columns[key], prefix=prefix, rename_orig_column=rename_orig_column, **kwargs) 
    
 def readExtraTablesFits(self, lhdu, close=False, read_only=None, **kwargs):
  if isinstance(lhdu, str):
   lhdu = pyfits.open(lhdu)
   close = True
  read_only = checkList(read_only)
  onames = [name.upper() for name in self.ltables]
  if read_only is not None and len(onames) > 0:
   badnames = [name.upper() for name in read_only if not name.upper() in onames]
   names = [name.upper() for name in read_only if name.upper() in onames]
   if len(badnames) > 0:
    print ('WARNING: Some tables names [%s] were NOT found in [%s]' % (' | '.join(badnames), ' | '.join(onames)))
  else:
   names = onames
  for name in names:
   table = maskTable(Table(lhdu[name].data, masked=True), **kwargs)
   table.name = name.lower().strip()
   # Read META
   if self.meta_keys in lhdu[name].header:
    lmkeys = ast.literal_eval(lhdu[name].header[self.meta_keys])
    dmeta = {key: lhdu[name].header[key.upper()] for key in lmkeys}
    for key, value in dmeta.iteritems():
     try:
       value = value.strip().replace('array','').replace('"','').replace('\\n', ' ')
       value = ast.literal_eval(value)
     except:
      pass
     table._meta[key] = value
   # Read DOBJECT
   if self.key_obj in lhdu[name].header:
    #dobject = ast.literal_eval(lhdu[name].header[self.key_obj]) # literal_eval cannot convert "nan" or "inf"
    dobject = eval(lhdu[name].header[self.key_obj].replace('nan','np.nan'))
    for key in dobject:
     try:
      data = np.array([np.ma.array(item, mask=~np.isfinite(np.array(item))) for item in dobject[key]], dtype=np.object)
     except:
      data = np.array([np.array(item) for item in dobject[key]], dtype=np.object)
     table.add_column(MaskedColumn(name=key, data=data))
   table_name = table._meta[self.key_name] if self.key_name in table._meta else name.lower().strip()
   self.dtables[table_name] = table
  if len(names) == 0 and len(onames) > 0:
   print ('WARNING: NO tables were read')
  if len(onames) == 0:
   print ('INFO: NO External tables available')
  if close:
   lhdu.close()

 def readExtraTablesHDF5(self, nfile, close=False, read_only=None, **kwargs):
  read_only = checkList(read_only)
  onames = [name for name in self.ltables]
  if read_only is not None and len(onames) > 0:
   badnames = [name for name in read_only if not name in onames]
   names = [name for name in read_only if name in onames]
   if len(badnames) > 0:
    print ('WARNING: Some tables names [%s] were NOT found in [%s]' % (' | '.join(badnames), ' | '.join(onames)))
  else:
   names = onames
  for name in names:
   table = self.readTableHDF5(name, nfile=nfile, **kwargs)
   table.name = name.lower().strip()
   table_name = table._meta[self.key_name] if self.key_name in table._meta else name.lower().strip()
   self.dtables[table_name] = table
  if len(names) == 0 and len(onames) > 0:
   print ('WARNING: NO tables were read')
  if len(onames) == 0:
   print ('INFO: NO External tables available')

 def readExtraTables(self, **kwargs):
  if 'fit' in self.ftype:
   self.readExtraTablesFits(self.lhdu, **kwargs)
  if 'h5' in self.ftype:
   self.readExtraTablesHDF5(self.nfile, **kwargs)

 def readLabelsTable(self, lhdu, close=False, **kwargs):
  if isinstance(lhdu, str):
   lhdu = pyfits.open(lhdu)
   close = True
  names =  [hdu.name for hdu in lhdu]
  if not self.hdu_labels in names:
   print ('WARNING: Labels HDU "%s" NOT present! [%s]' %  (self.hdu_labels, ' | '.join(names)))
   return
  self.tl = maskTable(Table(lhdu[self.hdu_labels].data, masked=True), **kwargs)
  if close:
   lhdu.close()

 def appendTableHDU(self, lhdu, table, name, force=False, verbose=True):
  table = table.copy()
  # Need too fill with custom values the fill_value attribute for every column in order to save to FITS file
  table = table.filled()
  name = name.upper()
  names = [hdu.name for hdu in lhdu]
  if name in names and not force:
   if verbose:
    print ('WARNING: Fits file "%s" has already an HDU with name "%s" [%s] (use "force" instead)' % (os.path.basename(lhdu.filename()), name, ' | '.join(names)))
   return lhdu
  if name in names and force:
   if verbose:
    print ('OVERWRITING: Fits file "%s" has already an HDU with name "%s" [%s]' % (os.path.basename(lhdu.filename()), name, ' | '.join(names)))
  else:
   if verbose:
    print ('>>> Saving table "%s"' % name)
  dobject = {}
  for col in table.colnames:
   # Objects type cannot be saved in a FITS binary table
   if table[col].dtype == np.dtype(object):
    dobject[col] = np.array(table[col].data)
    table.remove_column(col)
  bhdu = pyfits.BinTableHDU(pyfits.FITS_rec.from_columns(np.array(table.filled())), name=name)
  # DOBJECT before so all appended header keywords stay at the end
  if len(dobject) > 0:
   try:
    cols = []
    for key in dobject:
     # For variable length arrays "P" format is needed. Only works with 1-D arrays, seems to break with END missing card error
     fmt = 'P%s()' % pyfits.column.NUMPY2FITS[pyfits.column._dtype_to_recformat(dobject[key][0].dtype)[0]]
     cols.append(pyfits.Column(name=key, format=fmt, array=dobject[key]))
    new_cols = pyfits.BinTableHDU.from_columns(cols)
    bhdu = pyfits.BinTableHDU.from_columns(bhdu.columns + new_cols.columns, name=name)
   except:
    bhdu.header[self.key_obj] = dict2List(dobject, to_string=True)
  bhdu.header[self.meta_keys] = str(table._meta.keys())
  for key in table._meta:
   mdict = {}
   if isinstance(table._meta[key], dict):
    mdict = {mkey: dict2List(value) for (mkey, value) in table._meta[key].iteritems()}
    bhdu.header[key] = str(mdict)
   else:
    bhdu.header[key] = str(table._meta[key])
  if name in names:
   lhdu[names.index(name)] = bhdu
  else:
   lhdu.append(bhdu)
  return lhdu

 def appendTableHDF5(self, table, name=None, force=False, nfile=None, verbose=True):
  table = table.copy()
  # Need too fill with custom values the fill_value attribute for every column in order to save to HDF5 file
  table = table.filled()
  if nfile is None:
   nfile = self.nfile
  if name is None:
   name = table._meta[self.key_name] 
  names = self.getNamesTablesHDF5(nfile)
  if name in names and not force:
   if verbose:
    print ('WARNING: HDF5 file "%s" has already a group/dataset with name "%s" [%s] (use "force" instead)' % (os.path.basename(self.bfile), name, ' | '.join(names)))
   return
  if name in names and force:
   if verbose:
    print ('OVERWRITING: HDF5 file "%s" has already a group/dataset with name "%s" [%s]' % (os.path.basename(self.bfile), name, ' | '.join(names)))
  else:
   if verbose:
    print ('>>> Saving table "%s"' % name)
  import h5py
  f = h5py.File(nfile, 'r+')
  if name in f.keys():
   del f[name]
  data = f.create_group(name)
  for col in table.colnames:
   if table[col].dtype == np.dtype(object):
    dt = h5py.special_dtype(vlen=simpleArrayType(table[col][0]))
    data.create_dataset(col, data=np.array(table[col], dtype=np.object), dtype=dt)
   else:
    data.create_dataset(col, data=table[col])
  for key in table._meta:
   value = table._meta[key]
   if isinstance(value, dict):
    value = dict2List(value, to_string=True)
   data.attrs[key] = value if value is not None else 'None'
  f.close()
  
 def saveTablesFits(self, tables=None, force=False, exclude=None, dexclude=None, nfile=None, verbose=True):
  if len(self.dtables) < 1:
   return
  if nfile is None:
   nfile = self.nfile
  if os.path.exists(nfile):
   lhdu = pyfits.open(nfile, 'update')
  else:
   lhdu = [pyfits.PrimaryHDU([])]
  tables = self.dtables if tables is None else checkList(tables) 
  for key in tables:
   if not key in self.dtables:
    print ('WARNING: Table "%s" NOT present in tables dictionary! [%s]' % (key, ' | '.join(self.dtables.keys())))
    continue
   table = self.dtables[key]
   exclude_cols = dexclude[table] if isinstance(dexclude, dict) and table in dexclude else exclude
   if exclude_cols is not None:
    names = getListDefaultNames(table.colnames, exclude=exclude_cols)
    table = table[names]
   if len(table.colnames) > 1000:
    print ('WARNING: Table "%s" has more than 1000 columns (%i)! FITS files have a limit of 999 columns ("TFIELDS card has invalid value ERROR")' % (key, len(table.colnames)))
   lhdu = self.appendTableHDU(lhdu, table, key, force=force, verbose=verbose)
   if not key in self.ltables:
    self.ltables.append(key)
  lhdu[1].header[self.key_dtables] = str(self.ltables)
  if os.path.exists(nfile):
   lhdu.flush()
   lhdu.close()
  else:
   lhdu = pyfits.HDUList(lhdu)
   lhdu.writeto(nfile, clobber=True)

 def saveTablesHDF5(self, tables=None, force=False, exclude=None, dexclude=None, nfile=None, verbose=True):
  if len(self.dtables) < 1:
   return
  if nfile is None:
   nfile = self.nfile
  tables = self.dtables if tables is None else checkList(tables) 
  for key in tables:
   if not key in self.dtables:
    print ('WARNING: Table "%s" NOT present in tables dictionary! [%s]' % (key, ' | '.join(self.dtables.keys())))
    continue
   table = self.dtables[key]
   exclude_cols = dexclude[table] if isinstance(dexclude, dict) and table in dexclude else exclude
   if exclude_cols is not None:
    names = getListDefaultNames(table.colnames, exclude=exclude_cols)
    table = table[names]
   if isinstance(self.ltables, np.ndarray):
    self.ltables = self.ltables.tolist()
   if not key in self.ltables:
    self.ltables.append(key)
   self.saveAttributeKey(self.key_dtables, self.ltables, path=self.hpath)
   self.appendTableHDF5(table, name=key, force=force, nfile=nfile, verbose=verbose)

 def saveTables(self, tables=None, force=False, exclude=None, dexclude=None, nfile=None, verbose=True):
  if 'fit' in self.ftype:
   self.saveTablesFits(tables=tables, force=force, exclude=exclude, dexclude=dexclude, nfile=nfile, verbose=verbose)
  if 'h5' in self.ftype:
   self.saveTablesHDF5(tables=tables, force=force, exclude=exclude, dexclude=dexclude, nfile=nfile, verbose=verbose)

 def addExtraHeaderFits(self, extra_dict, ext='TABLE', nfile=None, verbose=True):
  if extra_dict is None:
   return
  nfile = nfile if nfile is not None else self.nfile
  if os.path.exists(nfile):
   lhdu = pyfits.open(nfile, 'update')
   if self.mdict is not None:
    self.mdict.update(extra_dict)
    mdict = self.mdict
   else:
    mdict = extra_dict
   lname = [hdu.name for hdu in lhdu]
   if isinstance(ext, str):
    ext = ext.upper()
    if ext in lname:
     ext = lname.index(ext)
    else:
     print ('WARNING: Extension "%s" NOT found! Saving in HDU 1 [%s]' % (ext, ' | '.join(lname)))
     ext = 1
   if ext > len(lhdu) - 1:
    print ('WARNING: Extension "%s" out of limits [%s]! Saving in HDU 1' % (ext, len(lhdu)))
    ext = 1
   if verbose:
    print ('>>> Saving Extra Dictionary in header of extension %s [%s]' % (ext, lname[ext]))
   mdict = dict2List(mdict)
   lhdu[ext].header[self.key_mdict] = str(mdict)
   if self.meta_keys in lhdu[ext].header:
    meta_keys = ast.literal_eval(lhdu[ext].header[self.meta_keys])
    meta_keys.append(self.key_mdict)
    lhdu[ext].header[self.meta_keys] = str(meta_keys)
    if self.t is not None and lname[ext] in self.t._meta[self.key_tname]:
     if self.key_mdict in self.t._meta:
      self.t._meta[self.key_mdict].update(mdict)
     else:
      self.t._meta[self.key_mdict] = mdict
    if self.ta is not None and lname[ext] in self.ta._meta[self.key_tname]:
     if self.key_mdict in self.ta._meta:
      self.ta._meta[self.key_mdict].update(mdict)
     else:
      self.ta._meta[self.key_mdict] = mdict
   lhdu.flush()
   lhdu.close()

 def addExtraHeaderHDF5(self, extra_dict, ext='TABLE', nfile=None, verbose=True, **kwargs):
  if extra_dict is None:
   return
  nfile = nfile if nfile is not None else self.nfile
  if os.path.exists(nfile):
   if self.mdict is not None:
    self.mdict.update(extra_dict)
    mdict = self.mdict
   else:
    mdict = extra_dict
   lname = self.getNamesTablesHDF5(nfile)
   if not ext in lname:
    print ('WARNING: Extension "%s" NOT found! Saving in DataSet %s' % (ext, ' | '.join(lname)))
   if verbose:
    print ('>>> Saving Extra Dictionary in header of Group/DataSet %s' % ext)
   table = self.readTableHDF5(ext, **kwargs)
   table._meta[self.key_mdict] = mdict
   if self.t is not None and ext in self.t._meta[self.key_tname]:
    if self.key_mdict in self.t._meta:
     self.t._meta[self.key_mdict].update(mdict)
    else:
     self.t._meta[self.key_mdict] = mdict
   if self.ta is not None and ext in self.ta._meta[self.key_tname]:
    if self.key_mdict in self.ta._meta:
     self.ta._meta[self.key_mdict].update(mdict)
    else:
     self.ta._meta[self.key_mdict] = mdict
   self.appendTableHDF5(table, name=ext, force=True, nfile=nfile, verbose=False)

 def addExtraHeader(self, extra_dict, **kwargs):
  if 'fit' in self.ftype:
   self.addExtraHeaderFits(extra_dict, **kwargs)
  if 'h5' in self.ftype:
   self.addExtraHeaderHDF5(extra_dict, **kwargs)

 def updateLabels(self, dclabels, force=False):
  if self.tl is None:
   self.tl = Table()
  for key in dclabels:
   if key in self.tl.colnames and not force:
    print ('WARNING: Labels HDU "%s" has already a column "%s" (use "force" instead)' % (self.hdu_labels, key))
   if key in self.tl.colnames and force:
    print ('OVERWRITING: Labels HDU "%s" has already a column "%s"' % (self.hdu_labels, key))
    self.tl.remove_column(key)
   else:
    print ('>>> Saving column "%s"' % key)
   self.tl.add_column(MaskedColumn(name=key, data=dclabels[key]))

 def saveLabelsFits(self, nfile=None, force=False, verbose=True):
  if self.tl is None or len(self.tl) == 0:
   return
  nfile = nfile if nfile is not None else self.nfile
  if os.path.exists(nfile):
   lhdu = pyfits.open(nfile, 'update')
  else:
   lhdu = [pyfits.PrimaryHDU([])]
  lhdu = self.appendTableHDU(lhdu, self.tl, self.hdu_labels, force=force, verbose=verbose)
  if os.path.exists(nfile):
   lhdu.flush()
   lhdu.close()
  else:
   lhdu = pyfits.HDUList(lhdu)
   lhdu.writeto(nfile, clobber=True)

 def saveLabelsHDF5(self, nfile=None, force=False):
  if self.tl is None or len(self.tl) == 0:
   return
  self.appendTableHDF5(self.tl, name=self.hdu_labels, force=force, nfile=nfile)

 def saveLabels(self, nfile=None, force=False):
  if 'fit' in self.ftype:
   self.saveLabelsFits(nfile=nfile, force=force)
  if 'h5' in self.ftype:
   self.saveLabelsHDF5(nfile=nfile, force=force)

 def mergeTableLabels(self):
  if self.t is not None:
   if self.tl is not None:
    if len(self.tl) == len(self.t):
     for col in self.tl.colnames:
      if not col in self.t.colnames:
       self.t.add_column(self.tl[col])
      else:
       print ('WARNING: Label column "%s" already present in table!' % col)
    else:
     print ('WARNING: Table and labels have different lengths!')

 def saveDataTableFits(self, force=True, merge=True, exclude=None, nfile=None, verbose=True, **kwargs):
  if self.t is None:
   return
  if merge:
   self.mergeTableLabels()
  nfile = nfile if nfile is not None else self.nfile
  if os.path.exists(nfile):
   lhdu = pyfits.open(nfile, 'update')
  else:
   lhdu = [pyfits.PrimaryHDU([])]
  if self.ta is None:
   self.averageTable(**kwargs)
  self.t._meta[self.key_dprops] = self.dprops
  if exclude is not None:
   names = getListDefaultNames(self.t.colnames, exclude=exclude)
   self.t = self.t[names]
  if len(self.t.colnames) > 1000:
   print ('WARNING: Table "%s" has more than 1000 columns (%i)! FITS files have a limit of 999 columns ("TFIELDS card has invalid value ERROR")' % (self.t._meta[self.key_name], len(self.t.colnames)))
  lhdu = self.appendTableHDU(lhdu, maskTable(self.t), self.t._meta[self.key_tname], force=force, verbose=verbose)
  if not self.t._meta[self.key_tname] in self.tables:
   self.tables.append(self.t._meta[self.key_tname])
  lhdu[1].header[self.key_tables] = str(self.tables)
  if self.ta is not None:
   self.t._meta[self.key_atname] = self.ta._meta[self.key_name]
   if exclude is not None:
    names = getListDefaultNames(self.ta.colnames, exclude=exclude)
    self.ta = self.ta[names]
   if len(self.ta.colnames) > 1000:
    print ('WARNING: Table "%s" has more than 1000 columns (%i)! FITS files have a limit of 999 columns ("TFIELDS card has invalid value ERROR")' % (self.ta._meta[self.key_name], len(self.ta.colnames)))
   lhdu = self.appendTableHDU(lhdu, maskTable(self.ta), self.ta._meta[self.key_name], force=force, verbose=verbose)
   if not self.ta._meta[self.key_name] in self.atables:
    self.atables.append(self.ta._meta[self.key_name])
   lhdu[1].header[self.key_atables] = str(self.atables)
  if os.path.exists(nfile):
   lhdu.flush()
   lhdu.close()
  else:
   lhdu = pyfits.HDUList(lhdu)
   lhdu.writeto(nfile, clobber=True)

 def saveDataTableHDF5(self, force=True, merge=True, exclude=None, nfile=None, hpath=None, verbose=True, **kwargs):
  if self.t is None:
   return
  if merge:
   self.mergeTableLabels()
  nfile = nfile if nfile is not None else self.nfile
  if self.ta is None:
   self.averageTable(**kwargs)
  self.t._meta[self.key_dprops] = self.dprops
  if exclude is not None:
   names = getListDefaultNames(self.t.colnames, exclude=exclude)
   self.t = self.t[names]
  if not self.t._meta[self.key_tname] in self.tables:
   self.tables.append(self.t._meta[self.key_tname])
  if self.ta is not None:
   self.t._meta[self.key_atname] = self.ta._meta[self.key_name]
   if exclude is not None:
    names = getListDefaultNames(self.ta.colnames, exclude=exclude)
    self.ta = self.ta[names]
   if not self.ta._meta[self.key_name] in self.atables:
    self.atables.append(self.ta._meta[self.key_name])
  import h5py
  f = h5py.File(nfile, 'r+')
  hpath = self.hpath if hpath is None else hpath
  f[hpath].attrs[self.key_tables] = self.tables
  if self.ta is not None:
   f[hpath].attrs[self.key_atables] = self.atables
  f.close()
  self.appendTableHDF5(maskTable(self.t), name=self.t._meta[self.key_tname], force=force, nfile=nfile, verbose=verbose)
  if self.ta is not None:
   self.appendTableHDF5(maskTable(self.ta), name=self.ta._meta[self.key_name], force=force, nfile=nfile, verbose=verbose)

 def saveDataTable(self, **kwargs):
  if 'fit' in self.ftype:
   self.saveDataTableFits(**kwargs)
  if 'h5' in self.ftype:
   self.saveDataTableHDF5(**kwargs)

 def getDictAttributes(self, dattr):
  ndict = OrderedDict()
  for key in dattr:
   value = dattr[key]
   try:
    value = value.strip().replace('array','').replace('"','').replace('\\n', ' ')
    value = ast.literal_eval(value)
   except:
    pass
   ndict[key] = value
  return ndict
 
 def updateMetaDictAttributes(self, table, dattr):
  for key in dattr:
   value = dattr[key]
   try:
    value = value.strip().replace('array','').replace('"','').replace('\\n', ' ')
    value = ast.literal_eval(value)
   except:
    pass
   table._meta[key] = value
  return table

 def readTableHDF5(self, tname, nfile=None, **kwargs):
  nfile = self.nfile if nfile is None else nfile
  import h5py
  f = h5py.File(nfile, 'r')
  tnames = f.keys()
  if not tname in tnames:
   print ('>>> WARNING: Table "%s" NOT present in HDF5 file [%s]' % (tname, ' | '.join(tnames)))
   f.close()
   return
  data = f[tname]
  if isinstance(data, h5py.highlevel.Dataset):
   table = maskTable(Table(np.array(data), masked=True), **kwargs)
  elif isinstance(data, h5py.highlevel.Group):
   dtable = OrderedDict()
   for key in data.keys():
    dtable[key] = np.array(data[key])
   table = maskTable(Table(dtable, masked=True), **kwargs)
  table = self.updateMetaDictAttributes(table, data.attrs)
  f.close()
  return table

 def readDataTableFits(self, tname, nfile=None, **kwargs):
  if not tname.upper() in self.tables:
   if len(self.tables) > 0:
    print ('WARNING: Table "%s" NOT in file! [%s]' % (tname, ' | '.join(self.tables)))
   return
  nfile = nfile if nfile is not None else self.nfile
  if os.path.exists(nfile):
   print ('>>> Reading processed Table "%s"' % tname)
   table, hdr = pyfits.getdata(nfile, extname=tname, header=True)
   self.t = maskTable(Table(table, masked=True), **kwargs)
   self.t._meta[self.key_tname] = tname
   if self.key_dprops.upper() in hdr:
    dprops = ast.literal_eval(hdr[self.key_dprops.upper()])
    self.dprops.update(dprops)
    self.t._meta[self.key_dprops] = dprops 
   if self.key_atname in hdr:
    self.readAverageDataTableFits(hdr[self.key_atname], nfile=nfile, **kwargs)
    self.t._meta[self.key_atname] = hdr[self.key_atname]
   if self.meta_keys in hdr:
    metakeys = ast.literal_eval(hdr[self.meta_keys])
    for key in metakeys:
     if not key in self.t._meta and key in hdr:
      try:
       self.t._meta[key] = ast.literal_eval(hdr[key])
      except:
       self.t._meta[key] = hdr[key]

 def readDataTableHDF5(self, tname, nfile=None, **kwargs):
  if not tname.upper() in self.tables:
   if len(self.tables) > 0:
    print ('WARNING: Table "%s" NOT in file! [%s]' % (tname, ' | '.join(self.tables)))
   return
  nfile = nfile if nfile is not None else self.nfile
  if os.path.exists(nfile):
   print ('>>> Reading processed Table "%s"' % tname)
   self.t = self.readTableHDF5(tname, nfile=nfile, **kwargs)
   if self.key_dprops in self.t._meta:
    self.dprops.update(self.t._meta[self.key_dprops])
   if self.key_atname in self.t._meta:
    self.readAverageDataTableHDF5(self.t._meta[self.key_atname], nfile=nfile)

 def readDataTable(self, tname, nfile=None, **kwargs):
  if 'fit' in self.ftype:
   self.readDataTableFits(tname, nfile=nfile, **kwargs)
  elif 'h5' in self.ftype:
   self.readDataTableHDF5(tname, nfile=nfile, **kwargs)

 def readDataTableDpropsFits(self, tname, nfile=None):
  if not tname.upper() in self.tables:
   if len(self.tables) > 0:
    print ('WARNING: Table "%s" NOT in file! [%s]' % (tname, ' | '.join(self.tables)))
   return
  nfile = nfile if nfile is not None else self.nfile
  if os.path.exists(nfile):
   hdr = pyfits.getheader(nfile, extname=tname)
   if self.key_dprops.upper() in hdr:
    dprops = ast.literal_eval(hdr[self.key_dprops.upper()])
    self.dprops.update(dprops)
   else:
    print ('WARNING: "%s" keyword not present in header of table "%s"' % (self.key_dprops, tname))

 def readDataTableDpropsHDF5(self, tname, nfile=None):
  if not tname in self.tables:
   if len(self.tables) > 0:
    print ('WARNING: Table "%s" NOT in file! [%s]' % (tname, ' | '.join(self.tables)))
   return
  nfile = nfile if nfile is not None else self.nfile
  if os.path.exists(nfile):
   import h5py
   f = h5py.File(nfile, 'r')
   mdict = self.getDictAttributes(f[tname].attrs)
   f.close()
   if self.key_dprops in mdict:
    dprops = mdict[self.key_dprops]
    self.dprops.update(dprops)
   else:
    print ('WARNING: "%s" keyword not present in header of table "%s"' % (self.key_dprops, tname))

 def readDataTableDprops(self, tname, nfile=None):
  if 'fit' in self.ftype:
   self.readDataTableDpropsFits(tname, nfile=nfile)
  elif 'h5' in self.ftype:
   self.readDataTableDpropsHDF5(tname, nfile=nfile)

 def readAverageDataTableFits(self, atname, nfile=None, **kwargs):
  if not atname.upper() in self.atables:
   print ('WARNING: Average Table "%s" NOT in file! [%s]' % (atname, ' | '.join(self.atables)))
   return
  nfile = nfile if nfile is not None else self.nfile
  if os.path.exists(nfile):
   print ('>>> Reading processed Average Table "%s"' % atname)
   table, hdr = pyfits.getdata(nfile, extname=atname, header=True)
   self.ta = maskTable(Table(table, masked=True), **kwargs)
   self.ta._meta[self.key_tname] = hdr[self.key_tname]
   self.ta._meta[self.key_name] = hdr[self.key_name]
   self.ta._meta[self.kdprop] = ast.literal_eval(hdr[self.kdprop])

 def readAverageDataTableHDF5(self, atname, nfile=None, **kwargs):
  if not atname.upper() in self.atables:
   print ('WARNING: Average Table "%s" NOT in file! [%s]' % (atname, ' | '.join(self.atables)))
   return
  nfile = nfile if nfile is not None else self.nfile
  if os.path.exists(nfile):
   print ('>>> Reading processed Average Table "%s"' % atname)
   self.ta = self.readTableHDF5(atname, nfile=nfile, **kwargs)

 def readAverageDataTable(self, atname, nfile=None, **kwargs):
  if 'fit' in self.ftype:
   self.readAverageDataTableFits(atname, nfile=nfile, **kwargs)
  elif 'h5' in self.ftype:
   self.readAverageDataTableHDF5(atname, nfile=nfile, **kwargs)

 def getTableProp(self, table, prop, error=False, guess=True, verbose=True, **kwargs):
  if not isinstance(prop, str):
   return prop
  t = self.get_table(table, **updateDictDefault(kwargs, dict(copy=False)))
  if guess:
   sprop = self.getNameProp(prop, error=error)
  else:
   sprop = prop
  if not isinstance(t, str):
   if not sprop in t.colnames:
    if verbose:
     print ('>>> WARNING: Property "%s" NOT in table "%s"' % (prop, table))
    return
   return t[sprop]

 def getTablePropMinMax(self, table, prop, error=True, error_val=0.0, **kwargs):
  vprop  = self.getTableProp(table, prop, **kwargs)
  evprop = error_val
  if error:
   evprop = self.getTableProp(table, prop, error=True, **kwargs)
  if vprop is not None:
   if evprop is None:
    evprop = error_val
   return np.nanmin(vprop - evprop), np.nanmax(vprop + evprop)

 def getTablesPropsMinMax(self, tables, props, error=True, **kwargs):
  tables = checkList(tables)
  props = checkList(props)
  pmin = []
  pmax = []
  for table in tables:
   for prop in props:
    pvalue = self.getTablePropMinMax(table, prop, error=error, **kwargs)
    if pvalue is not None:
     pmin.append(pvalue[0])
     pmax.append(pvalue[1])
  if len(pmin) == 0:
   return
  return np.nanmin(pmin), np.nanmax(pmax)

 def getTablesPropsLim(self, tables, props, error=True, func=None, fac=None, dl=0.05, **kwargs):
  minmax = np.array(self.getTablesPropsMinMax(tables, props, **kwargs))
  minmax = self.setValue(minmax, func=func, fac=fac)
  return self.getlim(minmax[0], minmax[1], dl=dl)

 def getPercentile(self, prop, bprop, table='TABLE', func=None, low=25., high=75., tprint=True, 
	fmt=None, float_fmt='.2f', simple=False, ptable=False, get_ptable=False, vfill=None):
  data = self.getTable(table)
  if table is None:
   return
  bins = self.dprops[bprop][bprop]['label']
  kbprop = self.slab % (bprop, bprop)
  dper = OrderedDict()
  if fmt is None:
   float_fmt = float_fmt if '%s' in float_fmt else '%' + float_fmt
   fmt = '%%s: %%s (%%s) --> %%s IQ (%%s, %%s): %s - %s (%s) [min: %s | max: %s | mean: %s | median: %s]' % ((float_fmt, ) * 7)
   fmt_simple = '%%s: %%s (%%s) --> %%s IQ: %s - %s (%s)' % ((float_fmt, ) * 3)
  for key in bins:
   idx = data[kbprop] == key
   bdata = data[prop][idx]
   if func is not None:
    bdata = func(bdata)
   iq = scipy.stats.iqr(bdata)
   percentile = np.percentile(bdata, [low, high])
   dper[key] = {'prop': prop, 'percentile': percentile, 'iq': iq, 'size': bdata.size, 'name': bprop, 'low_per': low, 'high_per': high, 'min': bdata.min(), 'max': bdata.max(), 'mean': bdata.mean(), 'median': np.ma.median(bdata)}
   if tprint:
    if simple:
     print fmt_simple % (bprop, key, bdata.size, prop, percentile[0], percentile[1], iq)
    else:
     print fmt % (bprop, key, bdata.size, prop, low, high, percentile[0], percentile[1], iq, dper[key]['min'], dper[key]['max'], dper[key]['mean'], dper[key]['median'])
  if ptable or get_ptable:
   import pandas as pd
   df = pd.DataFrame(dper).T
   if vfill is not None:
    df.fillna(vfill, inplace=True)
   if ptable:
    print(df)
   if get_ptable:
    dper = df
  return dper

 def getTable(self, table):
  if isinstance(table, str):
   if table in self.dtables:
    table = self.dtables[table]
   else:
    if table.lower() == self.t._meta[self.key_tname].lower():
     table = self.t
    elif table.lower() == self.t._meta[self.key_atname].lower():
     table = self.ta
    else:
     print ('>>> WARNING: Table "%s" does NOT exists [%s]' % (table, ' | '.join(self.dtables.keys())))
     return
  return table

 def getTableDataSelection(self, table, **kwargs):
  table = self.getTable(table)
  if table is not None:
   return readFitsTable(table, **kwargs)

 def getTableData(self, table, dsel=None, props=None, tprint=True, isort=0, fits=None, max_lines=-1, max_width=-1, mask=None, **kwargs):
  if isinstance(table, str):
   if table in self.dtables:
    table = self.dtables[table]
   else:
    fits = fits if fits is not None else self.fits
    try:
     table = pyfits.getdata(fits, extname=table)
    except:
     print ('Table "%s" NOT present in file "%s"' % (table, fits))
     return
  if dsel is not None:
   lmask = []
   for key in dsel:
    lmask.append(table[key] == dsel[key])
   mask = np.logical_and.reduce(lmask)
  if props is None:
   if mask is not None:
    table = table[mask]
   return table
  else:
   props = checkList(props)
   dt = {}
   for prop in props:
    dt[prop] = np.concatenate(table[prop][mask]) if mask is not None else np.concatenate(table[prop])
   dt = Table(dt)
   dt = dt[props]
   dt.sort(props[isort])
   if tprint:
    dt.pprint(max_lines=max_lines, max_width=max_width, **kwargs)
    print ('\n>>> Rows: %i' % len(dt))
   return dt

 def set_bin_hubtype(self, prop='hubtyp', table_prop='Nhubtyp', hprop='hubtypbin'):
  dprop = {}
  dprop['nbin'] = self.hnum
  dprop['bin_min'] = []
  dprop['bin_max'] = []
  dprop['label'] = []
  dprop['bin'] = []
  dprop['bin_counts'] = []
  dprop['table_prop'] = table_prop
  dprop['orig_prop'] = prop
  dprop['inclusive'] = True
  dprop['unit'] = None
  dprop['indices'] = None
  dprop['func'] = None
  dprop['axis'] = None
  for key in self.hbin:
   dprop['bin'].append(min(self.hbin[key]))
   dprop['bin_min'].append(min(self.hbin[key]))
   dprop['bin_max'].append(max(self.hbin[key]))
   dprop['bin_counts'].append((self.t[hprop] == key).sum())
   dprop['label'].append(key)
  dprop['bin'].append(max(self.hbin[key]))
  dprop['color'] = [self.dhc[key] for key in self.hbin]
  return {prop: dprop}

 def set_bin_string(self, prop, dbin, table_prop=None, kprop=None, dcolor=None):
  table_prop = prop if table_prop is None else table_prop
  kprop = prop if kprop is None else kprop
  dprop = {}
  dprop['nbin'] = len(dbin)
  dprop['bin_min'] = []
  dprop['bin_max'] = []
  dprop['label'] = []
  dprop['bin'] = []
  dprop['bin_counts'] = []
  dprop['table_prop'] = table_prop
  dprop['orig_prop'] = prop
  dprop['inclusive'] = True
  dprop['unit'] = None
  dprop['indices'] = None
  dprop['func'] = None
  dprop['axis'] = None
  for key in dbin:
   if isinstance(dbin, dict):
    dprop['bin'].append(min(dbin[key]))
    dprop['bin_min'].append(min(dbin[key]))
    dprop['bin_max'].append(max(dbin[key]))
   else:
    dprop['bin'].append(key)
    dprop['bin_min'].append(key)
    dprop['bin_max'].append(key)
   dprop['bin_counts'].append((self.t[prop] == key).sum())
   dprop['label'].append(key)
  if isinstance(dbin, dict):
   dprop['bin'].append(max(dbin[key]))
  else:
   dprop['bin'].append(dbin[-1])
  if isinstance(dcolor, dict):
   dprop['color'] = [dcolor[key] for key in dbin]
  else:
   ccolor = cycle(self.lcolor)
   dprop['color'] = [next(ccolor) for i in range(dprop['nbin'])]
  return {kprop: dprop}

 def _set_bin_prop(self, prop, bin_prop=None, bin_hprop='freedman', fmt='%.1f', dfmt='%s', lfmt='%s - %s', 
	kprop=None, table_prop=None, unit=None, add_unit=False, quantiles=True, indices=None, 
	axis=None, func=None, color=None):
  dprop = {}
  table_prop = prop if table_prop is None else table_prop
  kprop = prop if kprop is None else kprop
  dprop[kprop] = {}
  tprop = getArrayIndices(self.t[prop], index=indices, axis=axis, func=func)
  if not isinstance(bin_hprop, str) and quantiles:
   bin_hprop = scipy.stats.mstats.mquantiles(tprop, np.linspace(0., 1., bin_hprop+1))
  # Numpy 1.11 histogram complains when there are NaN values ("range parameter must be finite")
  idf = np.invert(tprop.mask)
  if np.any(tprop.mask):
   print(' >>> WARNING: Property "%s" contains masked values [%i / %i]' % (prop, tprop.mask.sum(), len(self.t)))
  dprop[kprop]['hist'], dprop[kprop]['bin_hist'] = astropy.stats.histogram(tprop[idf], bin_hprop)
  if bin_prop is None:
   dprop[kprop]['bin'] = dprop[kprop]['bin_hist']
   dprop[kprop]['bin_counts'] = dprop[kprop]['hist']
  else:
   if quantiles:
    dprop[kprop]['bin'] = scipy.stats.mstats.mquantiles(tprop, np.linspace(0., 1., bin_prop+1))
    dprop[kprop]['bin_counts'] = astropy.stats.histogram(tprop[idf], dprop[kprop]['bin'])[0]
    dprop[kprop]['bin_counts_tot'] = sum(dprop[kprop]['bin_counts'])
   else:
    dprop[kprop]['bin_counts'], dprop[kprop]['bin'] = astropy.stats.histogram(tprop[idf], bin_prop)
    dprop[kprop]['bin_counts_tot'] = sum(dprop[kprop]['bin_counts'])
  if dprop[kprop]['bin_counts_tot'] != len(tprop):
   print (' >>> WARNING: Total counts of the bins (%s) different from table length (%s) for "%s" [%s]!' % (dprop[kprop]['bin_counts_tot'], len(self.t[prop]), kprop, prop))
  lbin_min = []
  lbin_max = []
  label_prop = []
  dprop[kprop]['nbin'] = len(dprop[kprop]['bin']) - 1
  for i in range(dprop[kprop]['nbin']):
   bin_min = dprop[kprop]['bin'][i]
   bin_max = dprop[kprop]['bin'][i+1]
   lbin_min.append(bin_min)
   lbin_max.append(bin_max)
   try:
    lprop = lfmt % (fmt % bin_min, fmt % bin_max)
   except:
    lprop = lfmt % (sfmt % bin_min, sfmt % bin_max)
   if unit is not None and add_unit:
    lprop = '%s %s' % (lprop, unit)
   label_prop.append(lprop)
  dprop[kprop]['label'] = label_prop
  dprop[kprop]['bin_min'] = lbin_min
  dprop[kprop]['bin_max'] = lbin_max
  dprop[kprop]['table_prop'] = table_prop
  dprop[kprop]['orig_prop'] = prop
  dprop[kprop]['unit'] = unit
  dprop[kprop]['indices'] = indices
  dprop[kprop]['func'] = func
  dprop[kprop]['axis'] = axis
  if isinstance(color, list) and len(color) == len(dprop[kprop]['nbin']):
   dprop[kprop]['color'] = color
  else:
   ccolor = cycle(self.lcolor)
   dprop[kprop]['color'] = [next(ccolor) for i in range(dprop[kprop]['nbin'])]

  return dprop

 def set_bin_prop(self, prop, bin_prop=None, bin_hprop='freedman', fmt='%.1f', dfmt='%s', lfmt='%s - %s', 
	hubtype_key='hubtyp', table_prop=None, kprop=None, unit=None, add_unit=False, quantiles=True, 
	indices=None, axis=None, func=None, color=None, dbin=None):
  hubtype = hubtype_key == prop
  if hubtype:
   dprop = self.set_bin_hubtype(prop)
  else:
   if self.t[prop].dtype.kind == 'S':
    dprop = self.set_bin_string(prop, dbin, table_prop=table_prop, kprop=kprop, dcolor=color)
   else:
    dprop = self._set_bin_prop(prop, bin_prop, bin_hprop=bin_hprop, fmt=fmt, dfmt=dfmt, lfmt=lfmt, table_prop=table_prop, kprop=kprop, unit=unit, add_unit=add_unit, quantiles=quantiles, indices=indices, axis=axis, func=func, color=color)
  return dprop

 def get_object_prop(self, object_prop, key, i, default=None):
  prop = None
  if isinstance(object_prop, dict) and key is not None and key in object_prop:
   prop = object_prop[key]
  if isinstance(object_prop, list):
   prop = object_prop[i] 
  if isinstance(object_prop, (np.float, np.int, np.str, np.bool)):
   prop = object_prop
  if prop is None:
   prop = default
  return prop

 def set_bin_props(self, props, bin_props=None, bin_hprops=None, fmts=None, dfmts=None, lfmts=None, 
	default_bin_hprop='freedman', default_fmt='%.1f', default_dfmt='%s', default_lfmt='%s - %s',
	table_props=None, kprops=None, units=None, add_units=False, quantiles=True, 
	dindex=None, colors=None, dbins=None):
  dprops = {}
  props = checkList(props)
  dindex = dindex if dindex is not None else {}
  for i, prop in enumerate(props):
   dp_index   = dindex.get(prop, {})
   bin_prop   = self.get_object_prop(bin_props, prop, i)
   bin_hprop  = self.get_object_prop(bin_hprops, prop, i, default_bin_hprop)
   fmt        = self.get_object_prop(fmts, prop, i, default_fmt)
   dfmt       = self.get_object_prop(dfmts, prop, i, default_dfmt)
   lfmt       = self.get_object_prop(lfmts, prop, i, default_lfmt)
   table_prop = self.get_object_prop(table_props, prop, i)
   kprop      = self.get_object_prop(kprops, prop, i)
   unit       = self.get_object_prop(units, prop, i)
   add_unit   = self.get_object_prop(add_units, prop, i)
   index      = self.get_object_prop(dp_index.get('indices', None), prop, i)
   axis_prop  = self.get_object_prop(dp_index.get('axis', None), prop, i)
   func       = self.get_object_prop(dp_index.get('func', None), prop, i)
   color      = self.get_object_prop(colors, prop, i)
   dbin       = self.get_object_prop(dbins, prop, i)
   dprop      = self.set_bin_prop(prop, bin_prop=bin_prop, bin_hprop=bin_hprop, fmt=fmt, dfmt=dfmt, lfmt=lfmt, table_prop=table_prop, kprop=kprop, unit=unit, add_unit=add_unit, quantiles=quantiles, indices=index, axis=axis_prop, func=func, color=color, dbin=dbin)
   dprops.update(dprop)
  return dprops

 def table_bin_props(self, props, bin_props=None, bin_hprops='freedman', default_bin_hprop='freedman', 
	fmts='%.1f', lfmts='%s - %s', dfmts='%s', default_fmt='%.1f', default_dfmt='%s', default_lfmt='%s - %s',
	units=None, tprops=None, mask_value=np.nan, good=True, inclusive=False, table_key=None, quantiles=True, 
	dindex=None, table_props=None, kprops=None, add_units=False, force=False, save=True, get_labels=True, 
	save_labels=False, force_labels=True, dtable=True, exclude=None, dexclude=None, colors=None, 
	dbins=None, dlr=None, dlr_dict={}, check_common=False):

  if self.t is None:
   print ('WARNING: No table available! Re-start with "ignore_primary=False"!')
   return

  props = checkList(props)
  if kprops is None and len(props) == 1 and table_key is not None:
   kprops = table_key

  table_key = table_key if table_key is not None else '_'.join(props)
  if table_key in self.dtables and not force and dtable:
   return

  dprop = self.set_bin_props(props, bin_props=bin_props, bin_hprops=bin_hprops, fmts=fmts, lfmts=lfmts, dfmts=dfmts,
		default_fmt=default_fmt, default_dfmt=default_dfmt, default_lfmt=default_lfmt, units=units,
		table_props=table_props, kprops=kprops, add_units=add_units, quantiles=quantiles, 
		dindex=dindex, colors=colors, dbins=dbins)

  # Save prop in props dictionary
  if dtable:
   self.dprops[table_key] = dprop

  # Change props in case there has been a re-definition with kprops
  dprops_keys = {dprop[key]['orig_prop']: key for key in dprop}
  props = [dprops_keys[prop] for prop in props]

  tprops = self.check_props(self.t, tprops)
  irange = product(*[range(dprop[prop]['nbin']) for prop in props]) if len(props) > 1 else range(dprop[props[0]]['nbin'])

  # Add labels table
  if get_labels:
   ns = max([len(dprop[key]['label'][i]) for key in dprop for i in range(len(dprop[key]['label']))]) + 1
   dclabels = OrderedDict()
   for prop in props:
    dclabels[self.slab % (table_key, prop)] = np.zeros(len(self.t), dtype='S%i' % ns)

  lt = []
  for iteritem in irange:
   dt = {}
   iteritem = iteritem if isinstance(iteritem, (tuple, list)) else [iteritem]
   lmask = []
   for i, prop in zip(iteritem, props):
    table_prop = dprop[prop]['table_prop']
    inclusive_prop  = dprop[prop].get('inclusive', inclusive)
    data = getArrayIndices(self.t[table_prop], index=dprop[prop]['indices'], axis=dprop[prop]['axis'], func=dprop[prop]['func'])
    if isinstance(dprop[prop]['bin_min'][i], str):
     idm = (data == dprop[prop]['bin_min'][i]) & (data == dprop[prop]['bin_max'][i])
    else:
     if i != 0 and not inclusive_prop:
      idm = (data > dprop[prop]['bin_min'][i]) & (data <= dprop[prop]['bin_max'][i])
     else:
      idm = (data >= dprop[prop]['bin_min'][i]) & (data <= dprop[prop]['bin_max'][i])
    lmask.append(np.array(idm))
    if get_labels:
     dclabels[self.slab % (table_key, prop)][idm] = dprop[prop]['label'][i]
   mask = np.logical_and.reduce(lmask)
   t = self.t[mask]
   dt[self.key_ngal] = len(t)
   dt['hubtyp'] = np.array(t['hubtypbin'].data)
   hubtyp_unique = np.unique(t['hubtypbin'].data)
   dt['uhubtyp'] = hubtyp_unique if hubtyp_unique.size == 1 else t['hubtypbin'].data
   if 'lmass' in t.colnames:
    dt['lmass'] = t['lmass'].data
    dt['lmass_min'] = t['lmass'].min() if len(t['lmass']) > 0 else np.nan
    dt['lmass_max'] = t['lmass'].max() if len(t['lmass']) > 0 else np.nan
   for i, prop in zip(iteritem, props):
    data = getArrayIndices(t[dprop[prop]['table_prop']], index=dprop[prop]['indices'], axis=dprop[prop]['axis'], func=dprop[prop]['func'])
    try:
     dt[self.slabel % prop] = dprop[prop]['label'][i]
    except:
     dt[self.slabel % prop] = ''
    try:
     dt['%s_max' % prop] = data.max()
    except:
     dt['%s_max' % prop] = dprop[prop]['bin_max'][i]
    try:
     dt['%s_min' % prop] = data.min()
    except:
     dt['%s_min' % prop] = dprop[prop]['bin_min'][i]
   dt['names'] = np.array(t['REALNAME'].data) if 'REALNAME' in t.colnames else np.array(t['DBName'].data)
   dt['cid'] = np.array(t['CALIFAID'].data)
   dt['kid'] = np.array(t['CALIFA_ID'].data)
   for tprop in tprops:
    if not tprop in dt and not t[tprop].dtype.type is np.string_:
     dt[self.smean   % tprop]     = np.ma.mean(t[tprop], axis=0)
     # Error in numpy 1.11 for masked arrays median when the shape is 0 in the computed axis
     dt[self.smedian % tprop]     = np.ma.median(t[tprop], axis=0) if t[tprop].shape[0] > 0 else np.median(t[tprop], axis=0)
     dt[self.serr    % tprop]     = np.ma.std(t[tprop], axis=0)
     if t[tprop].ndim > 1:
      # If the resulting median array does not have masked values, the mask object is not created, it is 
      # only a single boolean, not array, so we need to initilize it
      if not isinstance(dt[self.smedian % tprop].mask, np.ndarray):
       dt[self.smedian % tprop].mask = False
      dt[self.smedian % tprop].data[dt[self.smedian % tprop].mask] = mask_value
      dt[self.smean   % tprop].data[dt[self.smean   % tprop].mask] = mask_value
      dt[self.serr    % tprop].data[dt[self.serr    % tprop].mask] = mask_value
      dt[self.serrn % tprop]       = dt[self.serr % tprop] / np.sqrt((~t[tprop].mask).sum(axis=0))
      dt[self.smask % tprop]       = (~t[tprop].mask).sum(axis=0) if good else t[tprop].mask.sum(axis=0)
     else:
      dt[self.serrn % tprop]       = dt[self.serr % tprop] / np.sqrt(t[tprop].size)
      dt[self.shist % tprop]       = np.array(t[tprop].data)
     if 'norm' in tprop and 'growth' in tprop and '__tr' in tprop and self.t_mg is not None:
      ntprop = tprop.replace('__tr', '__r_tg__average').replace('norm_','')
      dt[self.smean   % ntprop] = interpMaskedArray(dt[self.smean   % tprop].T, self.ageBase, self.t_mg, inverse=True)
      dt[self.smedian % ntprop] = interpMaskedArray(dt[self.smedian % tprop].T, self.ageBase, self.t_mg, inverse=True)
   # Linear regression
   if isinstance(dlr, dict):
    # Convert to array and squeeze, since list of 1 element are converted to 2D arrays of shape (X, 1) when merging list in tables 
    self.LinearRegressionDict(t, dlr, dtable=dt, **updateDictDefault(dlr_dict, dict(flatten=True, to_array=True, array_squeeze=True)))
   lt.append(dt)

  if check_common:
   lt = self.check_list_dict_common_keys(lt)

  # Table made from list of dicts does NOT keep masked values
  # Do NOT forget to fill with custom values the fill_value attribute for every column in order to save to FITS file
  table_props = maskTable(Table(lt, masked=True))
  table_props.name = table_key
  table_props._meta[self.key_name] = table_key
  table_props._meta[self.kdprop] = dprop
  table_props._meta[self.key_table_props] = props
  table_props._meta[self.key_tname] = self.t._meta[self.key_tname]
  if self.key_atname in self.t._meta:
   table_props._meta[self.key_atname] = self.t._meta[self.key_atname]
  if dtable:
   self.dtables[table_key] = table_props
  if get_labels:
   self.updateLabels(dclabels, force=force_labels)
   if save_labels:
    self.saveLabels(force=force_labels)
  if save:
   self.saveTables(tables=table_key, force=force, exclude=exclude, dexclude=dexclude)
  return table_props

 def get_table(self, table, average=False, copy=False, dcolumns=None, overwrite=False):
  if isinstance(table, dict):
   table = self.table_bin_props(**table)
  if isinstance(table, str):
   table_name = table
   table = self.dtables.get(table)
   if table is None:
    if len(self.dtables) > 0:
     sys.exit('WARNING: Table "%s" NOT available [%s]' % (table_name, ' | '.join(self.dtables.keys())))
    else:
     sys.exit('WARNING: Table dictionary is empty')
  if table is not None and isinstance(table, Table) and average:
   self.check_average_table(table)
  if copy:
   table = table.copy()
  if dcolumns is not None:
   self.addFuncErrorColumnsTable(table, dcolumns, overwrite=overwrite)
  return table

 def check_average_table(self, table):
  atname = table._meta[self.key_atname]
  if self.ta is None:
   print ('Empty Average Table Object: reading Average Table "%s"' % atname)
   self.readAverageDataTable(atname)
  if not self.ta._meta[self.key_name] == atname:
   print ('Reading Average Table "%s" (table "%s" in memory)' % (atname, self.ta._meta[self.key_name]))
   self.readAverageDataTable(atname)
  return self.ta

 def check_list_dict_common_keys(self, ldict, verbose=True):
  if len(ldict) < 2:
   return ldict
  dropped = []
  columns = ldict[0].keys()
  for dt in ldict[1:]:
   dropped.extend(set(columns) ^ set(dt.keys()))
   columns = np.intersect1d(columns, dt.keys())
  if len(dropped) > 0:
   if verbose:
    print ('WARNING: Dropping NON common columns! [%s]' % ' | '.join(dropped))
   nldict = []
   for dt in ldict:
    for col in dropped:
     dt.pop(col, None)
    nldict.append(dt)
   ldict = nldict
  return ldict

 def addFuncErrorColumnsTable(self, table, dcolumns, overwrite=False):
  for key in dcolumns:
   self.addFuncErrorColumnTable(table, key, **updateDictDefault(dcolumns[key], {'overwrite': overwrite}))

 def addFuncErrorColumnTable(self, table, key, func=None, args=None, data=None, data_error=None, error=True, overwrite=False):
  if args is not None and func is not None:
   largs = checkList(args)
   args = [self.getNameProp(arg) for arg in largs]
   if not np.all([arg in table.colnames for arg in args]):
    miss_cols = ' | '.join([arg for arg in largs if not self.getNameProp(arg) in table.colnames])
    if self.key_name in table._meta:
     print ('>>> WARNING: Could NOT found columns in table "%s" [%s]' % (table._meta[self.key_name], miss_cols))
    else:
     print ('>>> WARNING: Could NOT found columns in table [%s]' % miss_cols)
    return
   if error:
    eargs = [self.getNameProp(arg, error=True) for arg in largs]
    from uncertainties import unumpy
    args = [unumpy.uarray(np.array(table[arg]), np.array(table[earg])) for (arg, earg) in zip(args, eargs)]
    udata = func(*args)
    data = unumpy.nominal_values(udata)
    data_error = unumpy.std_devs(udata)
   else:
    args = [table[arg] for arg in args]
    data = func(*args)
  if args is None and data is not None and func is not None:
   if data_error is None:
    data = func(data)
   else:
    from uncertainties import unumpy
    udata = func(unumpy.uarray(data, data_error))
    data = unumpy.nominal_values(udata)
    data_error = unumpy.std_devs(udata)
  kdata  = self.getNameProp(key)
  ekdata = self.getNameProp(key, error=True)
  if data is not None:
   if kdata in table.colnames:
    if overwrite:
     print ('>>> WARNING: Column "%s" already in table! OVERWRITING!' % key)
     table.remove_column(kdata)
     table.add_column(MaskedColumn(name=kdata, data=data))
     if data_error is not None:
      if ekdata in table.colnames:
       table.remove_column(ekdata)
      table.add_column(MaskedColumn(name=ekdata, data=data_error))
    else:
     print ('>>> WARNING: Column "%s" already in table! Use overwrite!' % key)
   else:
    table.add_column(MaskedColumn(name=kdata, data=data))
    if data_error is not None:
     table.add_column(MaskedColumn(name=ekdata, data=data_error))
  return table

 def getLaps(self, lidx=None, index=True):
  laps = range(len(self.label_ap))
  if lidx is not None:
   laps = [laps[i] for i in checkList(lidx)]
  if not index:
   laps = [self.label_ap[i] for i in laps]
  return laps

 def getIndexAperture(self, aperture, tol=0.05, single=False, nearest=True, print_ref=True,
	key_bin='bin_apertures', key_meta_ap='set_bin_apertures', key_integrated='integrated',
	key_central='central', sval='aperture', mode='center', default_mode='center'):
  dmode = {'center': key_bin, 'in': '%s_in' % key_bin, 'out': '%s_out' % key_bin}
  mode = mode if mode in dmode else default_mode
  key_bin = dmode.get(mode.lower(), key_bin)
  bin_apertures = np.atleast_1d(self.mdict[key_bin])
  iprint = ['Central: %s  |  Integrated: %s' % (self.metadict[key_meta_ap][key_central], self.metadict[key_meta_ap][key_integrated]), 'Mode "%s" [%s]' % (mode, key_bin)]
  lindex = getIndexValue(bin_apertures, aperture, print_ref=print_ref, sval=sval, iprint=iprint)
  if self.metadict[key_meta_ap][key_central]:
   lindex += 1
  return lindex

 def getIndexRadius(self, radius, key_ap='bin_r_ap', apertures=False, single=False, nearest=True, tol=0.05, 
	key_path='MDATA', print_ref=True, key_ap_in='bin_r_ap_in', key_ap_out='bin_r_ap_out', mode='center'):
  dmode = {'center': key_ap, 'in': key_ap_in, 'out': key_ap_out}
  mode = mode if mode in dmode else default_mode
  key_ap = dmode.get(mode.lower(), key_ap)
  if apertures:
   bin_r = self.mdict[key_ap]
  else:
   bin_r = self.bin_r.copy()
   if mode == 'in':
    bin_r = self.bin_r_in.copy()
   if mode == 'out':
    bin_r = self.bin_r_out.copy()
  iprint = 'Mode "%s" [%s]' % (mode, (key_ap if apertures else 'bin_r'))
  lradius = getIndexValue(bin_r, radius, single=single, nearest=nearest, tol=tol, sval='radius', print_ref=print_ref, iprint=iprint)
  return lradius

 def getIndex_t_growth(self, tg):
  t_growth = self.t_mg
  tg = checkList(tg)
  if tg is None:
   return range(len(t_growth))
  else:
   return [np.searchsorted(t_growth, t) for t in tg]

 def set_plot(self, nx=1, ny=1, dx=5, dy=5, lfs=10, loc=3, labelpad=None, nfig=None,
        force=False, figsize=None, fig=None, grid=None, left=0.07, right=0.98,
        wspace=0.2, bottom=0.15, top=0.92, hspace=0.3, xpad=None, ypad=None, 
	fmt_ngal=None, dfmt_ngal='%i', fmt_lngal=None, dfmt_lngal='%s (%s)'):
  self.dx = dx
  self.dy = dy
  if nfig is not None and not force:
   nx = nfig // ny
   nx += nfig % ny
  self.nx = nx
  self.ny = ny
  self.figsize = (nx*dx, ny*dy) if figsize is None else figsize
  self.lfs = lfs
  self.loc = loc
  self.labelpad = labelpad if labelpad is not None else mpl.rcParams['axes.labelpad']
  self.xpad = self.labelpad if xpad is None else xpad
  self.ypad = self.labelpad if ypad is None else ypad
  self.fmt_ngal  = fmt_ngal  if fmt_ngal  is not None else dfmt_ngal
  self.fmt_lngal = fmt_lngal if fmt_lngal is not None else dfmt_lngal
  if fig is None:
   fig = mpl.pyplot.figure(figsize=self.figsize)
  if grid is None:
   gs = gridspec.GridSpec(self.ny, self.nx)
   gs.update(left=left, right=right, wspace=wspace, bottom=bottom, top=top, hspace=hspace)
  else:
   gs = gridspec.GridSpecFromSubplotSpec(self.ny, self.nx, subplot_spec=grid, wspace=wspace, hspace=hspace)
  return fig, gs

 def _set_dlabels(self, dlabels=None, lsuf=None):
  self.dlabels = {'at_flux': r'$\langle$log age$\rangle_{L}$ [yr]', 'at_mass': r'$\langle$log age$\rangle_{M}$ [yr]',
		'McorSD': r'log $\Sigma_{*}$ [M$_{\odot}$ pc$^{-2}$]', 'alogZ_flux': r'$\langle$logZ$_{*}\rangle_{L}$',
		'alogZ_mass': r'$\langle$logZ$_{*}\rangle_{M}$', 'A_V': r'A$_V$ [mag]', 
		'Mcor_growth__tg': r'Lookback Time [Gyr]', 'Mcor_growth__Ap_tg': r'Lookback Time [Gyr]'}
  lsuf = checkList(lsuf)
  if lsuf is not None:
   keys = self.dlabels.keys()
   for key in keys:
    for suf in lsuf:
     self.dlabels['%s%s' % (key, suf)] = self.dlabels[key]
  if dlabels is not None:
   self.dlabels.update(dlabels)

 def getLabel(self, prop, dlabel=None, guess=True, idl=0):
  if prop is None:
   return
  label = None if dlabel is None else dlabel
  if dlabel is None:
   lkeys = [key for key in self.dlabels if key in prop]
   if len(lkeys) > 0:
    label = self.dlabels[lkeys[idl]]
    if len(lkeys) > 1:
     print ('LABEL INFO: Chosen key "%s" from [%s]' % (lkeys[idl], ' | '.join(lkeys)))
  if guess and label is None:
   if 'dt' in prop:
    label = r'$\dot{M}$  [M$_{\odot}$ / yr]'
   if '_growth' in prop:
    label = r'Cumulative mass fraction'
  if label is None:
   label = prop
  return label

 def getLabelDict(self, prop, labels=None):
  label = None
  if labels is not None:
   if isinstance(labels, dict):
    if prop in labels:
     label = labels[prop]
   if isinstance(labels, str):
     label = labels
  if label is None:
   label = self.getLabel(prop)
  return label

 def setLabelUnitNgal(self, label, unit=None, ngal=None, add_unit=False, add_ngal=False, check_unit=True, dprop=None, prop=None):
  if dprop is not None and prop is not None:
   if unit is None:
    unit = dprop[prop]['unit']
  if label is not None and unit is not None and add_unit:
   if not check_unit or check_unit and not unit in label:
    label = '%s %s' % (label, unit)
  if label is not None and ngal is not None and add_ngal:
   label = self.fmt_lngal % (label, ngal)
  return label

 def fig_labels(self, fig, prop=None, sxlabel=None, sylabel=None, grid=None, xlabel=True, ylabel=True, 
	figname=None, info=None, pprop=True, xfs=22, yfs=18, ha='center', style='normal', va='center', 
	xinfo=0.99, yinfo=0.05, fsinfo=10, pinfo=True, xprop=0.07, yprop=None, fsprop=10, pt=72., 
	xpad=None, ypad=None, xlabel_x=0.5, xlabel_y=0.03, ylabel_x=0.02, dylabel={}, 
	dxlabel={}, dinfo={}, ylabel_y=0.5, ldtext=[], ldarrow=[]):
  info = self.info if info is None else info
  xpad = self.xpad if xpad is None else xpad
  ypad = self.ypad if ypad is None else ypad
  sylabel = self.getLabel(prop) if sylabel is None else sylabel
  if sylabel is not None:
   sylabel = mplText2latex(sylabel, single=True)
  if sxlabel is not None:
   sxlabel = mplText2latex(sxlabel, single=True)
  if grid is not None:
   x0, x1, y0, y1 = get_axe_position(grid, fig)
   xoffset = xpad / pt
   yoffset = ypad / pt
   if xlabel and sxlabel is not None:
    fig.text((x1 + x0)/2.0, y0 - xoffset, sxlabel, **updateDictDefault(dxlabel, dict(fontsize=xfs, ha=ha, style=style)))
   if ylabel and sylabel is not None:
    fig.text(x0 - yoffset, (y1 + y0)/2.0, sylabel, **updateDictDefault(dylabel, dict(rotation=90, fontsize=yfs, style=style, va=va, ha=ha)))
  else:
   if xlabel and sxlabel is not None:
    fig.text(xlabel_x, xlabel_y, sxlabel, fontsize=xfs, ha=ha, style=style, **dxlabel)
   if ylabel and sylabel is not None:
    fig.text(ylabel_x, ylabel_y, sylabel, **updateDictDefault(dylabel, dict(rotation=90, fontsize=yfs, style=style, ha=ha, va=va)))
  yinfo = yinfo if grid is None else y0 - xoffset
  if info is not None and pinfo:
   xinfo = xinfo if grid is None else x1
   fig.text(xinfo, yinfo, info, **updateDictDefault(dinfo, dict(va='center', ha='right', fontsize=fsinfo)))
  if pprop and prop is not None:
   yprop = yinfo if yprop is None else yprop
   xprop = xprop if grid is None else x0
   fig.text(xprop, yprop, prop, va='center', ha='left', fontsize=fsprop)
  if ldtext is not None and len(ldtext) > 0:
   for dtext in ldtext:
    fig.text(**dtext)
  if ldarrow is not None and len(ldarrow) > 0:
   for darrow in ldarrow:
    # Need to avoid deep copy since it complains copying "handles" (TransfromNode instances can not be copied. Consider using frozen() instead)
    ndarrow = updateDictDefault(darrow, dict(fig=fig, ax=grid), deep=False)
    addArrowsLabel(**ndarrow)
  if figname is not None:
   fig.savefig(figname)
  return fig

 def setTicksLim(self, ax, **kwargs):
  defaults = dict(xticklabels=True, yticklabels=True, left_idx=None, right_idx=None, bottom_idx=None, 
       top_idx=None, ylim=None, xlim=None, grid=None, legend=False, loc=2, ncol=1, legendfs=None, saxtitle=None, 
       axtitle=True, sxlabel=None, xlabel=True, sylabel=None, ylabel=True, ylabelp='right', artist=False, 
       framealpha=1.0, xticks=None, yticks=None, dxticklabel={}, dyticklabel={}, xhide_id=None, yhide_id=None, 
       dtitle={}, dxlabel={}, dylabel={}, xhide=True, yhide=True, astitle=True, xtit=0.95, ytit=0.92, 
       hatit='right', leg_prop={}, sxticklabels=None, syticklabels=None, fig=None, xgrid=True, 
       ygrid=True, legend_invisible=False, axisbelow=True, xminorticks=False, 
       yminorticks=False, minorticks=False, ldtext=None, 
       xticks_major=True, yticks_major=True)
  kwargs = updateDictDefault(kwargs, defaults)
  if kwargs['grid'] is None:
   kwargs['grid'] = self.grid
  if kwargs['legendfs'] is None:
   kwargs['legendfs'] = self.lfs
  return setTicksLim(ax, **kwargs)

 def getlim(self, lmin, lmax, dl=0.05):
  if not isinstance(lmin, (np.float, np.int)) and not isinstance(lmax, (np.float, np.int)):
   return None
  seg = lmax - lmin
  return lmin - seg*dl, lmax + seg*dl

 def getXYlim(self, xprop, yprop, xlim=None, ylim=None, nprop=None, n_min=None, check=True, 
	xfunc=None, yfunc=None, xfac=None, yfac=None, dtplot={}):
  def getlim(value):
   # This function is needed in case unumpy is used, since np.ma gives error on uncertainties objects: 
   # AttributeError: 'float' object has no attribute 'view'
   try:
    vlim = self.getlim(np.ma.min(value), np.ma.max(value))
   except:
    from uncertainties import unumpy
    value = unumpy.nominal_values(value)
    vlim = self.getlim(np.ma.min(value), np.ma.max(value))
   return vlim
  if nprop is not None and n_min is not None and check and (xlim is not None or ylim is not None):
   if yprop.ndim > 1 and nprop.size == yprop.shape[0] and ylim is None:
    yprop = yprop[nprop >= n_min, ...]
   if xprop.ndim > 1 and nprop.size == xprop.shape[0] and xlim is None:
    xprop = xprop[nprop >= n_min, ...]
  if ylim is None:
   ypropv = self.setValue(yprop, func=dtplot.get('yfunc', yfunc), fac=dtplot.get('yfac', yfac))
   ylim = getlim(ypropv)
  if xlim is None:
   xpropv = self.setValue(xprop, func=dtplot.get('xfunc', xfunc), fac=dtplot.get('xfac', xfac))
   xlim = getlim(xpropv)
  return xlim, ylim

 def getExtent(self, x, y, extent=None, extent_x=None, extent_y=None):
  if extent is None:
   if extent_x is None:
    extent_x = [min(x), max(x)]
   else:
    extent_x = list(extent_x)
    if extent_x[0] is None:
     extent_x[0] = min(x)
    if extent_x[1] is None:
     extent_x[1] = max(x)
   if extent_y is None:
    extent_y = [min(y), max(y)]
   else:
    extent_y = list(extent_y)
    if extent_y[0] is None:
     extent_y[0] = min(y)
    if extent_x[1] is None:
     extent_y[1] = max(y)
   extent = [extent_x[0], extent_x[1], extent_y[0], extent_y[1]]
  return extent

 def getListAps(self, n_ap, laps=None):
  laps = checkList(laps)
  if laps is None:
   laps = range(n_ap)
  else:
   raps = range(n_ap)
   laps = [raps[item] for item in laps]
  return laps

 def guess_legend_axes(self, dngal, exclude=None):
  exclude = checkList(exclude, return_None=False)
  if isinstance(exclude, tuple):
   exclude = [exclude]
  for item in dngal:
   if dngal[item] != 0 and not item in exclude:
    break
  return item

 def get_id_dict(self, dictionary, idx=None):
  if idx is not None:
   keys = [key for key in dictionary.keys() if not isinstance(key, str)]
   if isinstance(idx, tuple) and isinstance(keys[0], tuple):
    y, x = zip(*keys)
    idx = (range(max(y) + 1)[idx[0]], range(max(x) + 1)[idx[1]])
   else:
    idx = range(max(keys) + 1)[idx]
  return idx

 def get_axes_legend(self, axleg, dngal, exclude=None):
  if axleg is None:
   axleg = self.guess_legend_axes(dngal, exclude=exclude)
  if axleg is False:
   axleg = None
  axleg = self.get_id_dict(dngal, axleg)
  return axleg

 def check_axes_legend(self, axleg, dngal, default=None):
  if axleg is not None and axleg in dngal and dngal[axleg] == 0:
   axleg = default
  return axleg

 def add_legend(self, dax, dngal, axleg=None, loc='lower left', frameon=False, fontsize=None, 
	color=None, artist=True, exclude=None, hcolor=None, **kwargs):
  axleg = self.get_axes_legend(axleg, dngal, exclude=exclude)
  if axleg is not None:
   fontsize = self.lfs if fontsize is None else fontsize
   leg = dax[axleg].legend(loc=loc, fontsize=fontsize, frameon=frameon, **kwargs)
   if color is not None:
    if not isinstance(color, list):
     color = [color] * len(leg.get_texts())
    for i, text in enumerate(leg.get_texts()):
     text.set_color(color[i])
   if hcolor is not None:
    if not isinstance(hcolor, list):
     hcolor = [hcolor] * len(leg.legendHandles)
    for i in range(len(leg.legendHandles)):
     leg.legendHandles[i].set_color(hcolor[i])
   if artist:
    dax[axleg].add_artist(leg)
  return axleg

 def addAxesCustomLineLegend(self, ax, table, prop=None, title_unit=True, title=None, leg_title_size=None, **kwargs):
  if not table in self.dprops:
   print ('>>> WARNING: Table "%s" NOT in dprops!! [%s]' % (table, ' | '.join(self.dprops.keys())))
   return
  if prop is None:
   prop = self.dprops[table].keys()[0]
  if not prop in self.dprops[table]:
   print ('>>> WARNING: Property "%s" NOT in table "%s"!! [%s]' % (prop, table, ' | '.join(self.dprops[table].keys())))
   return
  dprop = self.dprops[table][prop]
  if title_unit and dprop['unit'] is not None and not title is not None:
   title = dprop['unit']
  leg = addCustomLineLegend(dprop['label'], dprop['color'], ax=ax, title=title, **kwargs)
  if leg_title_size is not None and leg is not None:
   leg.get_title().set_fontsize(leg_title_size)
  return leg

 def set_dax_title_yx(self, dax, dngal=None, xnbin=None, ynbin=None, simple=True, pt=72., cor=0):
  if xnbin is None or ynbin is None:
   keys = [key for key in dax.keys() if not isinstance(key, str)]
   y, x = zip(*keys)
   xnbin = max(x) + 1 if xnbin is None else xnbin
   ynbin = max(y) + 1 if ynbin is None else ynbin
  if simple:
   imin = 0
   for j in range(xnbin):
    for i in range(ynbin):
     if imin == i:
      dax[(i,j)].title.set_visible(True)
     else:
      dax[(i,j)].title.set_visible(False)
   jmin = xnbin - 1
   for i in range(ynbin):
    for j in range(xnbin)[::-1]:
     if jmin == j:
      dax[(i,j)].yaxis.label.set_visible(True)
      if not dax[(i,j)].axison:
       label = dax[(i,j)].yaxis.label
       labelpad = dax[(i,j)].yaxis.labelpad
       dax[(i,j)].text(1.0 + labelpad / pt + cor / pt, 0.5, label.get_text(), fontsize=label.get_fontsize(), rotation=90, 
		va='center', ha='center', transform=dax[(i,j)].transAxes, fontproperties=label._fontproperties, 
		color=label.get_color())
     else:
      dax[(i,j)].yaxis.label.set_visible(False)
  else:
   for j in range(xnbin):
    imin = 0
    for i in range(ynbin):
     if dngal[(i,j)] != 0 and imin == i:
      dax[(i,j)].title.set_visible(True)
     else:
      dax[(i,j)].title.set_visible(False)
     if dngal[(i,j)] == 0:
      dax[(i,j)].title.set_visible(False)
      imin += 1
   for i in range(ynbin):
    jmin = xnbin - 1
    for j in range(xnbin)[::-1]:
     if dngal[(i,j)] == 0:
      dax[(i,j)].yaxis.label.set_visible(False)
      jmin -= 1
     if dngal[(i,j)] != 0 and jmin == j:
      dax[(i,j)].yaxis.label.set_visible(True)
     else:
      dax[(i,j)].yaxis.label.set_visible(False)
  return dax

 def set_dax_ticklabels_xy(self, dax, dngal, xnbin=None, ynbin=None, xticks=True, yticks=True, gticks=True):
  if not gticks:
   return
  if xnbin is None or ynbin is None:
   keys = [key for key in dax.keys() if not isinstance(key, str)]
   y, x = zip(*keys)
   xnbin = max(x) + 1 if xnbin is None else xnbin
   ynbin = max(y) + 1 if ynbin is None else ynbin
  # Y-ticks
  jmin = 0
  for j in range(xnbin):
   imin = 0
   for i in range(ynbin):
    dax[(i,j)].tick_params(axis='y', which='major', labelleft='off')
    if dax[(i,j)] == 0:
     imin += 1
   if imin == ynbin - 1:
    jmin += 1
  for i in range(ynbin):
   if dngal[i,jmin] != 0 and yticks:
    dax[(i,jmin)].tick_params(axis='y', which='major', labelleft='on')
  # X-ticks
  imin = ynbin - 1
  for i in range(ynbin)[::-1]:
   jmin = 0
   for j in range(xnbin):
    dax[(i,j)].tick_params(axis='x', which='major', labelbottom='off')
    if dax[(i,j)] == 0:
     jmin += 1
   if jmin == xnbin - 1:
    imin -= 1
  for j in range(xnbin):
   if dngal[imin, j] != 0 and xticks:
    dax[(imin,j)].tick_params(axis='x', which='major', labelbottom='on')
  return dax

 def get_inset_plot(self, dax=None, idax=None, axes=None, fig=None, shrink=1.0, xpad=0.0, ypad=0.0, pos=None):
  ax = None
  if dax is not None and axes is None:
   idax = self.get_id_dict(dax, idx=idax)
   sax = idax if idax is not None else [key for key in dax.keys() if not isinstance(key, str)][0]
   x0, x1, y0, y1 = get_axe_position(dax[sax], fig=fig)
   dx = (x1-x0)
   dy = (y1-y0)
   if idax is not None:
    x0 += dx * xpad
    y0 += dy * ypad
   if pos is not None:
    x0, y0 = pos
   axes = [x0, y0, dx * shrink, dy * shrink]
  if axes is not None:
   ax = fig.add_axes(axes)
  return ax

 def annotate_labels(self, ax, xlabels, ylabels, labels, fontsize=8):
  for x,y,l in zip(xlabels, ylabels, labels):
   ax.annotate(l, (x, y), fontsize=fontsize)

 def setValue(self, value, fac=None, func=None):
  if fac is not None:
   value *= fac
  if func is not None:
   if isinstance(func, str):
    func = getattr(np, func)
   value = func(value)
  return value

 def setValueError(self, value, evalue=None, fac=None, func=None, error=True):
  if not error and evalue is not None:
   evalue = None
  if fac is not None:
   value *= fac
  if evalue is not None and fac is not None:
   evalue *= fac
  if func is not None and evalue is None:
   if isinstance(func, str):
    func = getattr(np, func)
   value = func(value)
  if func is not None and evalue is not None:
   from uncertainties import unumpy
   uvalue = unumpy.uarray(value, evalue)
   if isinstance(func, str):
    if '.' in func:
     func = func.split('.')[1]
    func = getattr(unumpy, func)
   uvalue = func(uvalue)
   value  = unumpy.nominal_values(uvalue)
   evalue = unumpy.std_devs(uvalue)
  return value, evalue

 def plot(self, ax, x, y, ey=None, ex=None, color='black', label=None, lmask=None, pmask=False, pmaskfs=10, 
	ls='-', alpha=0.4, error=True, lw=1, ngal=None, add_ngal=True, unit=None, add_unit=True, 
	check_unit=True, xfac=None, yfac=None, xfunc=None, yfunc=None):
  x, ex = self.setValueError(x, evalue=ex, fac=xfac, func=xfunc, error=error)
  y, ey = self.setValueError(y, evalue=ey, fac=yfac, func=yfunc, error=error)
  if ey is not None and error:
   y1 = y - ey
   y2 = y + ey
   ax.fill_between(x, y1, y2, facecolor=color, alpha=alpha)
  if ex is not None and error:
   x1 = x - ex 
   x2 = x + ex
   ax.fill_betweenx(y, x1, x2, facecolor=color, alpha=alpha)
  label = self.setLabelUnitNgal(label, unit=unit, ngal=ngal, add_unit=add_unit, add_ngal=add_ngal, check_unit=check_unit)
  ax.plot(x, y, color=color, label=label, ls=ls, lw=lw)
  if pmask and lmask is not None:
   self.annotate_labels(ax, x, y, lmask, fontsize=pmaskfs)

 def getNameProp(self, prop, error=False):
  if error:
   if not self.serror.replace('%s','') in prop:
    prop = self.serror % prop
  else:
   if not self.sprop.replace('%s','') in prop:
    prop = self.sprop % prop
  return prop

 def tplot(self, ax, table, xprop, yprop, idx=None, idy=None, axis=0, xaxis=None, yaxis=None, dprop=None, idp=None, 
	ex=None, ey=None, xmask=None, ymask=None, mask_min=None, **kwargs):
  xaxis = axis if xaxis is None else xaxis
  yaxis = axis if yaxis is None else yaxis
  # X
  if isinstance(xprop, str):
   sprop = self.getNameProp(xprop) if not xprop in table.colnames else xprop
   x     = table[sprop] if sprop in table.colnames else None
   xmask = table[self.smask % xprop] if self.smask % xprop in table.colnames else None
   if ex is None:
    ex = table[self.serror % xprop] if self.serror % xprop in table.colnames else None
  else:
   x = xprop
  if idx is not None:
   if x is not None:
    x = np.take(x, idx, axis=xaxis).squeeze()
   if ex is not None:
    ex = np.take(ex, idx, axis=xaxis).squeeze()
   if xmask is not None:
    xmask = np.take(xmask, idx, axis=xaxis).squeeze()
  # Y
  if isinstance(yprop, str):
   sprop = self.getNameProp(yprop) if not yprop in table.colnames else yprop
   y     = table[sprop] if sprop in table.colnames else None
   ymask = table[self.smask % yprop] if self.smask % yprop in table.colnames else None
   if ey is None:
    ey = table[self.serror % yprop] if self.serror % yprop in table.colnames else None
  else:
   y = yprop
  if idy is not None:
   if y is not None:
    y = np.take(y, idy, axis=yaxis).squeeze()
   if ey is not None:
    ey = np.take(ey, idy, axis=yaxis).squeeze()
   if ymask is not None:
    ymask = np.take(ymask, idy, axis=yaxis).squeeze()
  # Dprop
  if dprop is not None and idp is not None:
   for key in ['label', 'color']:
    if key in dprop and not key in kwargs:
     kwargs[key] = dprop[key][idp]
   for key in ['unit']:
    if key in dprop and not key in kwargs:
     kwargs[key] = dprop[key]
  lmask = ymask if ymask is not None else xmask
  if mask_min is not None and lmask is not None:
   # y = np.ma.array(y, mask=lmask < mask_min) --> points are not connected if mask is True False True False
   # Remove points so it can be connected with a line
   nlmask = ~(lmask < mask_min)
   lmask = lmask[nlmask]
   if x is not None:
    x = x[nlmask]
   if y is not None:
    y = y[nlmask]
   if ey is not None:
    ey = ey[nlmask]
   if ex is not None:
    ex = ex[nlmask]
  if x is not None and y is not None:
   self.plot(ax, x, y, ey=ey, ex=ex, lmask=lmask, **kwargs)

 def tplots(self, ax, table, props, dict_tplots=None, xprop=None, **kwargs):
  if not isinstance(props, dict):
   props = checkList(props)
   xprop = checkList(xprop)
   if xprop is None:
    print ('WARNING (tplots): You need to provide either a dictionary (props), or a list and a common xprop (props, xprop)')
    return
   if len(xprop) != len(props):
    xprop = [xprop[0]] * len(props)
   props = OrderedDict((yp, xp) for (yp, xp) in zip(props, xprop))
  for yp in props:
   dtplots = dict_tplots[yp] if isinstance(dict_tplots, dict) and yp in dict_tplots else {}
   self.tplot(ax, table, props[yp], yp, **updateDictDefault(dtplots, kwargs))

 def get_axes(self, i, gs, grid=None, fig=None, dax=None):
  if grid is None:
   if dax is None or not i in dax:
    ax = fig.add_subplot(gs[i])
   else:
    ax = dax[i]
  else:
   ny = i // self.nx
   nx = i % self.nx
   ax = mpl.pyplot.Subplot(fig, gs[ny, nx])
  return ax

 def add_colorbar(self, fig, img, dax, axcbar, cbticks=None, position='right', shrink=1.0, 
	aspect=0.05, pad=1, yf=None, xf=None, rpad=None, rpad2=0.0, tick_params={}, 
	invert_axis=False):
  if axcbar is not None:
   if isinstance(axcbar, (tuple, list)):
    keys = [key for key in dax.keys() if not isinstance(key, str)]
    y, x = zip(*keys)
    xnbin = max(x) + 1 
    ynbin = max(y) + 1
    axcbar = (range(ynbin)[axcbar[0]], range(xnbin)[axcbar[1]])
   else:
    axcbar = dax.keys()[axcbar]
  orientation = 'vertical' if position in ['right', 'left'] else 'horizontal'
  cax = get_axe_colorbar(dax[axcbar], fig=fig, position=position, shrink=shrink, aspect=aspect, yf=yf, xf=xf, rpad=rpad, rpad2=rpad2, pad=pad)
  cbar = fig.colorbar(img, cax=cax, orientation=orientation)
  if cbticks is None:
   cbticks = 'right' if orientation == 'vertical' else 'bottom'
  cbar.ax.xaxis.set_ticks_position(cbticks) if orientation == 'horizontal' else cbar.ax.yaxis.set_ticks_position(cbticks)
  cbar.ax.tick_params(**tick_params)
  if invert_axis:
   if orientation == 'vertical':
    cbar.ax.invert_yaxis()
   else:
    cbar.ax.invert_xaxis()
  dax['colorbar'] = cbar
  return dax

 def get_table_props(self, table, table_props=None, invert=False):
  if isinstance(table, dict):
   tprops = table.keys()
  else:
   tprops = table._meta[self.key_table_props]
  if table_props is not None:
   if all([prop in tprops for prop in table_props]):
    tprops = table_props
   else:
    print ('WARNING: NOT all table_props [%s] present in table! Using table props [%s]' % (' | '.join(table_props), ' | '.join(tprops)))
  tprops = tprops if not invert else tprops[::-1]
  return tprops

 def update_dprops(self, udprop=None):
  if isinstance(self.dprops, dict) and isinstance(udprop, dict):
   self.dprops = updateNestedDict(self.dprops, udprop)

 def update_dprops_units(self, dunits=None, tables=None):
  if isinstance(self.dprops, dict) and isinstance(dunits, dict):
   tables = checkList(tables) if tables is not None else self.dprops.keys()
   for table in self.dprops:
    if table in tables:
     for key in dunits:
      if key in self.dprops[table]:
       self.dprops[table][key]['unit'] = dunits[key]

 def get_dprop_from_dict(self, table):
  if isinstance(table, str):
   if self.dprops is None or len(self.dprops) == 0:
    sys.exit('WARNING: dprop dictionary is EMPTY!')
   if not table in self.dprops:
    sys.exit('WARNING: Table "%s" not available [%s]' % (table, ' | '.join(self.dprops.keys())))
   dprop = copy.deepcopy(self.dprops[table])
   return dprop

 def get_dprop(self, table, udprop=None, kdprop=None):
  dprop = None
  if isinstance(table, str):
   dprop = self.get_dprop_from_dict(table)
  else:
   name = table._meta.get(self.key_name)
   if name in self.dprops:
    dprop = self.get_dprop_from_dict(name)
   else:
    kdprop = self.kdprop if kdprop is None else kdprop
    dprop = copy.deepcopy(table._meta[kdprop])
  if dprop is not None and isinstance(udprop, dict):
   dprop = updateNestedDict(dprop, udprop)
  return dprop

 def sub_dprop(self, dprop, idx):
  if idx is not None:
   dprop = copy.deepcopy(dprop)
   if not isinstance(idx, dict):
    idx = checkList(idx)
   for key in dprop:
    for prop, value in dprop[key].iteritems():
     if isinstance(value, (list, np.ndarray)):
      if isinstance(idx, dict):
       if key in idx:
        dprop[key][prop] = [value[i] for i in checkList(idx[key])]
      else:
       dprop[key][prop] = [value[i] for i in idx]
      dprop[key]['nbin'] = len(dprop[key]['label'])
  return dprop

 def invert_dprop(self, dprop, iprop, iprop_bool=None, exclude=None):
  iprop = checkList(iprop)
  iprop_bool = checkList(iprop_bool)
  if iprop_bool is not None and len(iprop) == len(iprop_bool):
   iprop = [prop for (prop, bprop) in zip(iprop, iprop_bool) if bprop]
  if iprop is not None and len(iprop) > 0:
   ndprop = dprop.copy()
   for prop in iprop:
    if prop in ndprop:
     ndprop[prop] = invert_arrays_in_dictionary(ndprop[prop], exclude=exclude)
   dprop = ndprop
  return dprop

 def get_dprop_value(self, dprop, prop, subprop=None, idx=None, def_value=None, default=False):
  if not prop in dprop:
   return def_value
  value = dprop[prop]
  if subprop is not None and subprop in value:
   value = value[subprop]
  if idx is not None and idx < len(value):
   value = value[idx]
  if default and def_value is not None:
   value = def_value
  return value

 def get_dprop_color(self, dprop, prop, idx, def_value='k', default=False):
  return self.get_dprop_value(dprop, prop, subprop='color', idx=idx, def_value=def_value, default=default)

 def get_dplot(self, dplots, plot, default_props=None):
  dplot = dplots.get(plot, {})
  dplot = copy.deepcopy(dplot)
  if default_props is not None:
   for key in default_props:
    if not key in dplot:
     dplot[key] = default_props[key]
  return dplot

 def interpProp(self, table, x, prop, dx, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate', transpose=False):
  sprop = self.getNameProp(prop)
  improp = sci.interp1d(x, table[sprop], axis=axis, kind=kind, bounds_error=bounds_error, fill_value=fill_value)(dx)
  if transpose:
   improp = improp.T
  return improp

 def getAge(self, prop=None, age=None, sxlabel=None, xlim=None, logage=False):
  if logage:
   if age is None and prop is not None:
    age = self.liageBase if 'dt' in prop else self.lageBase
   xlim = self.lagelim if xlim is None else xlim
   sxlabel = 'log Lookback time [yr]' if sxlabel is None else sxlabel
  else:
   if age is None and  prop is not None:
    age = self.iageGyr if 'dt' in prop else self.ageGyr
   xlim = self.agelim if xlim is None else xlim
   sxlabel = 'Lookback time [Gyr]' if sxlabel is None else sxlabel
  return age, xlim, sxlabel

 def getFunctionTableProps(self, table, prop, func=None, index=None, axis=None, squeeze=False, idx=None, 
	flatten=False, guess=False, verbose=False, fac=None):
  if isinstance(prop, (tuple, list)):
   if func is None:
    print ('>>> WARNING: There are several Y properties [%s]! Need to provide a function!' % ' | '.join(prop))
    return
   vprops = []
   for sprop in prop:
    vprop = self.getTableProp(table, sprop, error=False, guess=guess, verbose=verbose)
    if vprop is None:
     return
    vprops.append(vprop)
   value = func(*vprops)
  else:
   value = self.getTableProp(table, prop, error=False, guess=guess, verbose=verbose)
   if value is None:
    return
   if func is not None:
    value = func(value)
  if fac is not None:
   value *= fac
  value = getArrayIndices(value, index=index, axis=axis, squeeze=squeeze)
  if idx is not None:
   value = value[..., idx]
  if flatten:
   value = value.ravel()
  return value

 def LinearRegression(self, table, x, y, idx=None, index=None, axis=None, func=None, constant=True, method='RLM', 
	prop=None, nx=None, coeff='%s_a', interp='%s_b', e_coeff='%s_ea', e_interp='%s_eb', sfit='%s_fit_r', 
	save_table=False, add_column=False, name_table=None, squeeze=True, name=None, xfit=False, xindex=None, 
	xaxis=None, xfunc=None, xsqueeze=None, x_share_index=False, flatten=False, overwrite=False, 
	verbose=True, verbose_save=True, to_array=False, array_squeeze=True, 
	xguess=False, yguess=False):
  if isinstance(table, str):
   name_table = table
   table = self.dtables[name_table]
  else:
   name_table = table._meta[self.key_name] if self.key_name in table._meta else name_table
  if x_share_index:
   xindex   = index   if xindex   is None else xindex
   xaxis    = axis    if xaxis    is None else xaxis
   xsqueeze = squeeze if xsqueeze is None else xsqueeze
  prop = y    if prop is None else prop
  name = prop if name is None else name
  if isinstance(name, (list, tuple)):
   print ('>>> WARNING: There are seveal "Y" properties! Chosen property "%s" [%s]! Use "name" otherwise!' % (name[0], ' | '.join(name)))
   name = name[0]
  lnames = [item % name for item in [coeff, e_coeff, interp, e_interp]]
  dfit = OrderedDict([(key, []) for key in lnames])
  if nx is not None or xfit:
   dfit[sfit % name] = []
  ly = self.getFunctionTableProps(table, y, func=func, index=index, axis=axis, squeeze=squeeze, idx=idx, flatten=flatten, guess=yguess)
  lx = self.getFunctionTableProps(table, x, func=xfunc, index=xindex, axis=xaxis, squeeze=xsqueeze, flatten=flatten, guess=xguess)
  if lx is None:
   if verbose:
    print ('>>> WARNING: Prop X do NOT exists! [%s]' % name)
   return
  if ly is None:
   if verbose:
    print ('>>> WARNING: Prop Y do NOT exists! [%s]' % name)
   return
  if isinstance(lx, np.ma.MaskedArray) and np.all(lx.mask):
   if verbose:
    print ('>>> WARNING: X values are all masked! [%s]' % name)
   return
  if isinstance(ly, np.ma.MaskedArray) and np.all(ly.mask):
   if verbose:
    print ('>>> WARNING: Y values are all masked! [%s]' % name)
   return
  if ly.ndim > 1 and lx.ndim == 1:
   lx = [lx] * ly.shape[0]
  if ly.ndim == 1:
   ly = [ly]
   if lx.ndim == 1:
    lx = [lx]
  for ix, iy in zip(lx, ly):
   lr = LinearRegression(ix, iy, constant=constant, method=method, verbose=verbose)
   for key, val in zip(lnames, [lr.a, lr.ea, lr.b, lr.eb]):
    dfit[key].append(val)
   if nx is not None:
    dfit[sfit % name].append(dfit[coeff % name][-1] * nx + dfit[interp % name][-1])
   if xfit and nx is None:
    dfit[sfit % name].append(dfit[coeff % name][-1] * x + dfit[interp % name][-1])
  
  if nx is not None or xfit:
   dfit[sfit % name] = np.vstack(dfit[sfit % name])
  if to_array:
   for key in dfit:
    value = np.array(dfit[key])
    if array_squeeze:
     value = value.squeeze()
    dfit[key] = value
  if (save_table or add_column) and name_table is not None:
   self.addColumnsTable(name_table, dfit, overwrite=overwrite)
  if save_table and name_table is not None:
   self.saveTables(name_table, force=True, verbose=verbose_save)
  return dfit

 def LinearRegressionDict(self, table, dlr, cpus=None, verbose_dict=False, dict_get_table={}, dtable=None, **kwargs):
  if dlr is None or len(dlr) < 1:
   return
  table   = self.get_table(table, **dict_get_table)
  cpus    = getCPUS(cpus)
  dict_lr = OrderedDict()
  if cpus is not None:
   # Does not work with numpy < 1.12: TypeError: object pickle not returning list because 
   # of MaskedArrays bug. Even solving the issue, is slower than the serialize option!
   import multiprocess
   dict_lr_pool = OrderedDict()
   pool = multiprocess.Pool(processes=cpus)
   for key in dlr:
    dkwargs = updateDictDefault(dlr[key], kwargs)
    dict_lr_pool[key] = pool.apply_async(self.LinearRegression, (table,), dkwargs)
   pool.close()
   pool.join()
   for i, key in enumerate(dlr, 1):
    if verbose_dict:
     print ('>>> %i/%i [%s]' % (i, len(dlr), key))
    dict_lr[key] = dict_lr_pool[key].get()
    if isinstance(dtable, dict) and isinstance(dict_lr[key], dict):
     dtable.update(dict_lr[key])
  else:
   for i, key in enumerate(dlr, 1):
    if verbose_dict:
     print '%i/%i [%s]' % (i, len(dlr), key)
    dkwargs = updateDictDefault(dlr[key], kwargs)
    dict_lr[key] = self.LinearRegression(table, **dkwargs)
    if isinstance(dtable, dict) and isinstance(dict_lr[key], dict):
     dtable.update(dict_lr[key])
  return dict_lr

 def radialGradients(self, y, tables=None, radius_limits=None, gradient_limits=None, 
	method='RLM', constant=True, average=True, main=True, overwrite=False, 
	no_tables=False, save_dtable=True, save_table=True, nx=None, 
	bin_r=None, **kwargs):
  if bin_r is None:
   bin_r = self.bin_r
  if radius_limits is None:
   radius_limits = (bin_r.min(), bin_r.max())
  radius_min = self.bin_r.min() if radius_limits[0] is None else radius_limits[0]
  radius_max = self.bin_r.max() if radius_limits[1] is None else radius_limits[1]

  radius_limits = (radius_min, radius_max)
  idx = (bin_r >= radius_limits[0])   & (bin_r <= radius_limits[1])
  x = self.bin_r[idx]

  if gradient_limits is not None and nx is None:
   grad_rad_min = self.bin_r.min() if gradient_limits[0] is None else gradient_limits[0]
   grad_rad_max = self.bin_r.max() if gradient_limits[1] is None else gradient_limits[1]
   gradient_limits = (grad_rad_min, grad_rad_max)
   idxg = (bin_r >= gradient_limits[0]) & (bin_r <= gradient_limits[1])
   nx = self.bin_r[idxg]

  self.getGradients(x, y, tables=tables, method=method, constant=constant, average=average, main=main, nx=nx,
	overwrite=overwrite, no_tables=no_tables, save_dtable=save_dtable, 
	save_table=save_table, idx=idx, **kwargs)

 def radialGradientsDict(self, dradial, verbose_dict=False, **kwargs):
  if dradial is None:
   return
  for i, key in enumerate(dradial, 1):
   if verbose_dict:
    print '%i/%i [%s]' % (i, len(dradial), key)
    # Remember to avoid saving to file each time (save_table, main, average)
    self.radialGradients(**updateDictDefault(dradial[key], kwargs))

 def getGradients(self, x, y, tables=None, method='RLM', constant=True, average=True, main=True, average_flatten=False,  
	save_dtable=True, no_tables=False, idx=None, overwrite=False, save_table=True, **kwargs):

  tables = checkList(tables)
  if tables is None:
   tables = self.dtables.keys() if not no_tables else []
  else:
   tables = [table for table in tables if table in self.dtables]
   excluded_tables = [table for table in tables if not table in self.dtables]
   if len(excluded_tables) > 0:
    print ('>>> Some tables NOT found! [%s]' % (' | '.join(excluded_tables)))

  for table in tables:
   self.LinearRegression(table, x, y, xguess=True, yguess=True, add_column=True, idx=idx, constant=constant, method=method, prop=y, overwrite=overwrite, save_table=save_table, **kwargs)

  if main and self.t is not None:
   dgrad = self.LinearRegression(self.t, x, y, xguess=False, yguess=False, idx=idx, constant=constant, method=method, **kwargs)
   if dgrad is not None:
    self.t = addTableColumns(self.t, dgrad, overwrite=overwrite)
   else:
    main = False
  if average and self.ta is not None:
   if average_flatten:
    dgrad = self.LinearRegression(self.t, x, y, xguess=False, yguess=False, flatten=True, idx=idx, constant=constant, method=method, prop=y, **kwargs)
   else:
    dgrad = self.LinearRegression(self.ta, x, y, xguess=True, yguess=True, idx=idx, constant=constant, method=method, prop=y, **kwargs)
   if dgrad is not None:
    self.ta = addTableColumns(self.ta, dgrad, overwrite=overwrite)
   else:
    average = False
  if save_dtable and (main or average):
   self.saveDataTable(verbose=kwargs.get('verbose_save', True))

 def getGradientsDict(self, dgrad, verbose_dict=False, **kwargs):
  if dgrad is None:
   return
  for i, key in enumerate(dgrad, 1):
   if verbose_dict:
    print '%i/%i [%s]' % (i, len(dgrad), key)
    # Remember to avoid saving to file each time (save_table, main, average)
    self.getGradients(**updateDictDefault(dgrad[key], kwargs))

 def radialPlot(self, table, props, bin_r=None, sxlabel='radial distance [HLR]', **kwargs):
  if bin_r is None:
   bin_r = self.bin_r
  self.xPropsPlot(table, bin_r, props, sxlabel=sxlabel, **kwargs)

 def xPropsPlot(self, table, xprop, props, fig=None, figname=None, left=0.07, right=0.98, wspace=0.2,
        bottom=0.15, top=0.92, hspace=0.3, info=None, xinfo=0.03, fsinfo=10, grid=None, t=None, 
	xlabel=True, axtitle=True, xticklabels=True, yticklabels=True, pmask=False, pmaskfs=8, 
	sxlabel=None, error=False, dax=None, legend=None, alpha=0.4, loc=1, ylim=None, dplot={}, 
	lw=1, dflab={}, add_unit=True, leg_title_unit=True, leg_prop=None, min_ngal=None, 
	leg_title_size=None, mask_min=None, dtplot={}, udprop=None, saxtitle=None, 
	dict_get_table={}, dlims=None, **kwargs):
  t = self.get_table(table, **dict_get_table)
  if ylim is None and isinstance(dlims, dict):
   ylim = self.getTablesPropsLim(t, props, **dlims)
  dprop = self.get_dprop(t, udprop=udprop)
  tprop = t._meta[self.key_table_props]
  leg_prop = {} if leg_prop is None else leg_prop
  if len(tprop) > 1:
   print ('WARNING: More than 1 property! [%s]' % ' | '.join(tprop))
   return
  tprop = tprop[0]
  props = checkList(props)
  min_ngal = self.min_ngal if min_ngal is None else min_ngal
  fig, gs = self.set_plot(nfig=len(props), fig=fig, grid=grid, left=left, right=right, wspace=wspace, bottom=bottom, top=top, hspace=hspace, **dplot)
  legend = (len(props) - 1) if legend is None else legend
  dax = OrderedDict() if dax is None else dax
  for i, prop in enumerate(props):
   ax = self.get_axes(i, gs, grid=grid, fig=fig, dax=dax)
   for lb, cl in zip(dprop[tprop]['label'], cycle(dprop[tprop]['color'])):
    tt = t[(t[self.slabel % tprop] == lb)][0]
    if tt[self.key_ngal] < min_ngal:
     continue
    self.tplot(ax, tt, xprop, prop, color=cl, label=lb, ngal=tt[self.key_ngal], pmask=pmask, pmaskfs=pmaskfs, error=error, alpha=alpha, lw=lw, add_unit=add_unit, mask_min=mask_min, **dtplot)
   ilegend = i == legend if not isinstance(legend, np.bool) else legend
   if leg_title_unit and not 'title' in leg_prop:
    leg_prop['title'] = dprop[tprop]['unit']
   isaxtitle = self.getLabelDict(prop, labels=saxtitle)
   self.setTicksLim(ax, xticklabels=xticklabels, yticklabels=yticklabels, legend=ilegend, loc=loc, ylim=ylim, 
        saxtitle=isaxtitle, axtitle=axtitle, xlabel=xlabel, sxlabel=sxlabel, leg_prop=leg_prop, **kwargs)
   if leg_title_size is not None and ax.legend_ is not None:
    ax.legend_.get_title().set_fontsize(leg_title_size)
   if grid is not None:
    fig.add_subplot(ax)
   dax[i] = ax
  fig = self.fig_labels(fig, grid=gs, figname=figname, info=info, xinfo=xinfo, fsinfo=fsinfo, **dflab)
  return fig, dax
 
 def xPropAndAperturePlot(self, table, xprop, props, fig=None, figname=None, left=0.07, right=0.99, wspace=0.0, bottom=0.12,
        top=0.92, hspace=0.0, grid=None, xlabel=True, ylabel=False, axtitle=True, xticklabels=True, saxtitle=None,
        yticklabels=0, pmask=False, pmaskfs=8, laps=None, info=None, pprop=False, sxlabel=None, 
	error=False, dax=None, alpha=0.4, ylim=None, dplot={}, invert=False, lw=1, 
	lid=None, iexclude=None, xlim=None, lstyle=None, dflab={}, add_unit=True, 
	axleg=None, axlap=None, leg_prop=dict(fontsize=10, loc=1), capleg='k',
	leg_aps_prop=dict(fontsize=10, loc=3), mask_min=None, udprop=None, 
	dtplot={}, dict_get_table={}, dlims=None, **kwargs):
  t = self.get_table(table, **dict_get_table)
  if ylim is None and isinstance(dlims, dict):
   ylim = self.getTablesPropsLim(t, props, **dlims)
  dprop = self.get_dprop(t, udprop=udprop)
  laps = self.getListAps(self.n_ap, laps)
  lstyle = lstyle if lstyle is not None else self.lsty
  dplot = updateDictDefault(dplot, dict(xpad=8, ypad=4))
  tprop = t._meta[self.key_table_props]
  if len(tprop) > 1:
   print ('WARNING: More than 1 property! [%s]' % ' | '.join(tprop))
   return
  tprop = tprop[0]
  dprop = self.sub_dprop(dprop, lid)
  dprop = self.invert_dprop(dprop, tprop, invert, exclude=iexclude)
  nbin = dprop[tprop]['nbin']
  props = checkList(props)
  fig, gs = self.set_plot(nfig=len(props), fig=fig, grid=grid, left=left, right=right, wspace=wspace, bottom=bottom, top=top, hspace=hspace, **dplot)
  dax = OrderedDict() if dax is None else dax
  dngal = OrderedDict()
  for i, prop in enumerate(props):
   ax = self.get_axes(i, gs, grid=grid, fig=fig, dax=dax)
   for j, lb in enumerate(dprop[tprop]['label']):
    tt = t[t[self.slabel % tprop] == lb][0]
    dngal[i] = tt[self.key_ngal]
    for k, lsty in zip(laps, cycle(lstyle)):
     self.tplot(ax, tt, xprop, prop, idy=k, yaxis=0, dprop=dprop[tprop], idp=j, ls=lsty, pmask=pmask, pmaskfs=pmaskfs, error=error, alpha=alpha, lw=lw, add_unit=add_unit, mask_min=mask_min, **dtplot)
   xlim, ylim = self.getXYlim(xprop, tt[self.getNameProp(prop)], xlim=xlim, ylim=ylim, dtplot=dtplot)
   iyticklabels = i in checkList(yticklabels) if not isinstance(yticklabels, np.bool) else yticklabels
   isaxtitle = self.getLabelDict(prop, labels=saxtitle)
   self.setTicksLim(ax, xticklabels=xticklabels, yticklabels=iyticklabels, legend=False, ylim=ylim, saxtitle=isaxtitle, axtitle=axtitle, xlim=xlim, **kwargs)
   dax[i] = ax
   if grid is not None:
    fig.add_subplot(ax)
  axleg = self.get_axes_legend(axleg, dngal)
  if axleg is not None:
   addCustomLineLegend(dprop[tprop]['label'], dprop[tprop]['color'], ax=dax[axleg], **leg_prop)
  axlap = self.get_axes_legend(axlap, dngal)
  if axlap is not None:
   clsty = cycle(lstyle)
   addCustomLineLegend([self.label_ap[l] for l in laps], capleg, lstyle=[next(clsty) for l in laps], ax=dax[axlap], **leg_aps_prop)
  fig = self.fig_labels(fig, prop, sxlabel, grid=gs, xlabel=xlabel, ylabel=ylabel,figname=figname, info=info, pprop=pprop, xprop=left, xinfo=right, **dflab)
  return fig, dax

 def xProp_XY_Plot(self, table, xprop, prop, fig=None, figname=None, left=0.04, right=0.98, wspace=0.0,
        bottom=0.12, top=0.94, hspace=0.0, pmask=False, pmaskfs=8, xlabel=True, ylabel=True, axtitle=True, 
	xticklabels=True, yticklabels=0, info=None, pprop=False, grid=None, sxlabel=None, error=False, 
	xlim=None, ylim=None, dax=None, invert=False, legend=None, alpha=0.4, dplot={}, cgal='k', 
	xygal=(0.96, 0.9), hagal='right', xinvert=False, yinvert=False, table_props=None, lw=1,
	iexclude=None, lstyle=None, axleg=None, leg_prop={}, loc=0, legendfs=9, min_ngal=None, 
	lid=None, horizontal=True, dflab={}, ngal_prop={}, add_unit_legend=False, 
	leg_title_unit=True, add_unit=True, add_ngal=True, check_unit=True, 
	leg_title_size=None, update_ylim=True, mask_min=None, udprop=None, 
	dtplot={}, dict_get_table={}, dlims=None, **kwargs):
  t = self.get_table(table, **dict_get_table)
  if ylim is None and isinstance(dlims, dict):
   ylim = self.getTablesPropsLim(t, prop, **dlims)
  dprop = self.get_dprop(t, udprop=udprop)
  dprop = self.sub_dprop(dprop, lid)
  tname = t.name
  tprops = self.get_table_props(t, table_props=table_props)
  dplot = updateDictDefault(dplot, dict(xpad=8, ypad=2))
  lstyle = lstyle if lstyle is not None else self.lsty
  min_ngal = self.min_ngal if min_ngal is None else min_ngal
  if len(tprops) < 2:
   print ('>>> Table "%s" has less than 2 properties! [%s]' % (tname, ' | '.join(tprops)))
   return
  xp, yp = tprops[0], tprops[1]
  dprop = self.invert_dprop(dprop, [xp, yp], [xinvert, yinvert], exclude=iexclude)
  xnbin, ynbin = dprop[xp]['nbin'], dprop[yp]['nbin']
  if not 'ny' in dplot:
   dplot['ny'] = 1 if horizontal else xnbin
  fig, gs = self.set_plot(nfig=xnbin, fig=fig, grid=grid, left=left, right=right, wspace=wspace, bottom=bottom, top=top, hspace=hspace, **dplot)
  dngal = OrderedDict()
  dax = OrderedDict() if dax is None else dax
  for i, xpl in enumerate(dprop[xp]['label']):
   ngal = t[t[self.slabel % xp] == xpl][self.key_ngal].sum()
   dngal[i] = ngal
   if ngal > 0:
    ax = self.get_axes(i, gs, grid=grid, fig=fig, dax=dax)
    kngal = []
    for j, ypl in enumerate(dprop[yp]['label']):
     tt = t[(t[self.slabel % xp] == xpl) & (t[self.slabel % yp] == ypl)][0]
     dax[i] = ax
     if tt[self.key_ngal] >= min_ngal:
      self.tplot(ax, tt, xprop, prop, dprop=dprop[yp], idp=j, pmask=pmask, pmaskfs=pmaskfs, error=error, alpha=alpha, lw=lw, add_unit=add_unit_legend, mask_min=mask_min, **dtplot)
      kngal.append((tt[self.key_ngal],dprop[yp]['color'][j]))
    if xygal is not None and len(kngal) > 0:
     strings, colors = zip(*kngal)
     multicolor_label(xygal[0], xygal[1], strings, colors, ax=ax, **updateDictDefault(ngal_prop, dict(ha='right', fontsize=10)))
   else:
    ax.axis('off')
   iyticklabels = i in checkList(yticklabels) if not isinstance(yticklabels, np.bool) else yticklabels
   saxtitle = None if not horizontal else self.setLabelUnitNgal(xpl, unit=dprop[xp]['unit'], ngal=ngal, add_unit=add_unit, add_ngal=add_ngal, check_unit=check_unit)
   sylabel  = None if horizontal else self.setLabelUnitNgal(xpl, unit=dprop[xp]['unit'], ngal=ngal, add_unit=add_unit, add_ngal=add_ngal, check_unit=check_unit)
   self.setTicksLim(ax, xticklabels=xticklabels, yticklabels=iyticklabels, legend=False, saxtitle=saxtitle, axtitle=axtitle, xlim=xlim, ylim=ylim, sylabel=sylabel, **kwargs)
  axleg = self.get_axes_legend(axleg, dngal)
  if axleg is not None and axleg in dax:
   if leg_title_unit and dprop[yp]['unit'] is not None and not 'title' in leg_prop:
    leg_prop['title'] = dprop[yp]['unit']
   leg = addCustomLineLegend(dprop[yp]['label'], dprop[yp]['color'], ax=dax[axleg], fontsize=legendfs, loc=loc, **leg_prop)
   if leg_title_size is not None and leg is not None:
    leg.get_title().set_fontsize(leg_title_size)
  dax = self.setDictAxesLim(dax, ylim=ylim, update_ylim=update_ylim, xlim=xlim)
  fig = self.fig_labels(fig, prop, sxlabel, grid=gs, xlabel=xlabel, ylabel=ylabel, figname=figname, info=info, pprop=pprop, xprop=left, xinfo=right, **dflab)
  return fig, dax

 def xProp_XYZ_Plot(self, table, xprop, prop, fig=None, figname=None, left=0.07, right=0.97, wspace=0.0,
        bottom=0.08, top=0.96, hspace=0.0, pmask=False, pmaskfs=8, xlabel=True, ylabel=True, 
	axtitle=False, xticklabels=True, yticklabels=True, info=None, pprop=False, grid=None, 
	sxlabel=None, error=False, xlim=None, laps=None, ylim=None, dax=None, invert=False, 
	legend=None, alpha=0.4, dplot={}, cgal='k', xygal=(0.96, 0.9), hagal='right', 
	xinvert=False, yinvert=False, zinvert=False, table_props=None, simple=True, 
	lw=1, cor=0, iexclude=None, lstyle=None, axleg=None, leg_prop={}, loc=0, 
	loc_aps=0, leg_aps_prop={}, cl=None, legendfs=9, ngal_prop={}, axlap=None, 
	text_colors=None, dflab={}, add_unit_leg=False, add_unit=True, udprop=None,
	add_ngal=True, check_unit=True, min_ngal=None, mask_min=None, dtplot={}, 
	gticks=True, check_ngal=True, dict_get_table={}, dlims=None, **kwargs):
  t = self.get_table(table, **dict_get_table)
  if ylim is None and isinstance(dlims, dict):
   ylim = self.getTablesPropsLim(t, prop, **dlims)
  dprop = self.get_dprop(t, udprop=udprop)
  tname = t.name
  tprops = self.get_table_props(t, table_props=table_props)
  dplot = updateDictDefault(dplot, dict(xpad=5, ypad=4))
  laps = self.getListAps(self.n_ap, laps)
  min_ngal = self.min_ngal if min_ngal is None else min_ngal
  lstyle = lstyle if lstyle is not None else self.lsty
  sprop = self.getNameProp(prop)
  if len(tprops) < 3:
   print ('>>> Table "%s" has less than 3 properties! [%s]' % (tname, ' | '.join(tprops)))
   return
  xp, yp, zp = tprops[0], tprops[1], tprops[2]
  dprop = self.invert_dprop(dprop, [xp, yp, zp], [xinvert, yinvert, zinvert], exclude=iexclude)
  xnbin, ynbin, znbin = dprop[xp]['nbin'], dprop[yp]['nbin'], dprop[zp]['nbin']
  fig, gs = self.set_plot(nfig=xnbin*ynbin, ny=ynbin, fig=fig, grid=grid, left=left, right=right, wspace=wspace, bottom=bottom, top=top, hspace=hspace, **dplot)
  xlim, ylim = self.getXYlim(xprop, t[sprop], xlim=xlim, ylim=ylim, nprop=t[self.key_ngal], n_min=min_ngal, check=check_ngal, dtplot=dtplot)
  dngal = OrderedDict()
  dax = OrderedDict() if dax is None else dax
  for i, ypl in enumerate(dprop[yp]['label']):
   for j, xpl in enumerate(dprop[xp]['label']):
    ax = mpl.pyplot.subplot(gs[i, j]) if not (i, j) in dax else dax[(i, j)]
    tt = t[(t[self.slabel % xp] == xpl) & (t[self.slabel % yp] == ypl)]
    ngal = tt[self.key_ngal].sum()
    dngal[(i,j)] = ngal if ngal >= min_ngal else 0
    dax[(i,j)] = ax
    if ngal >= min_ngal:
     kngal = []
     for k, zpl in enumerate(dprop[zp]['label']):
      ttt = tt[tt[self.slabel % zp] == zpl][0]
      if ttt[self.key_ngal] > 0:
       kngal.append([ttt[self.key_ngal], dprop[zp]['color'][k]])
       if ttt[sprop].ndim == 1:
        self.tplot(ax, ttt, xprop, prop, dprop=dprop[zp], idp=k, ngal=ttt[self.key_ngal], pmask=pmask, pmaskfs=pmaskfs, error=error, alpha=alpha, lw=lw, add_unit=add_unit_leg, mask_min=mask_min, **dtplot)
       if ttt[sprop].ndim == 2:
        for l, lsty in zip(range(len(laps)), cycle(lstyle)):
         self.tplot(ax, ttt, xprop, prop, idy=l, yaxis=0, dprop=dprop[zp], idp=k, ls=lsty, pmask=pmask, pmaskfs=pmaskfs, error=error, alpha=alpha, lw=lw, add_unit=add_unit_leg, mask_min=mask_min, **dtplot)
     if xygal is not None and len(kngal) > 0:
      strings, colors = zip(*kngal)
      multicolor_label(xygal[0], xygal[1], strings, colors, ax=ax, **updateDictDefault(ngal_prop, dict(ha='right', fontsize=10)))
    else:
     ax.axis('off')
    saxtitle = self.setLabelUnitNgal(xpl, unit=dprop[xp]['unit'], ngal=dprop[xp]['bin_counts'][j], add_unit=add_unit, add_ngal=add_ngal, check_unit=check_unit)
    sylabel  = self.setLabelUnitNgal(ypl, unit=dprop[yp]['unit'], ngal=dprop[yp]['bin_counts'][i], add_unit=add_unit, add_ngal=add_ngal, check_unit=check_unit)
    self.setTicksLim(ax, xticklabels=xticklabels, yticklabels=yticklabels, legend=False, saxtitle=saxtitle, axtitle=axtitle, xlim=xlim, ylim=ylim, sylabel=sylabel, ylabel=ylabel, **kwargs)
  axleg = self.get_axes_legend(axleg, dngal)
  if axleg is not None:
   addCustomLineLegend(dprop[zp]['label'], dprop[zp]['color'], ax=dax[axleg], fontsize=legendfs, loc=loc, text_colors=text_colors, **leg_prop)
  if ttt[sprop].ndim == 2 and laps is not None:
   axlap = self.get_axes_legend(axlap, dngal, exclude=axleg)
   clsty = cycle(lstyle)
   addCustomLineLegend([self.label_ap[l] for l in laps], 'k', lstyle=[next(clsty) for l in laps], ax=dax[axlap], fontsize=legendfs, loc=loc_aps, **leg_aps_prop)
  self.set_dax_title_yx(dax, dngal, simple=simple, cor=cor)
  self.set_dax_ticklabels_xy(dax, dngal, gticks=gticks)
  fig = self.fig_labels(fig, prop, sxlabel, grid=gs, xlabel=xlabel, ylabel=ylabel, figname=figname, info=info, pprop=pprop, xprop=left, xinfo=right, **dflab)
  return fig, dax

 def agePropAperturePlot(self, table, prop, age=None, sxlabel=None, xlim=None, logage=False, **kwargs):
  age, xlim, sxlabel = self.getAge(prop=prop, age=age, sxlabel=sxlabel, xlim=xlim, logage=logage)
  return self.xPropAperturePlot(table, age, prop, sxlabel=sxlabel, xlim=xlim, **kwargs)

 def xPropAperturePlot(self, table, xprop, prop, fig=None, figname=None, left=0.07, right=0.99, wspace=0.0, bottom=0.12,
        top=0.92, hspace=0.0, grid=None, xlabel=True, ylabel=True, axtitle=True, xticklabels=True, yticklabels=0, 
	pmask=False, pmaskfs=8, laps=None, info=None, pprop=False, sxlabel=None, error=False, label_mass=False, 
	lmass_fmt='%.1f - %.1f', lidax=1, dax=None, loc_ngal=None, keyhub='hubtyp', legend=0, alpha=0.4, 
	ylim=None, dplot={}, invert=False, lw=1, lid=None, iexclude=None, xlim=None, lstyle=None, 
	dflab={}, aadd_unit_leg=False, add_unit=True, add_ngal=True, check_unit=True, udprop=None,
	mask_min=None, dtplot={}, dict_get_table={}, dlims=None, **kwargs):
  t = self.get_table(table, **dict_get_table)
  if ylim is None and isinstance(dlims, dict):
   ylim = self.getTablesPropsLim(t, prop, **dlims)
  dprop = self.get_dprop(t, udprop=udprop)
  laps = self.getListAps(self.n_ap, laps)
  lstyle = lstyle if lstyle is not None else self.lsty
  dplot = updateDictDefault(dplot, dict(xpad=8, ypad=4))
  tprop = t._meta[self.key_table_props][0]
  dprop = self.sub_dprop(dprop, lid)
  dprop = self.invert_dprop(dprop, tprop, invert, exclude=iexclude)
  nbin = dprop[tprop]['nbin']
  fig, gs = self.set_plot(nfig=nbin, fig=fig, grid=grid, left=left, right=right, wspace=wspace, bottom=bottom, top=top, hspace=hspace, **dplot)
  xlim, ylim = self.getXYlim(xprop, t[self.getNameProp(prop)], xlim=xlim, ylim=ylim, dtplot=dtplot)
  dax = OrderedDict() if dax is None else dax
  labels_mass = []
  for i, (tt, lb, cl) in enumerate(zip(t, dprop[tprop]['label'], cycle(dprop[tprop]['color']))):
   if label_mass:
    labels_mass.append(lmass_fmt % (tt['lmass_min'], tt['lmass_max']))
   ax = self.get_axes(i, gs, grid=grid, fig=fig, dax=dax)
   for j, lsty in zip(laps, cycle(lstyle)):
    self.tplot(ax, tt, xprop, prop, idy=j, yaxis=0, color=cl, ls=lsty, label=self.label_ap[j], pmask=pmask, pmaskfs=pmaskfs, error=error, alpha=alpha, lw=lw, add_unit=add_unit_leg, mask_min=mask_min, **dtplot)
   ilegend = i == legend if not isinstance(legend, np.bool) else legend
   iyticklabels = i in checkList(yticklabels) if not isinstance(yticklabels, np.bool) else yticklabels
   artist = False if loc_ngal is None else True
   saxtitle = self.setLabelUnitNgal(lb, unit=dprop[tprop]['unit'], ngal=tt[self.key_ngal], add_unit=add_unit, add_ngal=add_ngal, check_unit=check_unit)
   self.setTicksLim(ax, xticklabels=xticklabels, yticklabels=iyticklabels, legend=ilegend, ylim=ylim, saxtitle=saxtitle, axtitle=axtitle, xlim=xlim, artist=artist, **kwargs)
   dax[i] = ax
   if grid is not None:
    fig.add_subplot(ax)
   if loc_ngal is not None:
    addTextLegend(self.hbin, [(tt[keyhub] == hb).sum() for hb in self.hbin], ax=ax, colors=[self.dhc[item] for item in self.hbin], fontsize=self.lfs, loc=loc_ngal)
  if label_mass:
   addTextLegend(dprop[tprop]['label'], labels_mass, ax=dax[lidax], colors=dprop[tprop]['color'], fontsize=self.lfs, loc=self.loc)
  fig = self.fig_labels(fig, prop, sxlabel, grid=gs, xlabel=xlabel, ylabel=ylabel,figname=figname, info=info, pprop=pprop, xprop=left, xinfo=right, **dflab)
  return fig, dax

 def ageXYAperturePlot(self, table, prop, age=None, xlim=None, sxlabel=None, logage=False, **kwargs):
  age, xlim, sxlabel = self.getAge(prop=prop, age=age, sxlabel=sxlabel, xlim=xlim, logage=logage)
  return self.xXYAperturePlot(table, age, prop, sxlabel=sxlabel, xlim=xlim, **kwargs)

 def xXYAperturePlot(self, table, xprop, prop, fig=None, figname=None, left=0.07, right=0.97, wspace=0.0,
        bottom=0.08, top=0.96, hspace=0.0, pmask=False, pmaskfs=8, xlabel=True, ylabel=True, axtitle=False, 
	xticklabels=True, yticklabels=True, info=None, pprop=False, grid=None, sxlabel=None, error=False, 
	xlim=None, laps=None, ylim=None, dax=None, invert=False, legend=None, alpha=0.4, dplot={}, 
	ngal_prop={}, xygal=(0.96, 0.9), xinvert=False, yinvert=False, table_props=None, simple=True, 
	lw=1, cor=0, iexclude=None, dflab={}, add_unit_leg=True, add_unit=True, add_ngal=True, 
	check_unit=True, min_ngal=None, mask_min=None, dtplot={}, lcolor=None, xcolor=True, 
	gticks=True, cxprop=False, cyprop=False, com_line=False, com_plot=False, dcom_plot={}, 
	dcom_tplot={}, dcom_ticks={}, tcom=None, check_ngal=True, plot_columns=None, 
	dict_tplots={}, def_tplots={}, udprop=None, apertures=True, dlims=None,
	dict_get_table={}, **kwargs):
  t = self.get_table(table, **dict_get_table)
  if ylim is None and isinstance(dlims, dict):
   ylim = self.getTablesPropsLim(t, prop, **dlims)
  if tcom is None and (com_plot or com_line):
   tcom = self.check_average_table(t)[0]
  dprop = self.get_dprop(t, udprop=udprop)
  tname = t.name
  laps = self.getListAps(self.n_ap, laps) if apertures else None
  tprops = self.get_table_props(t, table_props=table_props, invert=invert)
  dplot = updateDictDefault(dplot, dict(xpad=5, ypad=4))
  if len(tprops) < 2:
   print ('>>> Table "%s" has less than 2 properties! [%s]' % (tname, ' | '.join(tprops)))
   return
  xp, yp = tprops[0], tprops[1]
  dprop = self.invert_dprop(dprop, [xp, yp], [xinvert, yinvert], exclude=iexclude)
  xnbin, ynbin = dprop[xp]['nbin'], dprop[yp]['nbin']
  legend = ((ynbin -1), (xnbin - 1)) if legend is None else legend
  min_ngal = self.min_ngal if min_ngal is None else min_ngal
  fig, gs = self.set_plot(nfig=xnbin*ynbin, ny=ynbin, fig=fig, grid=grid, left=left, right=right,
        wspace=wspace, bottom=bottom, top=top, hspace=hspace, **dplot)
  xlim, ylim = self.getXYlim(xprop, t[self.getNameProp(prop)], xlim=xlim, ylim=ylim, nprop=t[self.key_ngal], n_min=min_ngal, check=check_ngal, dtplot=dtplot)
  dngal = OrderedDict()
  dax = OrderedDict() if dax is None else dax
  for i, ypl in enumerate(dprop[yp]['label']):
   for j, xpl in enumerate(dprop[xp]['label']):
    ax = mpl.pyplot.subplot(gs[i, j]) if not (i, j) in dax else dax[(i, j)]
    tt = t[(t[self.slabel % xp] == xpl) & (t[self.slabel % yp] == ypl)][0] # Shape (1, Y, X) --> Get (Y, X)
    ngal = tt[self.key_ngal]
    dngal[(i,j)] = ngal if ngal >= min_ngal else 0
    dax[(i,j)] = ax
    if ngal >= min_ngal:
     sprop = self.getNameProp(prop) if not prop in tt.colnames else prop
     tcolor = lcolor
     if tcolor is None:
      tcolor = dprop[xp]['color'][j] if xcolor else dprop[yp]['color'][i]
     if tt[sprop].ndim == 1:
      self.tplot(ax, tt, xprop, prop, color=tcolor, pmask=pmask, pmaskfs=pmaskfs, error=error, alpha=alpha, lw=lw, add_unit=add_unit_leg, mask_min=mask_min, **dtplot)
      if com_line and tcom is not None:
       self.tplot(ax, tcom, xprop, prop, **updateDictDefault(dcom_tplot, dict(pmask=pmask, pmaskfs=pmaskfs, error=error, alpha=alpha, lw=lw, add_unit=add_unit_leg, mask_min=mask_min)))
     else:
      if laps is not None:
       for k, lsty in izip(laps, cycle(self.lsty)):
        self.tplot(ax, tt, xprop, prop, **updateDictDefault(dtplot, dict(idy=k, yaxis=0, ls=lsty, label=self.label_ap[k], color=tcolor, pmask=pmask, pmaskfs=pmaskfs, error=error, alpha=alpha, lw=lw, add_unit=add_unit_leg, mask_min=mask_min)))
        if com_line and tcom is not None:
         self.tplot(ax, tcom, xprop, prop, **updateDictDefault(dcom_tplot, dict(ls=lsty, pmask=pmask, pmaskfs=pmaskfs, error=error, alpha=alpha, lw=lw, add_unit=add_unit_leg, mask_min=mask_min, idy=k, yaxis=0)))
      else:
       self.tplot(ax, tt, xprop, prop, **updateDictDefault(dtplot, dict(color=tcolor, pmask=pmask, pmaskfs=pmaskfs, error=error, alpha=alpha, lw=lw, add_unit=add_unit_leg, mask_min=mask_min)))
       if com_line and tcom is not None:
        self.tplot(ax, tcom, xprop, prop, **updateDictDefault(dcom_tplot, dict(pmask=pmask, pmaskfs=pmaskfs, error=error, alpha=alpha, lw=lw, add_unit=add_unit_leg, mask_min=mask_min)))
     if plot_columns is not None:
      self.tplots(ax, tt, plot_columns, xprop=xprop, dict_tplots=dict_tplots, **def_tplots)
     if xygal is not None:
      ax.text(xygal[0], xygal[1], self.fmt_ngal % ngal, transform=ax.transAxes, **updateDictDefault(ngal_prop, dict(ha='right', color='k', fontsize=self.lfs-1)))
    else:
     ax.axis('off')
    ilegend = (i == legend[0]) and (j == legend[1]) if not isinstance(legend, np.bool) else legend
    saxtitle = self.setLabelUnitNgal(xpl, unit=dprop[xp]['unit'], ngal=dprop[xp]['bin_counts'][j], add_unit=add_unit, add_ngal=add_ngal, check_unit=check_unit)
    sylabel  = self.setLabelUnitNgal(ypl, unit=dprop[yp]['unit'], ngal=dprop[yp]['bin_counts'][i], add_unit=add_unit, add_ngal=add_ngal, check_unit=check_unit)
    ctitle = self.get_dprop_color(dprop, xp, idx=j, default=(not cxprop))
    cylabel = self.get_dprop_color(dprop, yp, idx=i, default=(not cyprop))
    dpticks = updateDictDefault(kwargs, dict(dtitle=dict(color=ctitle), dylabel=dict(color=cylabel), sylabel=sylabel, ylabel=ylabel, xlim=xlim, ylim=ylim, 
				xticklabels=xticklabels, yticklabels=yticklabels, legend=ilegend, saxtitle=saxtitle, axtitle=axtitle))
    self.setTicksLim(ax, **dpticks)
  self.set_dax_title_yx(dax, dngal, simple=simple, cor=cor)
  self.set_dax_ticklabels_xy(dax, dngal, gticks=gticks)
  if com_plot and tcom is not None:
   ax_com = self.get_inset_plot(dax=dax, fig=fig, **dcom_plot)
   if tt[sprop].ndim == 1:
    self.tplot(ax_com, tcom, xprop, prop, **updateDictDefault(dcom_tplot, dict(pmask=pmask, pmaskfs=pmaskfs, error=error, alpha=alpha, lw=lw, add_unit=add_unit_leg, mask_min=mask_min)))
   else:
    if laps is not None:
     for k, lsty in izip(laps, cycle(self.lsty)):
      self.tplot(ax_com, tcom, xprop, prop, idy=k, yaxis=0, **updateDictDefault(dcom_tplot, dict(ls=lsty, pmask=pmask, pmaskfs=pmaskfs, error=error, alpha=alpha, lw=lw, add_unit=add_unit_leg, mask_min=mask_min)))
    else:
     self.tplot(ax_com, tcom, xprop, prop, **updateDictDefault(dcom_tplot, dict(pmask=pmask, pmaskfs=pmaskfs, error=error, alpha=alpha, lw=lw, add_unit=add_unit_leg, mask_min=mask_min)))
   if plot_columns is not None:
    self.tplots(ax_com, tcom, plot_columns, xprop=xprop, dict_tplots=dict_tplots, **def_tplots)
   if xygal is not None:
    ax_com.text(xygal[0], xygal[1], self.fmt_ngal % tcom[self.key_ngal], transform=ax_com.transAxes, **updateDictDefault(ngal_prop, dict(ha='right', color='k', fontsize=self.lfs-1)))
   self.setTicksLim(ax_com, **updateDictDefault(dcom_ticks, dict(xticklabels=xticklabels, yticklabels=yticklabels, xlim=xlim, ylim=ylim, sxlabel=sxlabel, sylabel=self.getLabel(prop), ylabelp='left', 
	dxlabel=updateDictDefault(dict(weight='normal'), kwargs.get('dxlabel',{})), dylabel=updateDictDefault(dict(weight='normal'), kwargs.get('dylabel',{})), **delKeyDict(kwargs, 'dylabel', new=True))))
   dax['axcom'] = ax_com
  fig = self.fig_labels(fig, prop, sxlabel=sxlabel, grid=gs, xlabel=xlabel, ylabel=ylabel, figname=figname, info=info, pprop=pprop, xprop=left, xinfo=right, **dflab)
  return fig, dax

 def ageRadialPropPlot2D(self, table, prop, fig=None, figname=None, left=0.05, right=0.94, wspace=0.0, bottom=0.12, 
	top=0.94, hspace=0.0, pmask=False, pmaskfs=8, xlabel=True, axtitle=True, xticklabels=True, yticklabels=0, 
	info=None, pprop=False, grid=None, error=False, sxlabel='Lookback time [Gyr]', laps=None, ylim=None, 
	plot_aps=False, colormap='califa_int_r', iage=None, cl='w', lcl='w', axcbar=-1, axleg=None, tprop=None, 
	loc=3, invert=False, shrink=1.0, aspect=0.05, cyf=None, cxf=None, rpad=None, alpha=0.4, cpad=1,
	sylabel='radial distance [HLR]', position='right', dplot={}, cleg='w', dprop=None, lw=2,
	leg_prop={}, lid=None, cbticks=None, iexclude=None, tg_props=None, tg_sel=None, dflab={}, 
	add_unit_leg=True, ctick_params={}, add_unit=True, add_ngal=True, check_unit=True, 
	min_ngal=None, mask_min=None, dtplot={}, cb_invert_axis=False, bin_r_extent=None, 
	extent=None, extent_age=None, extent_r=None, dict_get_table={}, 
	dlims=None, **kwargs):
  if 'califa' in colormap:
   from pyfu.pmas import califa_cmap
   califa_cmap.register_califa_cmap()
  colormap = mpl.pyplot.cm.get_cmap(colormap)
  t = self.get_table(table, **dict_get_table)
  if ylim is None and isinstance(dlims, dict):
   ylim = self.getTablesPropsLim(t, prop, **dlims)
  dprop = self.get_dprop(t, udprop=udprop)
  table_prop = t._meta[self.key_table_props][0]
  dprop = self.sub_dprop(dprop, lid)
  dprop = self.invert_dprop(dprop, table_prop, invert, exclude=iexclude)
  dplot = updateDictDefault(dplot, dict(xpad=7, ypad=2.5))
  laps = self.getListAps(self.n_ap, laps)
  age = self.iageGyr if 'dt' in prop else self.ageGyr
  nbin = dprop[table_prop]['nbin']
  tprop = prop.replace('tr','Apt') if tprop is None else tprop
  tprop = self.getNameProp(tprop)
  axleg = nbin - 1 if axleg is None else range(nbin)[axleg]
  min_ngal = self.min_ngal if min_ngal is None else min_ngal
  fig, gs = self.set_plot(nfig=nbin, fig=fig, grid=grid, left=left, right=right, wspace=wspace, bottom=bottom, top=top, hspace=hspace, **dplot)
  if bin_r_extent is None:
   bin_r_extent = [self.bin_r_in.min(), self.bin_r_out.max()] if self.bin_r_in is not None and self.bin_r_out is not None else self.bin_r
  if iage is None:
   iage = np.logspace(np.log10(self.ageBase.min()), np.log10(self.ageBase.max()), self.ageBase.size) / 1e9
  extent = self.getExtent(iage, bin_r_extent, extent=extent, extent_x=extent_age, extent_y=extent_r)
  dax = OrderedDict()
  for i, label in enumerate(dprop[table_prop]['label']):
   ax = self.get_axes(i, gs, grid=grid, fig=fig, dax=dax)
   if plot_aps:
    axp = ax.twinx()
    axp.autoscale(False)
   tt = t[t[self.slabel % table_prop] == label][0] # Shape (1, Y, X) --> Get (Y, X)
   ngal = tt[self.key_ngal]
   dax[i] = ax
   if ngal >= min_ngal:
    improp = self.interpProp(tt, self.ageBase, prop, self.iageBase, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate', transpose=True)
    ims = ax.imshow(improp, aspect='auto', vmin=0., vmax=1., extent=extent, interpolation='none', cmap=colormap)
    ax.autoscale(False)
    if plot_aps and tprop in tt.colnames:
     for k, lsty in izip(laps, cycle(self.lsty)):
      self.tplot(axp, tt, age, tprop, idy=k, yaxis=0, ls=lsty, label=self.label_ap[k], color=cl, pmask=pmask, pmaskfs=pmaskfs, error=error, alpha=alpha, lw=lw, add_unit=add_unit_leg, mask_min=mask_min, **dtplot)
     axp.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off', labelright='off')
     if axleg is not None and i == axleg:
      leg = axp.legend(loc=loc, **updateDictDefault(leg_prop, dict(frameon=False)))
      [text.set_color(cleg) for text in leg.get_texts()]
    if tg_props is not None:
     tg_props = checkList(tg_props)
     for tg_prop, tcl in izip(tg_props, cycle(checkList(lcl))):
      for l, lsty in izip(getIndexSubList(self.t_mg, tg_sel), cycle(self.lsty)):
       self.tplot(ax, tt, tg_prop, self.bin_r, idx=l, xaxis=1, xfac=1e-9, ls=lsty, label=self.label_t_mg[l], color=tcl, pmask=pmask, pmaskfs=pmaskfs, error=error, alpha=alpha, lw=lw, add_unit=add_unit, mask_min=mask_min, **dtplot)
   else:
    ax.axis('off')
    if plot_aps and tprop in tt.colnames:
     axp.axis('off')
   iyticklabels = i in checkList(yticklabels) if not isinstance(yticklabels, np.bool) else yticklabels
   saxtitle = self.setLabelUnitNgal(label, unit=dprop[table_prop]['unit'], ngal=ngal, add_unit=add_unit, add_ngal=add_ngal, check_unit=check_unit)
   self.setTicksLim(ax, xticklabels=xticklabels, yticklabels=iyticklabels, grid=False, legend=False, saxtitle=saxtitle, axtitle=axtitle, **kwargs)
   if grid is not None:
    fig.add_subplot(ax)
  dax = self.add_colorbar(fig, ims, dax, axcbar, cbticks=cbticks, position=position, shrink=shrink, aspect=aspect, pad=cpad, yf=cyf, xf=cxf, rpad=rpad, tick_params=ctick_params, invert_axis=cb_invert_axis)
  fig = self.fig_labels(fig, sxlabel=sxlabel, sylabel=sylabel, grid=gs, xlabel=xlabel, figname=figname, info=info, pprop=pprop, xprop=left, xinfo=right, **dflab)
  return fig, dax

 def ageRadialXYPlot2D(self, table, prop, fig=None, figname=None, left=0.07, right=0.97, wspace=0.0, bottom=0.08, 
	top=0.96, hspace=0.0, pmask=False, pmaskfs=8, xlabel=True, ylabel=True, axtitle=False, xticklabels=True, 
	yticklabels=True, info=None, pprop=False, grid=None, sxlabel='Lookback time [Gyr]', error=False, 
	laps=None, ylim_aps=None, plot_aps=False, colormap='califa_int_r', iage=None, cl='w', axcbar=(-1,0), 
	axleg=None, tprop=None, loc=3, invert=False, position='left', shrink=1.0, aspect=0.05, cyf=None, 
	cxf=None, rpad=None, rpad2=0.0, cpad=2, alpha=0.4, dplot={}, xinvert=False, yinvert=False, lw=2,
	table_props=None, cbticks=None, lstyle=None, sylabel='radial distance [HLR]', xygal=(0.98, 0.92), 
	simple=True, leg_prop={}, cor=0, iexclude=None, tg_props=None, lcl='w', tg_sel=None, ngal_prop={}, 
	axleg_tg=None, dflab={}, ctick_params={}, add_unit_leg=True, add_unit=True, add_ngal=True, 
	check_unit=True, min_ngal=None, mask_min=None, dtplot={}, leg_tg_prop={}, cxprop=False,
	cyprop=False, gticks=True, com_line=False, com_plot=False, com_plot_line=False, 
	dcom_plot={}, dcom_tplot={}, dcom_ticks={}, dcom_pticks={}, daxp_ticks={}, 
	tcom=None, laps_com=None, lstyle_com=None, laps_com_id=None, udprop=None, 
	cb_invert_axis=False, bin_r_extent=None, extent=None, extent_age=None, 
	extent_r=None, dict_get_table={}, dlims=None, **kwargs):
  if 'califa' in colormap:
   from pyfu.pmas import califa_cmap
   califa_cmap.register_califa_cmap()
  colormap = mpl.pyplot.cm.get_cmap(colormap)
  t = self.get_table(table, **dict_get_table)
  if ylim is None and isinstance(dlims, dict):
   ylim = self.getTablesPropsLim(t, prop, **dlims)
  if tcom is None and (com_plot or com_line):
   tcom = self.check_average_table(t)[0]
  dprop = self.get_dprop(t, udprop=udprop)
  tname = t.name
  laps = self.getListAps(self.n_ap, laps)
  laps_com = self.getListAps(self.n_ap, laps_com) if laps_com is not None else laps
  laps_com_id = checkList(laps_com_id)
  tg_props = checkList(tg_props)
  tprops = self.get_table_props(t, table_props=table_props, invert=invert)
  tprop = prop.replace('tr','Apt') if tprop is None else tprop
  dplot = updateDictDefault(dplot, dict(xpad=4, ypad=3))
  xp, yp = tprops[0], tprops[1]
  age = self.iageGyr if 'dt' in prop else self.ageGyr
  dprop = self.invert_dprop(dprop, [xp, yp], [xinvert, yinvert], exclude=iexclude)
  lstyle = self.lsty if lstyle is None else checkList(lstyle)
  lstyle_com = lstyle if lstyle_com is None else checkList(lstyle_com)
  min_ngal = self.min_ngal if min_ngal is None else min_ngal
  if len(tprops) < 2:
   print ('>>> Table "%s" has less than 2 properties! [%s]' % (tname, ' | '.join(tprops)))
   return
  xnbin, ynbin = dprop[xp]['nbin'], dprop[yp]['nbin']
  fig, gs = self.set_plot(nfig=xnbin*ynbin, ny=ynbin, fig=fig, grid=grid, left=left, right=right, wspace=wspace, bottom=bottom, top=top, hspace=hspace, **dplot)
  if bin_r_extent is None:
   bin_r_extent = [self.bin_r_in.min(), self.bin_r_out.max()] if self.bin_r_in is not None and self.bin_r_out is not None else self.bin_r
  if iage is None:
   iage = np.logspace(np.log10(self.ageBase.min()), np.log10(self.ageBase.max()), self.ageBase.size) / 1e9
  extent = self.getExtent(iage, bin_r_extent, extent=extent, extent_x=extent_age, extent_y=extent_r)
  dngal = OrderedDict()
  dax  = OrderedDict()
  daxp = OrderedDict()
  for i, ypl in enumerate(dprop[yp]['label']):
   for j, xpl in enumerate(dprop[xp]['label']):
    ax = mpl.pyplot.subplot(gs[i, j]) if not (i, j) in dax else dax[(i, j)]
    if plot_aps:
     axp = ax.twinx()
     axp.autoscale(False)
     daxp[(i,j)] = axp
    tt = t[(t[self.slabel % xp] == xpl) & (t[self.slabel % yp] == ypl)][0] # Shape (1, Y, X) --> Get (Y, X)
    ngal = tt[self.key_ngal]
    dngal[(i,j)] = ngal if ngal >= min_ngal else 0
    dax[(i,j)] = ax
    if ngal >= min_ngal:
     improp = self.interpProp(tt, self.ageBase, prop, self.iageBase, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate', transpose=True)
     ims = ax.imshow(improp, aspect='auto', vmin=0., vmax=1., extent=extent, interpolation='none', cmap=colormap)
     ax.autoscale(False)
     if plot_aps and self.getNameProp(tprop) in tt.colnames:
      for k, lsty in izip(laps, cycle(lstyle)):
       self.tplot(axp, tt, age, tprop, idy=k, yaxis=0, ls=lsty, label=self.label_ap[k], color=cl, pmask=pmask, pmaskfs=pmaskfs, error=error, alpha=alpha, lw=lw, add_unit=add_unit_leg, mask_min=mask_min, **dtplot)
      if com_line and tcom is not None:
       for k, lsty in izip(laps_com, cycle(lstyle_com)):
        if laps_com_id is None or (laps_com_id is not None and k in laps_com_id):
         self.tplot(axp, tcom, age, tprop, idy=k, yaxis=0, **updateDictDefault(dcom_tplot, dict(ls=lsty, color=cl, pmask=pmask, pmaskfs=pmaskfs, error=error, alpha=alpha, lw=lw, add_unit=add_unit_leg, mask_min=mask_min)))
      if ylim_aps is not None:
       axp.set_ylim(ylim_aps)
      axp.tick_params(**updateDictDefault(daxp_ticks, dict(axis='both', which='both', bottom='off', top='off', labelbottom='off', labelleft='off', labelright='off')))
     if xygal is not None:
      ax.text(xygal[0], xygal[1], self.fmt_ngal % ngal, transform=ax.transAxes, **updateDictDefault(ngal_prop, dict(ha='right', color='w', fontsize=10)))
     if tg_props is not None:
      for tg_prop, tcl in izip(tg_props, cycle(checkList(lcl))):
       for l, lsty in izip(getIndexSubList(self.t_mg, tg_sel), cycle(self.lsty)):
        self.tplot(ax, tt, tg_prop, self.bin_r, idx=l, xaxis=1, xfac=1e-9, ls=lsty, label=self.label_t_mg[l], color=tcl, pmask=pmask, pmaskfs=pmaskfs, error=error, alpha=alpha, lw=lw, add_unit=add_unit, mask_min=mask_min, **dtplot)
    else:
     ax.axis('off')
     if plot_aps and self.getNameProp(tprop) in tt.colnames:
      axp.axis('off')
    saxtitle = self.setLabelUnitNgal(xpl, unit=dprop[xp]['unit'], ngal=dprop[xp]['bin_counts'][j], add_unit=add_unit, add_ngal=add_ngal, check_unit=check_unit)
    isylabel  = self.setLabelUnitNgal(ypl, unit=dprop[yp]['unit'], ngal=dprop[yp]['bin_counts'][i], add_unit=add_unit, add_ngal=add_ngal, check_unit=check_unit)
    ctitle = self.get_dprop_color(dprop, xp, idx=j, default=(not cxprop))
    cylabel = self.get_dprop_color(dprop, yp, idx=i, default=(not cyprop))
    dpticks = updateDictDefault(kwargs, dict(dtitle=dict(color=ctitle), dylabel=dict(color=cylabel)))
    self.setTicksLim(ax, xticklabels=xticklabels, yticklabels=yticklabels, saxtitle=saxtitle, axtitle=axtitle, sylabel=isylabel, ylabel=ylabel, **updateDictDefault(dpticks, dict(grid=False, legend=False)))
    if grid is not None:
     fig.add_subplot(ax)
  if plot_aps:
   haxleg = self.get_axes_legend(None, dngal)
   handles, labels = daxp[haxleg].get_legend_handles_labels()
   axleg = self.get_axes_legend(axleg, dngal)
   # Need to avoid deep copy since it complains copying "handles" (TransfromNode instances can not be copied. Consider using frozen() instead)
   axlegp = self.add_legend(daxp, dngal, axleg, **updateDictDefault(leg_prop, dict(handles=handles, labels=labels, loc=loc, color=cl), deep=False)) 
  if tg_props is not None:
   axleg_tg = axleg if axleg_tg is None else axleg_tg
   axleg_tg = self.check_axes_legend(axleg_tg, dngal)
   axlegp = axlegp if plot_aps else None
   self.add_legend(dax, dngal, axleg_tg, exclude=axlegp, **updateDictDefault(leg_tg_prop, dict(color=cl, loc=loc)))
  if com_plot and tcom is not None:
   ax_com = self.get_inset_plot(dax=dax, fig=fig, **dcom_plot)
   improp = self.interpProp(tcom, self.ageBase, prop, self.iageBase, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate', transpose=True)
   ax_com.imshow(improp, aspect='auto', vmin=0., vmax=1., extent=extent, interpolation='none', cmap=colormap)
   ax_com.autoscale(False)
   if xygal is not None:
    ax_com.text(xygal[0], xygal[1], self.fmt_ngal % tcom[self.key_ngal], transform=ax_com.transAxes, **updateDictDefault(ngal_prop, dict(ha='right', color='w', fontsize=10)))
   if com_plot_line:
    axp_com = ax_com.twinx()
    axp_com.autoscale(False)
    for k, lsty in izip(laps_com, cycle(lstyle_com)):
     self.tplot(axp_com, tcom, age, tprop, idy=k, yaxis=0, **updateDictDefault(dcom_tplot, dict(ls=lsty, color=cl, pmask=pmask, pmaskfs=pmaskfs, error=error, alpha=alpha, lw=lw, add_unit=add_unit_leg, mask_min=mask_min)))
    self.setTicksLim(axp_com, **updateDictDefault(dcom_pticks, dict(grid=False, ylabelp='right', dxticklabel=kwargs.get('dxticklabel', {}), dyticklabel=kwargs.get('dyticklabel', {}),  dylabel=kwargs.get('dylabel', {}))))
    daxp['axpcom'] = axp_com
    if tg_props is not None:
     for tg_prop, tcl in izip(tg_props, cycle(checkList(lcl))):
      for l, lsty in izip(getIndexSubList(self.t_mg, tg_sel), cycle(self.lsty)):
       self.tplot(ax_com, tt, tg_prop, bin_r, idx=l, xaxis=1, xfac=1e-9, ls=lsty, label=self.label_t_mg[l], color=tcl, pmask=pmask, pmaskfs=pmaskfs, error=error, alpha=alpha, lw=lw, add_unit=add_unit, mask_min=mask_min, **dtplot)
   self.setTicksLim(ax_com, **updateDictDefault(dcom_ticks, dict(xticklabels=xticklabels, yticklabels=yticklabels, sxlabel=sxlabel, sylabel=sylabel, ylabelp='left', grid=False,
	dxlabel=updateDictDefault(dict(weight='normal'), kwargs.get('dylabel',{})), dylabel=updateDictDefault(dict(weight='normal'), kwargs.get('dylabel',{})), **delKeyDict(kwargs, 'dylabel', new=True))))
   dax['axcom'] = ax_com
  self.set_dax_title_yx(dax, dngal, simple=simple, cor=cor)
  self.set_dax_ticklabels_xy(dax, dngal, gticks=gticks)
  dax = self.add_colorbar(fig, ims, dax, axcbar, cbticks=cbticks, position=position, shrink=shrink, aspect=aspect, pad=cpad, yf=cyf, xf=cxf, rpad=rpad, rpad2=rpad2, tick_params=ctick_params, invert_axis=cb_invert_axis)
  fig = self.fig_labels(fig, prop, sxlabel, grid=gs, xlabel=xlabel, ylabel=ylabel, sylabel=sylabel, figname=figname, info=info, pprop=pprop, xprop=left, xinfo=right, **dflab)
  return fig, dax

 def ageAperturePropPlot(self, table, prop, age=None, xlim=None, sxlabel=None, logage=False, **kwargs):
  age, xlim, sxlabel = self.getAge(prop=prop, age=age, sxlabel=sxlabel, xlim=xlim, logage=logage)
  return self.xAperturePropPlot(table, age, prop, sxlabel=sxlabel, xlim=xlim, **kwargs)

 def xAperturePropPlot(self, table, xprop, prop, fig=None, figname=None, left=0.06, right=0.98, wspace=0.0, bottom=0.15,
        top=0.92, hspace=0.0, grid=None, xlabel=True, ylabel=True, axtitle=True, xticklabels=True, yticklabels=0, 
	pmask=False, pmaskfs=8, pprop=False, info=None, error=False, laps=None, sxlabel=None, dax=None, 
	lmass_fmt='%.1f - %.1f', idmass=None, legend=0, alpha=0.4, dplot={}, ylim=None, udprop=None,
	xlim=None, lw=1, dflab={}, add_unit_leg=True, leg_prop={}, leg_title_unit=True, dtplot={}, 
	leg_title_size=None, leg_mass_prop={}, mask_min=None, dict_get_table={}, dlims=None, **kwargs):
  t = self.get_table(table, **dict_get_table)
  if ylim is None and isinstance(dlims, dict):
   ylim = self.getTablesPropsLim(t, prop, **dlims)
  dprop = self.get_dprop(t, udprop=udprop)
  tname = t.name
  laps = self.getListAps(self.n_ap, laps)
  n_ap = len(laps)
  if legend is not None:
   legend = range(n_ap)[legend]
  dplot = updateDictDefault(dplot, dict(xpad=9, ypad=3))
  fig, gs = self.set_plot(nfig=n_ap, fig=fig, grid=grid, left=left, right=right, wspace=wspace, bottom=bottom, top=top, hspace=hspace, **dplot)
  dax = OrderedDict() if dax is None else dax
  xlim, ylim = self.getXYlim(xprop, t[self.getNameProp(prop)], xlim=xlim, ylim=ylim, dtplot=dtplot)
  for iax, i in enumerate(laps):
   ax = self.get_axes(iax, gs, grid=grid, fig=fig, dax=dax)
   lbmass = []
   for j, (tt, lb, cl) in enumerate(zip(t, dprop[tname]['label'], cycle(dprop[tname]['color']))):
    label_mass = lmass_fmt % (tt['lmass_min'], tt['lmass_max'])
    lbmass.append(label_mass)
    self.tplot(ax, tt, xprop, prop, idy=i, yaxis=0, color=cl, label=lb, ngal=tt[self.key_ngal], pmask=pmask, pmaskfs=pmaskfs, error=error, alpha=alpha, lw=lw, add_unit=add_unit_leg, mask_min=mask_min, **dtplot)
   ilegend = iax == legend if not isinstance(legend, np.bool) else legend
   iyticklabels = iax in checkList(yticklabels) if not isinstance(yticklabels, np.bool) else yticklabels
   if leg_title_unit and not 'title' in leg_prop:
    leg_prop['title'] = dprop[tname]['unit']
   self.setTicksLim(ax, xticklabels=xticklabels, yticklabels=iyticklabels, legend=ilegend, saxtitle=self.label_ap[i], axtitle=axtitle, xlim=xlim, ylim=ylim, artist=True, leg_prop=leg_prop, **updateDictDefault(kwargs, dict(legendfs=self.lfs)))
   if leg_title_size is not None and ax.legend_ is not None:
    ax.legend_.get_title().set_fontsize(leg_title_size)
   if idmass is not None and i == idmass:
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, lbmass, **updateDictDefault(leg_mass_prop, dict(fontsize=self.lfs)))
   if grid is not None:
    fig.add_subplot(ax)
   dax[iax] = ax
  fig = self.fig_labels(fig, prop, sxlabel, grid=gs, xlabel=xlabel, ylabel=ylabel, figname=figname, info=info, pprop=pprop, xprop=left, xinfo=right, **dflab)
  return fig, dax

 def ageApertureXYPlot(self, table, prop, age=None, xlim=None, sxlabel=None, logage=False, **kwargs):
  age, xlim, sxlabel = self.getAge(prop=prop, age=age, sxlabel=sxlabel, xlim=xlim, logage=logage)
  return self.xApertureXYPlot(table, age, prop, sxlabel=sxlabel, xlim=xlim, **kwargs)

 def xApertureXYPlot(self, table, xprop, prop, fig=None, figname=None, left=0.07, right=0.97, wspace=0.0,
        bottom=0.08, top=0.96, hspace=0.0, pmask=False, pmaskfs=8, xlabel=True, ylabel=True, axtitle=True, 
	xticklabels=True, yticklabels=False, info=None, pprop=False, grid=None, sxlabel=None, error=False, 
	laps=None, dax=None, invert=False, legend=None, xlim=None, ylim=None, alpha=0.4, dplot={}, 
	table_props=None, xinvert=False, yinvert=False, lstyle=None, simple=True, iexclude=None, 
	lw=1, cor=0, dflab={}, add_unit_leg=True, add_unit=True, add_ngal=True, check_unit=True, 
	min_ngal=None, mask_min=None, dtplot={}, udprop=None, check_ngal=True, dlims=None,
	dict_get_table={}, **kwargs):
  t = self.get_table(table, **dict_get_table)
  if ylim is None and isinstance(dlims, dict):
   ylim = self.getTablesPropsLim(t, prop, **dlims)
  dprop = self.get_dprop(t, udprop=udprop)
  tname = t.name
  laps = self.getListAps(self.n_ap, laps)
  n_ap = len(laps)
  tprops = self.get_table_props(t, table_props=table_props, invert=invert)
  dplot = updateDictDefault(dplot, dict(xpad=5, ypad=4))
  if len(tprops) < 2:
   print ('>>> Table "%s" has less than 2 properties! [%s]' % (tname, ' | '.join(tprops)))
   return
  xp, yp = tprops[0], tprops[1]
  dprop = self.invert_dprop(dprop, [xp, yp], [xinvert, yinvert], exclude=iexclude)
  xnbin, ynbin = dprop[xp]['nbin'], dprop[yp]['nbin']
  legend = (n_ap - 1)
  min_ngal = self.min_ngal if min_ngal is None else min_ngal
  fig, gs = self.set_plot(nfig=n_ap*ynbin, ny=ynbin, fig=fig, grid=grid, left=left, right=right,
        wspace=wspace, bottom=bottom, top=top, hspace=hspace, **dplot)
  lstyle = self.lsty if lstyle is None else checkList(lstyle)
  xlim, ylim = self.getXYlim(xprop, t[self.getNameProp(prop)], xlim=xlim, ylim=ylim, nprop=t[self.key_ngal], n_min=min_ngal, check=check_ngal, dtplot=dtplot)
  dngal = OrderedDict()
  dax = OrderedDict() if dax is None else dax
  for i, ypl in enumerate(dprop[yp]['label']):
   for jax, j in enumerate(laps):
    ax = mpl.pyplot.subplot(gs[i, jax]) if not (i, jax) in dax else dax[(i, jax)]
    tgal = 0
    for k, xpl in enumerate(dprop[xp]['label']):
     tt = t[(t[self.slabel % xp] == xpl) & (t[self.slabel % yp] == ypl)][0] # Shape (1, Y, X) --> Get (Y, X)
     ngal = tt[self.key_ngal]
     tgal += ngal
     if ngal < min_ngal:
      continue
     self.tplot(ax, tt, xprop, prop, idy=j, yaxis=0, dprop=dprop[xp], idp=k, ngal=ngal, pmask=pmask, pmaskfs=pmaskfs, error=error, alpha=alpha, lw=lw, add_unit=add_unit_leg, mask_min=mask_min, **dtplot)
    dngal[(i, jax)] = tgal 
    dax[(i, jax)] = ax
    ilegend = jax == legend if not isinstance(legend, np.bool) else legend
    sylabel = self.setLabelUnitNgal(ypl, unit=dprop[yp]['unit'], ngal=dprop[yp]['bin_counts'][i], add_unit=add_unit, add_ngal=add_ngal, check_unit=check_unit)
    self.setTicksLim(ax, xticklabels=xticklabels, yticklabels=yticklabels, legend=ilegend, 
	saxtitle=self.label_ap[j], axtitle=axtitle, xlim=xlim, ylim=ylim, sylabel=ypl, ylabel=legend, **kwargs)
    if grid is not None:
     fig.add_subplot(ax)
  self.set_dax_title_yx(dax, dngal, simple=simple, cor=cor)
  self.set_dax_ticklabels_xy(dax, dngal, gticks=gticks)
  fig = self.fig_labels(fig, prop, sxlabel, grid=gs, xlabel=xlabel, ylabel=ylabel, figname=figname, info=info, pprop=pprop, xprop=left, xinfo=right, **dflab)
  return fig, dax

 def ageXApertureYPlot(self, table, prop, age=None, xlim=None, sxlabel=None, logage=False, **kwargs):
  age, xlim, sxlabel = self.getAge(prop=prop, age=age, sxlabel=sxlabel, xlim=xlim, logage=logage)
  return self.xXApertureYPlot(table, age, prop, sxlabel=sxlabel, xlim=xlim, **kwargs)

 def xXApertureYPlot(self, table, xprop, prop, fig=None, figname=None, left=0.07, right=0.97, wspace=0.0,
        bottom=0.08, top=0.96, hspace=0.0, pmask=False, pmaskfs=8, xlabel=True, ylabel=True, axtitle=True, 
	xticklabels=True, yticklabels=False, info=None, pprop=False, grid=None, sxlabel=None, error=False, 
	laps=None, dax=None, invert=False, legend=None, xlim=None, ylim=None, alpha=0.4, dplot={}, 
	table_props=None, xinvert=False, yinvert=False, lstyle=None, simple=True, iexclude=None, 
	lw=1, cor=0, dflab={}, add_unit_leg=False, add_unit=True, add_ngal=True, check_unit=True, 
	min_ngal=None, mask_min=None, dtplot={}, gticks=True, check_ngal=True, dlims=None,
	udprop=None, dict_get_table={}, **kwargs):
  t = self.get_table(table, **dict_get_table)
  if ylim is None and isinstance(dlims, dict):
   ylim = self.getTablesPropsLim(t, prop, **dlims)
  dprop = self.get_dprop(t, udprop=udprop)
  tname = t.name
  laps = self.getListAps(self.n_ap, laps)
  n_ap = len(laps)
  min_ngal = self.min_ngal if min_ngal is None else min_ngal
  tprops = self.get_table_props(t, table_props=table_props, invert=invert)
  dplot = updateDictDefault(dplot, dict(xpad=5, ypad=4))
  if len(tprops) < 2:
   print ('>>> Table "%s" has less than 2 properties! [%s]' % (tname, ' | '.join(tprops)))
   return
  xp, yp = tprops[0], tprops[1]
  dprop = self.invert_dprop(dprop, [xp, yp], [xinvert, yinvert], exclude=iexclude)
  xnbin, ynbin = dprop[xp]['nbin'], dprop[yp]['nbin']
  legend = 0
  fig, gs = self.set_plot(nfig=n_ap*xnbin, ny=n_ap, fig=fig, grid=grid, left=left, right=right,
        wspace=wspace, bottom=bottom, top=top, hspace=hspace, **dplot)
  lstyle = self.lsty if lstyle is None else checkList(lstyle)
  xlim, ylim = self.getXYlim(xprop, t[self.getNameProp(prop)], xlim=xlim, ylim=ylim, nprop=t[self.key_ngal], n_min=min_ngal, check=check_ngal, dtplot=dtplot)
  dngal = OrderedDict()
  dax = OrderedDict() if dax is None else dax
  for iax, (i, lsty) in enumerate(zip(laps, cycle(lstyle))):
   for j, xpl in enumerate(dprop[xp]['label']):
    ax = mpl.pyplot.subplot(gs[iax, j]) if not (iax, j) in dax else dax[(iax, j)]
    tgal = 0
    for k, ypl in enumerate(dprop[yp]['label']):
     tt = t[(t[self.slabel % xp] == xpl) & (t[self.slabel % yp] == ypl)][0] # Shape (1, Y, X) --> Get (Y, X)
     ngal = tt[self.key_ngal]
     tgal += ngal
     if ngal < min_ngal:
      continue
     self.tplot(ax, tt, xprop, prop, idy=i, yaxis=0, dprop=dprop[yp], idp=k, ngal=ngal, pmask=pmask, pmaskfs=pmaskfs, error=error, alpha=alpha, lw=lw, add_unit=add_unit_leg, mask_min=mask_min, **dtplot)
    dngal[(iax ,j)] = tgal 
    dax[(iax, j)] = ax
    ilegend = iax == legend if not isinstance(legend, np.bool) else legend
    saxtitle = self.setLabelUnitNgal(xpl, unit=dprop[xp]['unit'], ngal=dprop[xp]['bin_counts'][j], add_unit=add_unit, add_ngal=add_ngal, check_unit=check_unit)
    self.setTicksLim(ax, xticklabels=xticklabels, yticklabels=yticklabels, legend=ilegend, saxtitle=saxtitle, axtitle=axtitle, xlim=xlim, ylim=ylim, sylabel=self.label_ap[i], ylabel=legend, **kwargs)
    if grid is not None:
     fig.add_subplot(ax)
  self.set_dax_title_yx(dax, dngal, simple=simple, cor=cor)
  self.set_dax_ticklabels_xy(dax, dngal, gticks=gticks)
  fig = self.fig_labels(fig, prop, sxlabel, grid=gs, xlabel=xlabel, ylabel=ylabel, figname=figname, info=info, pprop=pprop, xprop=left, xinfo=right, **dflab)
  return fig, dax

 def getColumnTable(self, prop, table=None, idx3=None, idx=None, idcol=None, big_endian=True, axis3=-1, **kwargs):
  t = self.t if table is None else table
  if isinstance(prop, Table):
   prop = prop.data
  value = self.getFunctionTableProps(t, prop, **kwargs)
  if value is None:
   return
  if isinstance(value, np.ndarray) and value.ndim == 3:
   if idx3 is None:
    print ('WARNING: 3D array! Need to select an index with "idx3"')
    return
   else:
    value = np.squeeze(np.take(value, idx3, axis=axis3))
  if idx is not None:
   value = np.take(value, idx, axis=0)
  if idcol is not None:
   value = np.take(value, idcol, axis=1)
  # byteswap().newbyteorder() --> Solves Big-endian
  if big_endian and value.dtype.byteorder == '>':
   value = value.byteswap().newbyteorder()
  return value

 def getColumnsTable(self, props, table=None, idx3=None, idx=None, idcol=None, 
	big_endian=True, axis3=-1, extra_columns=None, **kwargs):
  t = self.t if table is None else table
  def_dict = dict(idx3=idx3, idx=idx, idcol=idcol, big_endian=big_endian, axis3=axis3, **kwargs)
  dt = OrderedDict()
  if isinstance(props, dict):
   for key in props:
    value = props[key]
    if isinstance(value, dict):
     prop = value.pop('props')
     dprop = updateDictDefault(value, def_dict)
     dt[key] = self.getColumnTable(prop, table=t, **dprop)
    else:
     dt[key] = self.getColumnTable(value, table=t, **def_dict)
  else:
   props = checkList(props)
   for prop in props:
    dt[prop] = self.getColumnTable(prop, table=t, **def_dict)
  if isinstance(extra_columns, dict):
   for col in extra_columns:
    value = extra_columns[col]
    if not isinstance(value, (np.ndarray, list, tuple)):
     value = [value] * len(t)
    dt[col] = value
  for key in dt:
   if dt[key] is None:
    dt.pop(key)
  return dt

 def getDataFrame(self, prop, table=None, fac=None, idx=None, columns=None, idcol=None, idx3=None, axis3=-1, 
	id_vars=None, value_vars=None, value_vars_rest=False, id_vars_rest=False, var_name='x', value_name='y', 
	diff=False, diff_fac=None, return_columns=False, index=None, dropnan=True, guess_diff_idvars=True, 
	update_columns=True, x=None, y=None, hue=None, nmin=None, percentile=None, low_per=None, upp_per=None, 
	dict_per=None, pivot_index=None, values=None, big_endian=True, pivot_columns=None, aggfunc='mean', 
	diff_avg=False, reindex=False, indexes=None, unit=False, order=None, mask_nmin=False, 
	mask=np.nan, extra_columns=None, **kwargs):
  import pandas as pd
  nmin = self.min_ngal if (nmin is None and mask_nmin) else nmin
  columns = checkList(columns)
  if idcol is not None and columns is not None:
   columns = np.take(columns, idcol).tolist()
  columns = checkList(columns)
  if columns is not None and extra_columns is not None:
   columns += extra_columns.keys()
  value = self.getColumnsTable(prop, fac=fac, idx=idx, idcol=idcol, idx3=idx3, axis3=axis3, extra_columns=extra_columns, **kwargs)
  data = pd.DataFrame(data=value, columns=columns) 
  if table in self.dprops:
   for tbin in self.dprops[table]:
    label = self.t[self.slab % (table, tbin)].data
    if idx is not None:
     label = np.take(label, idx)
    data[tbin] = label 
  if diff and (pivot_index is None or (pivot_index is not None and not diff_avg)):
   if table in self.dprops and index is None:
    index = self.dprops[table].keys()
   if index is not None:
    data.set_index(index, inplace=True)
   data = data.diff(axis=1)
   if diff_fac is not None:
    data *= diff_fac
   # Remove columns with all NaN
   if dropnan:
    data.dropna(axis=1, how='all', inplace=True)
   data.reset_index(level=index, inplace=True)
   if id_vars is None and table in self.dprops and guess_diff_idvars and len(data.columns) > 2:
    id_vars = self.dprops[table].keys()
  if id_vars is not None or value_vars is not None:
   if id_vars is not None and value_vars is None and value_vars_rest:
    value_vars = [item for item in data.columns.values if not item in id_vars]
   if id_vars is None and value_vars is not None and id_vars_rest:
    id_vars = [item for item in data.columns.values if not item in value_vars]
   data = pd.melt(data, id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)
   if update_columns:
    columns = data.columns.values
  if x is not None and y is not None:
   data = maskCategoricalDataFrame(data, x, y, hue, nmin=nmin, percentile=percentile, low_per=low_per, upp_per=upp_per, dict_per=dict_per)
  if index is not None and pivot_index is None and not diff:
   data = data.set_index(index).sortlevel(0) # Sort by multiindex so they are not repeated
  if pivot_index is not None:
   if nmin is not None:
    ncount = pd.pivot_table(data, index=pivot_index, columns=pivot_columns, values=values, aggfunc='count')
   data = pd.pivot_table(data, index=pivot_index, columns=pivot_columns, values=values, aggfunc=aggfunc)
   if nmin is not None:
    data[ncount < nmin] = mask
  if diff_avg and pivot_index is not None:
   data = data.diff(axis=1)
   if diff_fac is not None:
    data *= diff_fac
   # Remove columns with all NaN
   if dropnan:
    data.dropna(axis=1, how='all', inplace=True)
  if reindex:
   data = self.reindexData(data, table, indexes=indexes, unit=unit, order=order)
  if return_columns:
   return data, columns
  else:
   return data

 def getTableDataFrame(self, stable, prop, fac=1.0, idx=None, columns=None, idcol=None, idx3=None, 
	axis3=-1, set_index=True, invert=False, error=True, extra_columns=None, guess=True, 
	eprop=None, reindex=False, indexes=None, unit=False, nmin=None, 
	key_nmin=None, mask=np.nan, mask_nmin=False, big_endian=True):
  table = stable
  import pandas as pd
  if isinstance(table, str):
   if table in self.dtables:
    table = self.dtables[table]
   else:
    print ('>>> WARNING: Table "%s" NOT available [%s]' % (table, ' | '.join(self.dtables.keys())))
    return
  value = table[self.sprop % prop] * fac if guess else table[prop] * fac
  evalue = None
  if error:
   if guess:
    evalue = table[self.serror % prop] * fac
   if not guess and eprop is not None:
    evalue = table[eprop] * fac
  if value.ndim == 3:
   if idx3 is None:
    print ('WARNING: 3D array! Need to select an index with "idx3"')
    return
   else:
    value = np.squeeze(np.take(value, idx3, axis=axis3))
    if evalue is not None:
     evalue = np.squeeze(np.take(evalue, idx3, axis=axis3))
  if idx is not None:
   value = np.take(value, idx, axis=0)
   if evalue is not None:
    evalue = np.take(evalue, idx, axis=0)
  if idcol is not None:
   value = np.take(value, idcol, axis=1)
   if evalue is not None:
    evalue = np.take(evalue, idcol, axis=1)
   if columns is not None:
    columns = np.take(columns, idcol)
  if columns is not None and isinstance(columns, np.ndarray):
   columns = columns.tolist()
  columns = checkList(columns)
  if value.dtype.byteorder == '>' and big_endian:
   value = value.byteswap().newbyteorder()
  data = pd.DataFrame(data=value, columns=columns)
  if evalue is not None:
   ecolumns = ['e[%s]' % col for col in columns]
   if evalue.dtype.byteorder == '>' and big_endian:
    evalue = evalue.byteswap().newbyteorder()
   edata = pd.DataFrame(data=evalue, columns=ecolumns)
  for xprop in self.dprops[stable]:
   label = table[self.slabel % xprop]
   if idx is not None:
    label = np.take(label, idx)
   if label.dtype.byteorder == '>' and big_endian:
    label = label.byteswap().newbyteorder()
   data[xprop] = label
  if evalue is not None:
   data = pd.concat([data, edata], axis=1)
  if extra_columns is not None:
   extra_columns = checkList(extra_columns)
   extra_columns = [item for item in extra_columns if item in table.columns]
   extra_columns = None if len(extra_columns) == 0 else extra_columns
   if extra_columns is not None:
    ecdata = pd.DataFrame({key: table[key].byteswap().newbyteorder() if (table[key].dtype.byteorder == '>' and big_endian) else table[key] for key in extra_columns})
    data = pd.concat([data, ecdata], axis=1)
  xy = self.dprops[stable].keys()
  lcols = [item for sublist in zip(columns, ecolumns) for item in sublist] if evalue is not None else columns
  if mask_nmin:
   nmin     = self.min_ngal if nmin is None else nmin
   key_nmin = self.key_ngal if key_nmin is None else key_nmin
  if nmin is not None and (key_nmin is not None and key_nmin in table.colnames):
   for col in lcols:
    data[col][table[key_nmin] < nmin] = mask
  if invert:
   xy = xy[::-1]
  if set_index:
   data.set_index(xy, inplace=True)
   cols = lcols
  else:
   cols = xy + lcols
  if extra_columns is not None and cols is not None:
   cols.extend(extra_columns)
  if cols is not None and len(cols) > 0:
   data = data[cols]
  if reindex:
   data = self.reindexData(data, stable, indexes=indexes, unit=unit, order=xy)
  return data

 def reindexData(self, data, table, indexes=None, unit=False, order=None, set_index=False):
  import pandas as pd
  if indexes is None and table in self.dprops:
   index = self.dprops[table].keys() if order is None else order
   names = [self.dprops[table][key]['unit'] for key in index] if unit else index
   indexes = pd.MultiIndex.from_tuples(list(product(*[self.dprops[table][key]['label'] for key in index])), names=names)
  if indexes is not None:
   if set_index:
    data = data.set_index(index)
   data = data.reindex(index=indexes)
  return data

 def setDictAxesLim(self, dax, ylim=None, update_ylim=False, xlim=None, update_xlim=False, xfrac=None, yfrac=None):
  if ylim is None and update_ylim:
   ymin = min([min(dax[i].get_ylim()) for i in dax])
   ymax = max([max(dax[i].get_ylim()) for i in dax])
   if yfrac is not None:
    dy = np.abs(ymax - ymin) * yfrac
    ymin -= dy
    ymax += dy
   for i in dax:
    dax[i].set_ylim(ymin,ymax)
  if xlim is None and update_xlim:
   xmin = min([min(dax[i].get_xlim()) for i in dax])
   xmax = max([max(dax[i].get_xlim()) for i in dax])
   if xfrac is not None:
    dx = np.abs(xmax - xmin) * xfrac
    xmin -= dx
    xmax += dx
   for i in dax:
    dax[i].set_xlim(xmin,xmax)
  return dax

 def set_boxplot_mpl(self, boxplot, dbox=None, lw=1.5, ls='-', cmed='#d4d4d4', border='#2f2828'):
  # For maplotlib, no need for seaborn
  #boxplot = ax.boxplot(data, patch_artist=True, **dbox)
  #self.set_boxplot_mpl(boxplot, dict(color=dprop[xp]['color']))
  nbox = len(boxplot['boxes'])
  dbox = {} if dbox is None else dbox
  color = dbox['color'] if 'color' in dbox else None
  color = [color] * nbox if isinstance(color, str) else color

  for i,box in enumerate(boxplot['boxes']):
   box.update(dict(facecolor=color[i], edgecolor=border, linewidth=lw))
   box.update(dbox.get('boxes', {}))
  for box in boxplot['whiskers']:
   box.update(dict(color=border, linewidth=lw, linestyle=ls))
   box.update(dbox.get('whiskers', {}))
  for box in boxplot['caps']:
   box.update(dict(color=border, linewidth=lw))
   box.update(dbox.get('caps', {}))
  for box in boxplot['medians']:
   box.update(dict(color=cmed, linewidth=lw))
   box.update(dbox.get('medians', {}))
  for box in boxplot['fliers']:
   box.update(dict(markerfacecolor=border, markeredgecolor=border, marker='d'))
   box.update(dbox.get('fliers', {}))

 def set_boxplot_sns(self, dbox=None, lw=1, wlw=1, ls='-', cmed='#d4d4d4', border='#2f2828', marker='d'):
  dfbox = {}
  dfbox['boxprops']      = dict(edgecolor=border, linewidth=lw)
  dfbox['whiskerprops']  = dict(color=border, linewidth=wlw, linestyle=ls)
  dfbox['capprops']      = dict(color=border, linewidth=wlw)
  dfbox['medianprops']   = dict(color=cmed, linewidth=lw)
  dfbox['flierprops']    = dict(markerfacecolor=border, markeredgecolor=border, marker=marker)
  dbox = {} if dbox is None else dbox
  dbox = updateDictDefault(dbox, dfbox)
  return dbox

 def set_boxplot_color(self, ax, colors, groups=1, ialpha=0.5, legend=True, hatchs=None, alphas=None, 
	hatch_invert=False, alpha_invert=False, color_invert=False):
  ncolors = colors if isinstance(colors, int) else len(colors)
  hatch_func = np.repeat if hatch_invert else np.tile
  alpha_func = np.repeat if alpha_invert else np.tile
  color_func = np.tile   if color_invert else np.repeat
  colors = color_func([None] * colors, groups) if isinstance(colors, int) else color_func(colors, groups)
  alphas = np.linspace(ialpha, 1.0, groups) if alphas is None else alphas
  alphas = alpha_func(alphas, ncolors) 
  hatchs = hatch_func(checkList(hatchs), ncolors) if hatchs is not None else [None] * alphas.size
  for i, artist in enumerate(ax.artists):
   if colors[i] is not None:
    artist.set_facecolor(colors[i])
   if alphas[i] is not None:
    artist.set_alpha(alphas[i])
   if hatchs[i] is not None:
    artist.set_hatch(hatchs[i])
  if legend:
   for i, legpatch in enumerate(ax.get_legend().get_patches()):
    if colors[0] is not None:
     legpatch.set_facecolor(colors[0])
    legpatch.set_alpha(alphas[i])
    legpatch.set_hatch(hatchs[i])
  return ax

 def set_boxplot_alpha(self, ax, alpha=None):
  if alpha is not None:
   for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, alpha))
  return ax

 def tboxplot(self, ax, table, prop, dprop, xp, sylabel='y', sxlabel='x', nfile=None, idx=None, fac=1.0, idx2=None,
	idx3=None, blstyle=None, axis_order=['bin', 'prop', 'idx'], dread={}, order=None, hue_order=None, 
	palette=None, showfliers=False, color=None, leg_prop={}, violinplot=False, alpha=None, 
	leg_title_unit=False, leg_title_size=None, legend=False, ngal=False, xygal=(0.05, 0.9), 
	ngal_prop={}, alphas=None, hatchs=None, ialpha=0.5, desaturate=None, percentile=None, 
	dict_per=None, low_per=None, upp_per=None, nmin=None, box_width=None, box_color=None, 
	boxdot_size=None, boxdot_color=None, boxdot_prop_color=False, dboxdot={}, **kwargs):
  import seaborn as sn
  import pandas as pd
  if self.t is None:
   if tname is not None:
    self.readDataTable(tname, nfile=nfile, **dread)
   else:
    sys.exit('WARNING: Table empty! Need to provide a name in "tname" variable!')
  value = self.t[prop].data
  for indices, axis in zip([idx, idx2, idx3], [0,1,2]):
   if indices is not None:
    # Squeeze to remove extra single dimension: (1, 20, 30) --> (20, 30)
    # If indices is a one item list: np.take(a, [0]).shape = (1, 20, 30)  | np.take(a, 0).shape = (20, 30)
    value = np.squeeze(np.take(value, indices, axis=axis))
  groups = 1 if value.ndim == 1 else value.shape[-1]
  data = pd.DataFrame(data=value * fac)
  data[xp] = self.t[self.slab % (table, xp)]
  data = pd.melt(data, xp, var_name=sxlabel, value_name=sylabel)
  dbox = self.set_boxplot_sns(dbox=kwargs)
  dorder = {'idx': sxlabel, 'prop': sylabel, 'bin': xp}
  x   = dorder[axis_order[0]]
  y   = dorder[axis_order[1]]
  hue = dorder[axis_order[2]] if len(axis_order) == 3 and value.ndim == 2 else None
  if palette is None:
   palette = dprop[xp]['color']
  if desaturate is not None:
   palette = desaturate_colors(palette, desaturate)
  if order is None and (axis_order[0] == 'bin' or axis_order[1] == 'bin'):
   order = dprop[xp]['label']
  if hue_order is None and len(axis_order) == 3 and axis_order[2] == 'bin':
   hue_order = dprop[xp]['label'] 
  if x is not None and y is not None:
   data = maskCategoricalDataFrame(data, x, y, hue, nmin=nmin, percentile=percentile, hue_order=hue_order, order=order, low_per=low_per, upp_per=upp_per, dict_per=dict_per)
  if violinplot:
   ax = sn.violinplot(x=x, y=y, hue=hue, data=data, ax=ax, showfliers=showfliers, hue_order=hue_order, order=order, palette=palette, **dbox)
  else:
   ax = sn.boxplot(x=x, y=y, hue=hue, data=data, ax=ax, showfliers=showfliers, hue_order=hue_order, order=order, palette=palette, **dbox)
  if (axis_order[0] == 'bin' or axis_order[1] == 'bin') and len(axis_order) == 3:
   ax = self.set_boxplot_color(ax, palette, groups=groups, alphas=alphas, hatchs=hatchs, ialpha=ialpha)
  else:
   ax = self.set_boxplot_alpha(ax, alpha)
  #dbin_counts = dict(pd.value_counts(data[x]))
  #dbin_counts  = dict(data[x].value_counts())
  # Legend
  leg = None
  if not 'title' in leg_prop and len(axis_order) == 3 and axis_order[2] == 'bin':
   leg_prop['title'] = dprop[xp]['unit']
  if leg_title_unit and dprop[xp]['unit'] is not None:
   leg_prop['title'] = dprop[xp]['unit']
  if len(axis_order) == 3:
   handles, labels = ax.get_legend_handles_labels()
   if ngal:
    labels = [self.fmt_lngal % (item, nbin) for (item, nbin) in zip(labels, dprop[xp]['bin_counts'])]
   leg = ax.legend(handles, labels, **leg_prop)
  if len(axis_order) == 2 and legend:
   labels = dprop[xp]['label']
   if ngal:
    labels = [self.fmt_lngal % (item, nbin) for (item, nbin) in zip(labels, dprop[xp]['bin_counts'])]
   leg = addCustomPatchLegend(labels, dprop[xp]['color'], ax=ax, artist=True, **leg_prop)
  if leg_title_size is not None and leg is not None:
   leg.get_title().set_fontsize(leg_title_size)
  if len(axis_order) < 3 and not legend and ngal:
   multicolor_label(xygal[0], xygal[1], dprop[xp]['bin_counts'], dprop[xp]['color'], ax=ax, **updateDictDefault(ngal_prop, dict(ha='center left', fontsize=12, weight='bold')))
   
  if blstyle is not None:
   for i, (box, bls) in enumerate(zip(ax.artists, cycle(blstyle))):
    box.set_linestyle(bls)

  if violinplot:
   if box_color is not None:
    [line.set_color(box_color) for line in ax.lines]
   if box_width is not None:
    [line.set_linewidth(box_width) for line in ax.lines[1::2]]
    if boxdot_size is None:
     boxdot_size = box_width
   if boxdot_prop_color:
    [col.set_color(color) for (col, color) in zip(ax.collections[1::2], dprop[xp]['color'])]
   if boxdot_color is not None:
    [col.set_color(boxdot_color) for col in ax.collections[1::2]]
   if boxdot_size is not None or len(dboxdot) > 0:
    boxdot_color = dprop[xp]['color'] if boxdot_prop_color else ('w' if boxdot_color is None else boxdot_color)
    sx, sy = zip(*np.vstack([col.get_offsets() for col in ax.collections[1::2]]))
    boxdot_size = np.sqrt(ax.collections[1].get_sizes()[0])/2. if boxdot_size is None else boxdot_size
    [col.remove() for col in ax.collections[1::2]]
    ax.scatter(sx, sy, **updateDictDefault(dboxdot, dict(marker='o', s=boxdot_size**2, c=boxdot_color, zorder=1000)))

  return ax

 def propBoxPlot(self, tables, prop, fig=None, grid=None, dax=None, lstyle=None, invert=False, dplot={}, dbox={}, 
	dflab={}, sylabel=None, pprop=False, left=0.05, right=0.98, wspace=0.0, bottom=0.08, top=0.96, 
	hspace=0.0, info=None, figname=None, yticklabels=0, ylim=None, ltitle=None, axtitle=True, 
	update_ylim=True, idx=None, idx2=None, idx3=None, dinvert={}, dtable_box={}, dticks={},
	udprop=None, **kwargs):
  tables = checkList(tables)
  dplot = updateDictDefault(dplot, dict(xpad=3, ypad=2.5))
  fig, gs = self.set_plot(nfig=len(tables), fig=fig, grid=grid, left=left, right=right, wspace=wspace, bottom=bottom, top=top, hspace=hspace, **dplot)
  dax = OrderedDict() if dax is None else dax
  for i, table in enumerate(tables):
   ax = self.get_axes(i, gs, grid=grid, fig=fig, dax=dax)
   dprop = self.get_dprop(table, udprop=udprop)
   tprops = dprop.keys()
   if len(tprops) > 1:
    print ('>>> Table "%s" has to have 1 property! We try anyway... [%s]' % (table, ' | '.join(tprops)))
   xp = tprops[0]
   xpinvert = dinvert.get(xp, invert)
   dprop = self.invert_dprop(dprop, xp, xpinvert)
   lstyle = self.lsty if lstyle is None else checkList(lstyle)
   sylabel = self.getLabel(prop) if sylabel is None else sylabel
   iyticklabels = i in checkList(yticklabels) if not isinstance(yticklabels, np.bool) else yticklabels
   saxtitle = ltitle[i] if ltitle is not None else None
   dtbox = updateNestedDict(copy.deepcopy(dbox), dtable_box.get(table, {}))
   ax = self.tboxplot(ax, table, prop, dprop, xp, sylabel=sylabel, idx=idx, idx2=idx2, idx3=idx3, **dtbox)
   # Need to update fig=fig (canvas.draw) since yticklabels give empty text
   dbticks = updateNestedDict(updateDictDefault(kwargs, dict(ylabel=False, xlabel=False, legend=False, artist=False)), dticks.get(table, {}))
   self.setTicksLim(ax, yticklabels=iyticklabels, ylim=ylim, saxtitle=saxtitle, axtitle=axtitle, fig=fig, **dbticks)
   if grid is not None:
    fig.add_subplot(ax)
   dax[i] = ax
  dax = self.setDictAxesLim(dax, ylim=ylim, update_ylim=update_ylim)
  fig = self.fig_labels(fig, prop, grid=gs, sylabel=sylabel, figname=figname, info=info, pprop=pprop, xprop=left, xinfo=right, **dflab)
  return fig, dax

 def tnboxplot(self, ax, table, prop, nfile=None, invert=None, dread={}, order=None, hue_order=None, palette=None, showfliers=False, color=None, 
	leg_prop={}, x=None, hue=None, y=None, violinplot=False, alpha=None, leg_title_unit=False, leg_title_size=None, legend=False, 
	ngal=False, xygal=(0.05, 0.9), ngal_prop={}, desaturate=None, nmin=None, dframe={}, data=None, percentile=None, 
	low_per=None, upp_per=None, dict_per=None, blstyle=None, udprop=None, **kwargs):
  import seaborn as sn
  import pandas as pd
  if self.t is None:
   if tname is not None:
    self.readDataTable(tname, nfile=nfile, **dread)
   else:
    sys.exit('WARNING: Table empty! Need to provide a name in "tname" variable!')
  if data is None:
   dframe = updateDictDefault(dframe, dict(table=table, value_vars_rest=True, id_vars_rest=True))
   data, columns = self.getDataFrame(prop, return_columns=True, **dframe)
  dbox = self.set_boxplot_sns(dbox=kwargs)
  dprop = self.get_dprop(table, udprop=udprop)
  if invert is not None:
   if isinstance(invert, str):
    invert = {invert: True}
   if isinstance(invert, (list, tuple)):
    if not all([item in dprop.keys() for item in invert]):
     print ('WARNING: NOT all invert keys [%s] in dprop dictionary [%s]' % (' | '.join(invert), ' | '.join(dprop.keys())))
     linvert = [item for item in invert if item in dprop.keys()]
     binvert = [True] * len(invert)
   elif isinstance(invert, dict):
    linvert = [item for item in invert if item in dprop]
    binvert = [invert[item] for item in linvert]
   if len(linvert) > 0:
    dprop = self.invert_dprop(dprop, linvert, binvert)
  x = x if x is not None else dprop.keys()[0]
  numeric_columns = data._get_numeric_data().columns.values
  numeric_columns = [item for item in numeric_columns if not item in dprop.keys()]
  categorical_columns = [item for item in data.columns.values if not item in numeric_columns]
  if y is None:
   y = numeric_columns[0]
  if hue is None and len(categorical_columns) > 1:
   hue = [key for key in categorical_columns if key != x][0]
  if palette is None:
   if (hue is not None and not isinstance(hue, bool)) and hue in dprop:
    palette = dprop[hue]['color']
   else:
    palette = dprop[x]['color']
  if desaturate is not None and (palette is not None and not isinstance(palette, bool)):
   palette = desaturate_colors(palette, desaturate)
  if order is None and x in dprop:
   order = dprop[x]['label']
  if hue_order is None and (hue is not None and not isinstance(hue, bool)) and hue in dprop:
   hue_order = dprop[hue]['label']
  hue = None if isinstance(hue, bool) and not hue else hue
  data = maskCategoricalDataFrame(data, x, y, hue, nmin=nmin, percentile=percentile, hue_order=hue_order, order=order, low_per=low_per, upp_per=upp_per, dict_per=dict_per)
  if violinplot:
   ax = sn.violinplot(x=x, y=y, hue=hue, data=data, ax=ax, showfliers=showfliers, hue_order=hue_order, order=order, palette=palette, **dbox)
  else:
   ax = sn.boxplot(x=x, y=y, hue=hue, data=data, ax=ax, showfliers=showfliers, hue_order=hue_order, order=order, palette=palette, **dbox)
  # Legend 
  leg = None
  if not 'title' in leg_prop and hue is not None and hue in dprop: 
   leg_prop['title'] = dprop[hue]['unit']
  if leg_title_unit and hue is not None and hue in dprop and dprop[hue]['unit'] is not None:
   leg_prop['title'] = dprop[hue]['unit']
  if palette is not None and hue is not None:
   handles, labels = ax.get_legend_handles_labels()
   if ngal:
    labels = [self.fmt_lngal % (item, nbin) for (item, nbin) in zip(labels, dprop[x]['bin_counts'])]
   leg = ax.legend(handles, labels, **leg_prop)
  if palette is not None and hue is None and legend:
   labels = dprop[x]['label']
   if leg_title_unit and dprop[x]['unit'] is not None:
    leg_prop['title'] = dprop[x]['unit']
   if ngal:
    labels = [self.fmt_lngal % (item, nbin) for (item, nbin) in zip(labels, dprop[x]['bin_counts'])]
   leg = addCustomPatchLegend(labels, dprop[x]['color'], ax=ax, artist=True, **leg_prop)
  if leg_title_size is not None and leg is not None:
   leg.get_title().set_fontsize(leg_title_size)
  if hue is not None and not legend and ngal:
   multicolor_label(xygal[0], xygal[1], dprop[x]['bin_counts'], dprop[x]['color'], ax=ax, **updateDictDefault(ngal_prop, dict(ha='center left', fontsize=12, weight='bold')))
   
  if blstyle is not None:
   for i, (box, bls) in enumerate(zip(ax.artists, cycle(blstyle))):
    box.set_linestyle(bls)

  return ax

 def propBoxPlotND(self, tables, prop, fig=None, grid=None, dax=None, lstyle=None, dplot={}, dbox={}, 
	dflab={}, sylabel=None, pprop=False, left=0.05, right=0.98, wspace=0.0, bottom=0.08, top=0.96, 
	hspace=0.0, info=None, figname=None, yticklabels=0, ylim=None, ltitle=None, axtitle=True, 
	update_ylim=True, dtable_box={}, dticks={}, min_ngal=None, udprop=None, **kwargs):
  tables = checkList(tables)
  dplot = updateDictDefault(dplot, dict(xpad=3, ypad=2.5))
  fig, gs = self.set_plot(nfig=len(tables), fig=fig, grid=grid, left=left, right=right, wspace=wspace, bottom=bottom, top=top, hspace=hspace, **dplot)
  dax = OrderedDict() if dax is None else dax
  for i, table in enumerate(tables):
   ax = self.get_axes(i, gs, grid=grid, fig=fig, dax=dax)
   lstyle = self.lsty if lstyle is None else checkList(lstyle)
   sylabel = self.getLabel(prop) if sylabel is None else sylabel
   iyticklabels = i in checkList(yticklabels) if not isinstance(yticklabels, np.bool) else yticklabels
   saxtitle = ltitle[i] if ltitle is not None else None
   dtbox = updateNestedDict(copy.deepcopy(dbox), dtable_box.get(table, {}))
   min_ngal = self.min_ngal if min_ngal is None else min_ngal
   min_ngal = None if not min_ngal else min_ngal
   ax = self.tnboxplot(ax, table, prop, **updateDictDefault(dtbox, dict(nmin=min_ngal, udprop=udprop)))
   # Need to update fig=fig (canvas.draw) since yticklabels give empty text
   dbticks = updateNestedDict(updateDictDefault(kwargs, dict(ylabel=False, xlabel=False, legend=False, artist=False)), dticks.get(table, {}))
   self.setTicksLim(ax, yticklabels=iyticklabels, ylim=ylim, saxtitle=saxtitle, axtitle=axtitle, fig=fig, **dbticks)
   if grid is not None:
    fig.add_subplot(ax)
   dax[i] = ax
  dax = self.setDictAxesLim(dax, ylim=ylim, update_ylim=update_ylim)
  fig = self.fig_labels(fig, prop, grid=gs, sylabel=sylabel, figname=figname, info=info, pprop=pprop, xprop=left, xinfo=right, **dflab)
  return fig, dax

 def propGridBoxPlot(self, table, prop, xp, yp, dframe={}, fig=None, grid=None, dax=None, dbox={}, dflab={}, sxlabel=None, sylabel=None, 
	pprop=False, info=None, figname=None, xticklabels=True, yticklabels=True, ylim=None, axtitle=True, update_ylim=True, dplot={}, 
	var_name='x', value_name='y', violinplot=False, xinvert=False, yinvert=False, iexclude=None, simple=True, cor=0, add_unit=True, 
	add_ngal=True, check_unit=True, order=None, ylim_guess=True, hue=None, hue_order=None, palette=None, hue_x=False, hatchs=None, 
	alphas=None, axleg_h=None, axleg_x=None, leg_h_prop={}, leg_x_prop={}, leg_title_unit=True, add_leg_h=True, add_leg_x=True, 
	leg_h_title_size=None, leg_x_title_size=None, ialpha=0.5, ngal=False, ngal_prop={}, xygal=(0.05, 0.9), cleg=None, cw=None, 
	tngal=True, xytngal=(0.05, 0.05), tngal_prop={}, lcolor=None, min_ngal=None, gticks=True, com_plot=False, dcom_plot={}, 
	dcom_ticks={}, cxprop=False, cyprop=False, z=None, percentile=None, low_per=None, upp_per=None, 
	dict_per=None, udprop=None, **kwargs):
  import seaborn as sn
  id_vars = [xp, yp] if dframe.get('id_vars', None) is None else dframe.get('id_vars')
  dframe = updateDictDefault(dframe, dict(table=table, value_vars_rest=True, id_vars_rest=True, id_vars=id_vars))
  data, columns = self.getDataFrame(prop, return_columns=True, **dframe)
  dprop = self.get_dprop(table, udprop=udprop)
  dprop = self.invert_dprop(dprop, [xp, yp], [xinvert, yinvert], exclude=iexclude)
  xfunc = var_name
  if isinstance(palette, str):
   palette = dprop[palette]['color'] if palette in dprop else None
  if hue is not None:
   xfunc = hue if hue_x else var_name
   hue   = var_name if hue_x else hue
   if palette is None and hue in dprop:
    palette = dprop[hue]['color']
   if hue_order is None and hue in dprop:
    hue_order = dprop[hue]['label']
   if order is None and xfunc in dprop:
    order = dprop[xfunc]['label']
   if hue_order is None:
    hue_order = columns
  if palette is None:
   palette = self.lcolor[:len(columns)] if lcolor is None else lcolor[:len(columns)]
  if order is None:
   order = columns
  #facetgrid = updateDictDefault(facetgrid, dict(gridspec_kws=dict(left=0.05, right=0.98, wspace=0.0, bottom=0.08, top=0.96, hspace=0.0), despine=False, legend_out=True, margin_titles=True))
  #g = sn.FacetGrid(data, col=col, row=row, col_order=self.dprops[table][col]['label'], row_order=self.dprops[table][row]['label'], **facetgrid) # Problem with tight_layout and including grispec_kws
  #g.map(sn.boxplot, var_name, value_name)
  #g.fig.subplots_adjust(right=0.8) # --> This is how you modify size axes
  xnbin, ynbin = dprop[xp]['nbin'], dprop[yp]['nbin']
  dplot = updateDictDefault(dplot, dict(xpad=3, ypad=2, left=0.04, right=0.96, top=0.95, bottom=0.08, hspace=0.0, wspace=0.0))
  fig, gs = self.set_plot(ny=ynbin, nx=xnbin, fig=fig, grid=grid, **dplot)
  dngal = OrderedDict()
  dax = OrderedDict() if dax is None else dax
  pfunc = sn.violinplot if violinplot else sn.boxplot
  gxydata = data.groupby(yp)
  dbox = updateDictDefault(dbox, dict(showfliers=False))
  dbox = self.set_boxplot_sns(dbox=dbox)
  min_ngal = self.min_ngal if min_ngal is None else min_ngal
  dkwargs = updateDictDefault(kwargs, dict(xlabel=False, legend=False, artist=False, legend_invisible=True))
  if ylim_guess:
   ylim = ylim if ylim is not None else (data[value_name].min(), data[value_name].max())
  for i, ypl in enumerate(dprop[yp]['label']):
   gxdata = gxydata.get_group(ypl).groupby(xp)
   for j, xpl in enumerate(dprop[xp]['label']):
    ax = mpl.pyplot.subplot(gs[i, j]) if not (i, j) in dax else dax[(i, j)]
    if xpl in gxdata.groups:
     pdata = gxdata.get_group(xpl)
     dcounts = pdata[xfunc].value_counts()
     counts = dcounts.sum() if xfunc in dprop else dcounts[0]
     if counts >= min_ngal:
      pdata = maskCategoricalDataFrame(pdata, xp, value_name, yp, z=z, nmin=min_ngal, percentile=percentile, low_per=low_per, upp_per=upp_per, dict_per=dict_per)
      ax = pfunc(x=xfunc, y=value_name, data=pdata, ax=ax, order=order, hue=hue, hue_order=hue_order, palette=palette, **dbox)
      dngal[(i,j)] = counts # len(pdata.index)
      if xfunc in dprop and hue is not None:
       boxcolor = [dprop[xfunc]['color'][k] for k in range(len(dprop[xfunc]['label'])) if dprop[xfunc]['label'][k] in pdata[xfunc].unique()]
       ax = self.set_boxplot_color(ax, boxcolor, groups=pdata[hue].unique().size, alphas=alphas, hatchs=hatchs, ialpha=ialpha)
      if not xfunc in dprop and hue is not None:
       ax = self.set_boxplot_color(ax, pdata[hue].unique().size, groups=pdata[xfunc].unique().size, alphas=alphas, hatchs=hatchs, ialpha=ialpha, hatch_invert=True, alpha_invert=True, legend=False)
      if hue is None and (alphas is not None or hatchs is not None):
       ax = self.set_boxplot_color(ax, len(columns), groups=pdata[xfunc].unique().size, alphas=alphas, hatchs=hatchs, ialpha=ialpha, legend=False)
      if (ngal and xygal is not None) and (hue in dprop or xfunc in dprop):
       kngal = hue if hue in dprop else xfunc
       dcbin = dict(pdata[kngal].value_counts())
       boxcolor = [dprop[kngal]['color'][k] for k in range(len(dprop[kngal]['label'])) if dprop[kngal]['label'][k] in pdata[kngal].unique()]
       bin_counts = [dcbin[key] for key in dprop[kngal]['label'] if key in pdata[kngal].unique()]
       multicolor_label(xygal[0], xygal[1], bin_counts, boxcolor, ax=ax, **updateDictDefault(ngal_prop, dict(ha='center left', fontsize=12, weight='bold')))
      if tngal and xytngal is not None:
       ax.text(xytngal[0], xytngal[1], self.fmt_ngal % counts, transform=ax.transAxes, **updateDictDefault(tngal_prop, dict(ha='left')))
     else:
      ax.axis('off')
      dngal[(i,j)] = 0
    else:
     ax.axis('off')
     dngal[(i,j)] = 0
    saxtitle  = self.setLabelUnitNgal(xpl, unit=dprop[xp]['unit'], ngal=dprop[xp]['bin_counts'][j], add_unit=add_unit, add_ngal=add_ngal, check_unit=check_unit)
    saxylabel = self.setLabelUnitNgal(ypl, unit=dprop[yp]['unit'], ngal=dprop[yp]['bin_counts'][i], add_unit=add_unit, add_ngal=add_ngal, check_unit=check_unit)
    # Searborn's boxplot adds a legend if there is hue, so need to explicitly make them invisible with legend_invisible
    ctitle = self.get_dprop_color(dprop, xp, idx=j, default=(not cxprop))
    cylabel = self.get_dprop_color(dprop, yp, idx=i, default=(not cyprop))
    dpticks = updateDictDefault(dkwargs, dict(dtitle=dict(color=ctitle), dylabel=dict(color=cylabel)))
    ax = self.setTicksLim(ax, xticklabels=xticklabels, yticklabels=yticklabels, saxtitle=saxtitle, axtitle=axtitle, ylim=ylim, sylabel=saxylabel, **dpticks)
    dax[(i,j)] = ax
    if grid is not None:
     fig.add_subplot(ax)
  self.set_dax_title_yx(dax, dngal, simple=simple, cor=cor)
  self.set_dax_ticklabels_xy(dax, dngal, gticks=gticks)
  dax = self.setDictAxesLim(dax, ylim=ylim, update_ylim=update_ylim)

  # Comparison plot
  if com_plot:
   ax_com = self.get_inset_plot(dax=dax, fig=fig, **dcom_plot)
   ax_com = pfunc(x=xfunc, y=value_name, data=data, ax=ax_com, order=order, hue=hue, hue_order=hue_order, palette=palette, **dbox)
   if tngal and xytngal is not None:
    dcounts = data[xfunc].value_counts()
    counts = dcounts.sum() if xfunc in dprop else dcounts[0]
    ax_com.text(xytngal[0], xytngal[1], self.fmt_ngal % counts, transform=ax_com.transAxes, **updateDictDefault(tngal_prop, dict(ha='left')))
   if xfunc in dprop and hue is not None:
    boxcolor = [dprop[xfunc]['color'][k] for k in range(len(dprop[xfunc]['label'])) if dprop[xfunc]['label'][k] in data[xfunc].unique()]
    ax_com = self.set_boxplot_color(ax_com, boxcolor, groups=data[hue].unique().size, alphas=alphas, hatchs=hatchs, ialpha=ialpha)
   if not xfunc in dprop and hue is not None:
    ax_com = self.set_boxplot_color(ax_com, data[hue].unique().size, groups=data[xfunc].unique().size, alphas=alphas, hatchs=hatchs, ialpha=ialpha, hatch_invert=True, alpha_invert=True, legend=False)
   if hue is None and (alphas is not None or hatchs is not None):
    ax_com = self.set_boxplot_color(ax_com, len(columns), groups=data[xfunc].unique().size, alphas=alphas, hatchs=hatchs, ialpha=ialpha, legend=False)
   if (ngal and xygal is not None) and (hue in dprop or xfunc in dprop):
    kngal = hue if hue in dprop else xfunc
    dcbin = dict(data[kngal].value_counts())
    boxcolor = [dprop[kngal]['color'][k] for k in range(len(dprop[kngal]['label'])) if dprop[kngal]['label'][k] in data[kngal].unique()]
    bin_counts = [dcbin[key] for key in dprop[kngal]['label'] if key in data[kngal].unique()]
    multicolor_label(xygal[0], xygal[1], bin_counts, boxcolor, ax=ax_com, **updateDictDefault(ngal_prop, dict(ha='center left', fontsize=12, weight='bold')))
   ax_com = self.setTicksLim(ax_com, **updateDictDefault(dcom_ticks, dict(xticklabels=xticklabels, yticklabels=yticklabels, ylim=ylim, sylabel=sylabel, 
	ylabelp='left', dylabel=updateDictDefault(dict(weight='normal'), dkwargs.get('dylabel',{})), **delKeyDict(dkwargs, 'dylabel', new=True))))
   dax['axcom'] = ax_com

  # Legend
  cw = self.lcolor[0] if cw is None else cw
  xlabels = dprop[xfunc]['label'] if xfunc in dprop else order
  if hue is None:
   xcolors = palette if palette is not None else self.lcolor[:len(columns)]
   xhatchs = None
   xalphas = None
  else:
   xcolors = dprop[xfunc]['color'] if xfunc in dprop else cw
   xhatchs = hatchs if not xfunc in dprop else None
   xalphas = alphas if not xfunc in dprop else None
  xtitle  = dprop[xfunc]['unit'] if xfunc in dprop else None
  if add_leg_x:
   axleg_x = self.get_axes_legend(axleg_x, dngal)
   leg_x = addCustomPatchLegend(mplText2latex(xlabels), xcolors, ax=dax[axleg_x], artist=True, hatchs=xhatchs, alphas=xalphas, **updateDictDefault(leg_x_prop, dict(ncol=2, fontsize=9, title=xtitle)))
   if leg_x_title_size is not None:
    leg_x.get_title().set_fontsize(leg_x_title_size)
  if hue is not None:
   hlabels = dprop[hue]['label'] if hue in dprop else hue_order
   hcolors = dprop[hue]['color'] if hue in dprop else cw
   htitle  = dprop[hue]['unit'] if hue in dprop else None
   hhatchs = hatchs if xfunc in dprop else None
   halphas = alphas if xfunc in dprop else None
   if add_leg_h:
    axleg_h = self.get_axes_legend(axleg_h, dngal, exclude=axleg_x)
    leg_h = addCustomPatchLegend(mplText2latex(hlabels), hcolors, ax=dax[axleg_h], artist=True, hatchs=hhatchs, alphas=halphas, **updateDictDefault(leg_h_prop, dict(ncol=2, fontsize=9, title=htitle)))
    if leg_h_title_size is not None:
     leg_h.get_title().set_fontsize(leg_h_title_size)

  sylabel = self.getLabel(prop) if sylabel is None else sylabel
  dflab = updateDictDefault(dflab, dict(xprop=dplot['left'], xinfo=dplot['right']))
  fig = self.fig_labels(fig, prop, grid=gs, sylabel=sylabel, figname=figname, info=info, pprop=pprop, **dflab)
  return fig, dax
    
 def createPlots(self, props=None, tables=None, error=True, pmask=False, finfo=None, radial_props=None,
	sep1='.fits', isep1=0, sep2='_', isep2=1, name_sep='_', plot_aps=True, alpha=0.3, invert=False, 
	dout=None, ftype='png', suffix=None, laps=None, include_info=False, table_info=False, 
	info=None, colormap='califa_int_r', dplots={}, lplots=None, dxyz=None, xyz_laps=None, 
	dxy=None, dx=None, x_laps=None, ldbox=None, **kwargs):
  if (self.dtables) == 0:
   print ('WARNING: NO tables available')
   return
  props = checkList(props)
  lplots = checkList(lplots)
  dplots = {} if dplots is None else dplots
  default_plots = ['radialPlot', 'agePropAperturePlot', 'ageAperturePropPlot', 'ageRadialPropPlot2D', 
	'ageXYAperturePlot', 'ageXApertureYPlot', 'ageApertureXYPlot', 'ageRadialXYPlot2D', 
	'xProp_XY_Plot', 'xProp_XYZ_Plot', 'xPropAndAperturePlot', 'propBoxPlot']
  if lplots is None:
   lplots = default_plots
  else:
   lplots = [item for item in lplots if item in default_plots]
   lplots.extend([item for item in dplots if item in default_plots and not item in lplots])
  suffix = suffix if suffix is not None and len(suffix) > 0 else None
  try:
   linfo = self.bfits.split(sep1)[isep1].split(sep2)[isep2:]
  except:
   linfo = []
  finfo = '_'.join(linfo) if finfo is None else finfo
  finfo = '%s%s' % (name_sep, finfo) if len(finfo) > 0 else ''
  finfo = '%s%s%s' % (finfo, name_sep, suffix) if suffix is not None else finfo
  info = self.info if self.info is not None else info
  if info is None and len(linfo) > 0 and include_info:
   info = ' | '.join(linfo)
  tinfo = info
  if tables is None:
   tables = self.dtables.keys()
  else:
   tables = checkList(tables)
   badtables = [table for table in tables if not table in self.dtables]
   tables = [table for table in tables if table in self.dtables]
   if len(badtables) > 0:
    print ('WARNING: Some tables names [%s] were NOT found in [%s]' % (' | '.join(badtables), ' | '.join(self.dtables.keys())))
   if len(tables) == 0:
    print ('NO Tables selected from [%s]' % ' | '.join(self.dtables.keys()))
    return
  radial_props = checkList(radial_props)
  for table in tables:
   if table_info:
    tinfo = '%s | %s' % (info, table) if info is not None else table
   nbin_props = len(self.dtables[table]._meta['dprop'])
   if radial_props is not None and nbin_props == 1 and 'radialPlot' in lplots:
    figname = joinPath('radial_%s%s.%s' % (table, finfo, ftype), dout)
    dplot = self.get_dplot(dplots, 'radialPlot', kwargs)
    self.radialPlot(table, radial_props, pmask=pmask, error=error, alpha=alpha, info=tinfo, figname=figname, **dplot)
   if props is not None:
    for prop in props:
     if nbin_props == 1:
      if 'agePropAperturePlot' in lplots:
       sprop = '%s__Apt' % prop
       figname = joinPath('%s_age_%s%s.%s' % (prop, table, finfo, ftype), dout)
       dplot = self.get_dplot(dplots, 'agePropAperturePlot', kwargs)
       self.agePropAperturePlot(table, sprop, pmask=pmask, error=error, alpha=alpha, info=tinfo, figname=figname, laps=laps, **dplot)

      if 'ageAperturePropPlot' in lplots:
       figname = joinPath('%s_age_Aperture_%s%s.%s' % (prop, table, finfo, ftype), dout)
       dplot = self.get_dplot(dplots, 'ageAperturePropPlot', kwargs)
       self.ageAperturePropPlot(table, sprop, pmask=pmask, error=error, alpha=alpha, info=tinfo, figname=figname, laps=laps, **dplot)

      if 'ageRadialPropPlot2D' in lplots:
       sprop = '%s__tr' % prop
       dplot = self.get_dplot(dplots, 'ageRadialPropPlot2D', kwargs)
       figname = joinPath('%s_age_Radial_%s_2D%s.%s' % (prop, table, finfo, ftype), dout)
       self.ageRadialPropPlot2D(table, sprop, invert=invert, plot_aps=plot_aps, figname=figname, info=tinfo, laps=laps, colormap=colormap, **dplot)

     elif nbin_props == 2:
      if 'ageXYAperturePlot' in lplots:
       sprop = '%s__Apt' % prop
       figname = joinPath('%s_age_%s_Aperture%s.%s' % (prop, table, finfo, ftype), dout)
       dplot = self.get_dplot(dplots, 'ageXYAperturePlot', kwargs)
       self.ageXYAperturePlot(table, sprop, invert=invert, alpha=alpha, figname=figname, laps=laps, info=tinfo, **dplot)
 
      if 'ageApertureXYPlot' in lplots:
       figname = joinPath('%s_age_ApertureX_%s%s.%s' % (prop, table, finfo, ftype), dout)
       dplot = self.get_dplot(dplots, 'ageApertureXYPlot', kwargs)
       self.ageApertureXYPlot(table, sprop, invert=invert, alpha=alpha, figname=figname, laps=laps, info=tinfo, **dplot)

      if 'ageXApertureYPlot' in lplots:
       figname = joinPath('%s_age_ApertureY_%s%s.%s' % (prop, table, finfo, ftype), dout)
       dplot = self.get_dplot(dplots, 'ageXApertureYPlot', kwargs)
       self.ageXApertureYPlot(table, sprop, invert=invert, alpha=alpha, figname=figname, laps=laps, info=tinfo, **dplot)

      if 'ageRadialXYPlot2D' in lplots:
       sprop = '%s__tr' % prop
       figname = joinPath('%s_age_Radial_%s_2D%s.%s' % (prop, table, finfo, ftype), dout)
       dplot = self.get_dplot(dplots, 'ageRadialXYPlot2D', kwargs)
       self.ageRadialXYPlot2D(table, sprop, invert=invert, plot_aps=plot_aps, alpha=alpha, figname=figname, info=tinfo, laps=laps, colormap=colormap, **dplot)

   if nbin_props == 1 and dx is not None and 'xPropAndAperturePlot' in lplots:
    for prop in dx:
     figname = joinPath('%s_AndAperture_%s%s.%s' % (prop, table, finfo, ftype), dout)
     dplot = self.get_dplot(dplots, 'xPropAndAperturePlot', kwargs)
     xprop = getattr(self, dx[prop], None) if isinstance(dx[prop], str) else dx[prop]
     if xprop is not None:
      self.xPropAndAperturePlot(table, xprop, prop, info=tinfo, figname=figname, laps=x_laps, **dplot)

   if nbin_props == 2 and dxy is not None and 'xProp_XY_Plot' in lplots:
    for prop in dxy:
     figname = joinPath('%s_XY_%s%s.%s' % (prop, table, finfo, ftype), dout)
     dplot = self.get_dplot(dplots, 'xProp_XY_Plot', kwargs)
     xprop = getattr(self, dxy[prop], None) if isinstance(dxy[prop], str) else dxy[prop]
     if xprop is not None:
      self.xProp_XY_Plot(table, xprop, prop, info=tinfo, figname=figname, **dplot)

   if nbin_props == 3 and dxyz is not None and 'xProp_XYZ_Plot' in lplots:
    for prop in dxyz:
     figname = joinPath('%s_XYZ_%s%s.%s' % (prop, table, finfo, ftype), dout)
     dplot = self.get_dplot(dplots, 'xProp_XYZ_Plot', kwargs)
     xprop = getattr(self, dxyz[prop], None) if isinstance(dxyz[prop], str) else dxyz[prop]
     if xprop is not None:
      self.xProp_XYZ_Plot(table, xprop, prop, info=tinfo, figname=figname, laps=xyz_laps, **dplot)

  if ldbox is not None and 'propBoxPlot' in lplots:
   for dbox in checkList(ldbox):
    btables = dbox.pop('tables')
    prop = dbox.pop('prop')
    figname = dbox.pop('figname', None)
    if figname is None:
     figname = joinPath('%s_BoxPlot_%s%s.%s' % (prop, '_'.join(checkList(btables)), finfo, ftype), dout)
    self.propBoxPlot(btables, prop, figname=figname, **dbox)

 def getTableStats(self, table, columns=None, tname=None, xinvert=False, yinvert=False, 
	invert=False, latex=False, fill_values=[('0', '0')]):
  if tname is not None:
   self.readDataTable(tname)
  if not table in self.dprops:
   print ('WARNING: Table "%s" NOT present! [%s]' % (table, ' | '.join(self.dprops.keys())))
   return
  dprop = self.dprops[table]
  if len(dprop.keys()) != 2:
   print ('WARNING: Table "%s" have %i variables (!= 2): [%s]' % (table, len(dprop.keys()), ' | '.join(dprop.keys())))
   return
  if columns is None:
   columns = dprop.keys()
  if invert:
   columns = columns[::-1]
  propx, propy = columns
  dprop = self.invert_dprop(dprop, [propx, propy], [xinvert, yinvert])
  lpropx = self.slab % (table, propx)
  lpropy = self.slab % (table, propy)
  xunit = dprop[propx]['unit'] if dprop[propx]['unit'] is not None else 'label'
  columns = [xunit]
  dt = {xunit: dprop[propx]['label'] + ['Total']}
  for y in dprop[propy]['label']:
   dt[y] = []
   columns.append(y)
   for x in dprop[propx]['label']:
    idxy = (self.t[lpropy] == y) & (self.t[lpropx] == x)
    dt[y].append(idxy.sum())
   dt[y].append((self.t[lpropy] == y).sum())
  lt = []
  for x in dprop[propx]['label']:
   lt.append((self.t[lpropx] == x).sum())
  lt.append(len(self.t))
  columns.append('Total')
  dt['Total'] = lt
  dt = Table(dt, masked=True)[columns]
  if latex:
   from astropy.io import ascii as asc
   dt = asc.write(dt, sys.stdout, format='latex', fill_values=fill_values)
  return dt
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def getPycassoTablePercentile(nfile, prop, bprop, tname='TABLE', func=None, slab='%s_%s_label', 
	low=25., high=75., tprint=True, ftype=None, key_dprops='dprops', masked=True, fmt=None, 
	float_fmt='.2f', simple=False, ptable=False, get_ptable=False, vfill=None, **kwargs):
 if nfile.endswith(('hdf5', 'h5')):
  data = readTableHDF5(nfile, tname, get_attr=True, masked=masked, **kwargs)
  dprops = data._meta[key_dprops]
 elif nfile.endswith(('fits', 'fit')):
  data, hdr = pyfits.getdata(nfile, extname=tname, header=True)
  dprops = ast.literal_eval(hdr[key_dprops.upper()])
 bins = dprops[bprop][bprop]['label']
 kbprop = slab % (bprop, bprop)
 dper = OrderedDict()
 if fmt is None:
  float_fmt = float_fmt if '%s' in float_fmt else '%' + float_fmt
  fmt = '%%s: %%s (%%s) --> %%s IQ (%%s, %%s): %s - %s (%s) [min: %s | max: %s | mean: %s | median: %s]' % ((float_fmt, ) * 7)
  fmt_simple = '%%s: %%s (%%s) --> %%s IQ: %s - %s (%s)' % ((float_fmt, ) * 3)
 for key in bins:
  idx = data[kbprop] == key
  bdata = data[prop][idx]
  if func is not None:
   bdata = func(bdata)
  iq = scipy.stats.iqr(bdata)
  percentile = np.percentile(bdata, [low, high])
  dper[key] = {'prop': prop, 'percentile': percentile, 'iq': iq, 'size': bdata.size, 'name': bprop, 'low_per': low, 'high_per': high, 'min': bdata.min(), 'max': bdata.max(), 'mean': bdata.mean(), 'median': np.ma.median(bdata)}
  if tprint:
   if simple:
    print (fmt_simple % (bprop, key, bdata.size, prop, percentile[0], percentile[1], iq))
   else:
    print (fmt % (bprop, key, bdata.size, prop, low, high, percentile[0], percentile[1], iq, dper[key]['min'], dper[key]['max'], dper[key]['mean'], dper[key]['median']))
 if ptable or get_ptable:
  import pandas as pd
  df = pd.DataFrame(dper).T
  if vfill is not None:
   df.fillna(vfill, inplace=True)
  if ptable:
   print(df)
  if get_ptable:
   dper = df
 return dper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def getIndexValue(ref_values, values, single=False, nearest=True, tol=0.05, sval='value', print_ref=False, iprint=None):
 ref_values = np.atleast_1d(ref_values)
 values = np.atleast_1d(values)
 if nearest:
  lvalues = np.array([np.argmin(np.abs(ref_values - val)) for val in values])
 else:
  lvalues = np.array([np.searchsorted(ref_values, val) for val in values])
 if tol is not None:
  idiff = np.abs(np.take(ref_values, lvalues) - values) >= tol
  if idiff.sum() > 0:
   for tval, ival, val in zip(values[idiff], lvalues[idiff], np.take(ref_values, lvalues)[idiff]):
    print ('>>> Difference bigger than tolerance %s for [Target %s %s | %s %s | Index %i]' % (tol, sval, tval, sval, val, ival))
    if print_ref:
     print ('    Reference values: %s' % ref_values)
     iprint = checkList(iprint)
     if iprint is not None:
      for line in iprint:
       print ('    %s' % line)
 if len(lvalues) == 1 and single:
  lvalues = lvalues[0]
 return lvalues
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def getPycassoTableIndexRadius(nfile, radius, key_dict='MDICT', key='bin_r', key_ap='bin_r_ap', ext=1, 
	apertures=False, single=False, nearest=True, tol=0.05, key_path='MDATA', path=None, 
	print_ref=True, mode='center', default_mode='center'):
 dmode = {'center': key_ap, 'in': '%s_in' % key_ap, 'out': '%s_out' % key_ap}
 dbinr = {'center': key, 'in': '%s_in' % key, 'out': '%s_out' % key}
 mode = mode if mode in dmode else default_mode
 key_ap = dmode.get(mode.lower(), key_ap)
 key_r  = dbinr.get(mode.lower(), key)
 if nfile.endswith(('hdf5', 'h5')):
  if path is None:
   path = getAttributeHDF5(nfile, path=path)[key_path]
  hdr = getAttributeHDF5(nfile, path=path)
  mdict = hdr[key_dict]
 elif nfile.endswith(('fits', 'fit')):
  hdr = pyfits.getheader(nfile, ext=ext)
  mdict = ast.literal_eval(hdr[key_dict])
 if apertures:
  bin_r = mdict[key_ap]
 else:
  bin_r = mdict[key_r]
 iprint = 'Mode "%s" [%s]' % (mode, (key_ap if apertures else key_r))
 lradius = getIndexValue(bin_r, radius, single=single, nearest=nearest, tol=tol, sval='radius', print_ref=print_ref, iprint=iprint)
 return lradius
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def getPycassoTableIndexAperture(nfile, aperture, tol=0.05, single=False, nearest=True, ext=1, 
	path=None, print_ref=True, key_dict='MDICT', key_meta='METADICT', key_path='MDATA', 
	key_bin='bin_apertures', key_meta_ap='set_bin_apertures', key_central='central', 
	key_integrated='integrated', mode='center', default_mode='center'):
 dmode = {'center': key_bin, 'in': '%s_in' % key_bin, 'out': '%s_out' % key_bin}
 mode = mode if mode in dmode else default_mode
 key_bin = dmode.get(mode.lower(), key_bin)
 if nfile.endswith(('hdf5', 'h5')):
  if path is None:
   path = getAttributeHDF5(nfile, path=path)[key_path]
  hdr = getAttributeHDF5(nfile, path=path)
  mdict = hdr[key_dict]
  metadict = hdr[key_meta]
 elif nfile.endswith(('fits', 'fit')):
  hdr = pyfits.getheader(nfile, ext=ext)
  mdict = ast.literal_eval(hdr[key_dict])
  metadict = ast.literal_eval(hdr[key_meta])
 bin_ap = np.atleast_1d(mdict[key_bin])
 iprint = ['Central: %s  |  Integrated: %s' % (metadict[key_meta_ap][key_central], metadict[key_meta_ap][key_integrated]), 'Mode "%s" [%s]' % (mode, key_bin)]
 lindex = getIndexValue(bin_ap, aperture, print_ref=print_ref, sval='aperture', iprint=iprint)
 if metadict[key_meta_ap][key_central]:
  lindex += 1
 return lindex
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def readTableSelection(table, nfile=None, dsel=None, dlow={}, dtop={}, props=None, 
	tprint=True, isort=0, csort=None, verbose=True):
 if isinstance(table, str):
  if nfile is None:
   print ('>>> Need to provide a "nfile" variable to read table "%s"' % table)
   return
  if nfile.endswith(('hdf5', 'h5')):
   data = readTableHDF5(nfile, table)
  elif nfile.endswith(('fits', 'fit')):
   data = pyfits.getdata(nfile, extname=table)
 else:
  data = table
 lmask = []
 linfo = []
 if dsel is not None and len(dsel) > 0:
  for key in dsel:
   kmask = data[key] == dsel[key]
   lmask.append(kmask)
   linfo.append('%s = %s (%i)' % (key, dsel[key], kmask.sum()))
 if dlow is not None and len(dlow) > 0:
  for key in dlow:
   kmask = data[key] < dlow[key]
   lmask.append(kmask)
   linfo.append('%s < %s (%i)' % (key, dlow[key], kmask.sum()))
 if dtop is not None and len(dtop) > 0:
  for key in dtop:
   kmask = data[key] > dtop[key]
   lmask.append(kmask)
   linfo.append('%s > %s (%i)' % (key, dtop[key], kmask.sum()))
 if len(lmask) > 0:
  mask = np.logical_and.reduce(lmask)
 else:
  mask = np.ones(len(data), dtype=np.bool)
 if props is None:
  return data[mask]
 else:
  props = checkList(props)
  dt = {}
  for prop in props:
   dt[prop] = data[prop][mask]
  dt = Table(dt)
  dt = dt[props]
  if isort is not None:
   dt.sort(props[isort])
  if csort is not None:
   dt.sort(csort)
  if tprint:
   dt.pprint(max_lines=-1, max_width=-1)
   print ('\n>>> Rows: %i' % len(dt))
  if verbose:
   print '\n'.join(linfo)
  return dt
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def getTableValue(nfile, keys, extname=None):
 import h5py
 if nfile.endswith(('hdf5', 'h5')):
  ftype = 'Group/Dataset'
  data = readHDF5(nfile, extname)
  columns = data.keys()
 elif nfile.endswith(('fits', 'fit')):
  data = pyfits.getdata(nfile, extname=extname)
  columns = data.columns.names
  ftype = 'HDU'
 keys = checkList(keys)
 lout = []
 for key in keys:
  if key in columns:
   lout.append(data[key])
  else:
   print ('WARNING: Key "%s" NOT in %s "%s" of file "%s"' % (key, ftype, extanme, nfile))
 if len(lout) == 0:
  lout = None
 if len(lout) == 1:
  lout = lout[0]
 return lout
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def substituteTablesValues(sub_files, ref_files, keys, refkey, extname=None,
        out_keys=None, intuple=False, replace=False, to_dict=False):
 keys = [refkey] + checkList(keys)
 rdata = getTableValue(ref_files, keys, extname=extname)
 sdata = getTableValue(sub_files, keys, extname=extname)
 rref, rdata  = rdata[0], rdata[1:]
 sref, sdata  = sdata[0], sdata[1:]
 idr   = np.in1d(rref, sref)
 ids   = np.in1d(sref, rref)
 for i in range(len(sdata)):
  if replace:
   sdata[i] = rdata[i][idr]
  else:
   sdata[i][ids] = rdata[i][idr]
 if intuple:
  out_keys = keys[1:] if out_keys is None else pc.checkList(out_keys)
  sdata = [(out_keys[i], sdata[i]) for i in range(len(sdata))]
  if to_dict:
   sdata = OrderedDict(sdata)
 return sdata
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def convolveSpectraFWHM(wave, spectra, fwhm_in, fwhm_out, dl=None, verbose=True):
 '''
 Reads a base spectrum of resolution fwhm_in (FHWM in Angs)) and smooth it to fwhm_out
 (also a FWHM in Angs). Smoothing done in LAMBDA!

 In case of STARLIGHT, this is needed because STARLIGHT fits apply a purely kinematical 
 filter, ie, one based in velocities. But the spectral bases are said to have their 
 spectral resolution homogenized in lamda space. Hence, a Dirac-delta 'line' at 4000 Angs 
 will have be observed as a gaussian of FWHM = 6 Angs, while a Dirac-delta at 8000 Angs 
 will **also** be a gaussian of FWHM = 6 Angs.

 In velocity, FWHM = 6 Angs at 4000 Angs correspond to 450 km/s (or a vd of 191 km/s), bult at
 8000 Angs the velocity broadening would be half: 225 km/s in FWHM (or 96 km/s). Hence, using a
 purely kinematical filter, the blue would pull vd towards large values while the red would
 pull it to lower values!

 The solution to analyse homogenized spectra like CALIFA (FWHM = 6Angs = constant with lambda) 
 is to degrade the spectral resoltion of the base to the same FWHM = 6 Angs. This way the vd 
 outputed by STARLIGHT will be the true one, fixing this conceptual mismatch and circumventing 
 the need for a posteriori corrections on vd for instrumental effects and possible differences 
 between the resolution of the base and observed spectra.

 Here, we are assuming a constant wavelength step!
 '''
 sigma2fwhm = 2 * np.sqrt(2 * np.log(2))
 # FWHM in Angstroms
 fwhm_diff = np.sqrt(fwhm_out**2 - fwhm_in**2)
 if dl is None:
  dl = np.unique(np.diff(wave))
  if dl.size > 1 and verbose:
   print ('>>> Wavelength step not unique. The minimum step is chosen: %.1f (%s)' % (dl.min(), str(dl)))
  dl = dl.min()
 # Sigma difference in pixels
 sigma = fwhm_diff / sigma2fwhm / dl
 return gaussian_filter1d(spectra, sigma)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def getVelscale(wave, c=299792.458):
 '''VELSCALE: velocity scale in km/s per pixels'''
 nlam = wave.size
 lamRange = np.array([wave.min(), wave.max()])
 dLam = np.diff(lamRange)/(nlam - 1.)     # Assume constant dLam
 lim = lamRange/dLam + [-0.5, 0.5]        # All in units of dLam
 logLim = np.log(lim)
 return np.diff(logLim) * c / nlam
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def resampleConvolveSpec(wave,flux,file_out=None,file_in=None,fwhm_in=None,fwhm_out=None,
	wmin=None,wmax=None,dw=None,dirout=None,base=None,age=None,age_units=None,
	get_header=True,header=True,lheader=None,verbose=True,left=0.0,right=0.0,
	fmt='%.1f %.8e',fmt_date='RGB@IAA - %s'):

 # Convolve
 if fwhm_in is not None and fwhm_out is not None:
  flux = convolveSpectraFWHM(wave, flux, fwhm_in, fwhm_out, verbose=verbose)

 # Resample
 if wmin is not None or wmax is not None or dw is not None:
  if wmin is None:
   wmin = wave.min()
  if wmax is None:
   wmax = wave.max()
  if dw is None:
   dw = np.diff(wave).mean()
  rwave = np.arange(wmin, wmax + dw/2., dw)
  rflux  = np.interp(rwave, wave, flux, left=left, right=right)
 else:
  rwave = wave
  rflux = flux

 if dirout is not None and file_out is not None:
  file_out = os.path.join(dirout, os.path.basename(file_out))

 hdr = ''
 # Build header
 if get_header:
  lhdr = []
  if header:
   input_dw = np.unique(np.diff(wave)).mean()
   if fmt_date is not None:
    today = datetime.date.today()
    today = '%i/%i/%i' % (today.day,today.month,today.year)
    lhdr.append(fmt_date % today)
   if file_in is not None:
    lhdr.append('Input file =  %s' % os.path.basename(file_in))
   if fwhm_in is not None:
    lhdr.append('Input Spectral resolution: %s (A) FWHM' % fwhm_in)
   if fwhm_out is not None:
    lhdr.append('Output Spectral resolution: %s (A) FWHM' % fwhm_out)
   if dw is not None and input_dw != dw:
    lhdr.append('Resampled from %s A to %s A' % (input_dw, dw))
   if age is not None:
    age_units = '' if age_units is None else '(%s)' % age_units
    lhdr.append('Age %s = %s' % (age_units, age))   
   if base is not None:
    lhdr.append('Input Base =  %s' % base) 
   lhdr.append('Lambda (A) & L_lambda [L_sun/A/M_sun]')
  if lheader is not None:
   if not isinstance(lheader, list):
    lheader = [lheader]
   lhdr.extend(lheader)
  hdr = '\n'.join(lhdr)

 if file_out is not None:
  np.savetxt(file_out, np.column_stack((rwave, rflux)), fmt=fmt, header=hdr)

 if get_header:
  return rwave, rflux, lhdr
 else:
  return rwave, rflux
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class SSPLibrary2StarlightBase(object):

 def __init__(self, lfiles_in, dout=None, suffix=None, wmin=None, wmax=None, dw=None, fwhm_in=None, 
	fwhm_out=None, fmt='%.1f  %.6e', left=0.0, right=0.0, lfiles_out=None, get_header=True, 
	header=True, lheader=True, file_str_remove='.fits', 
	fmt_date='RGB@IAA - %s', verbose=True):

  lfiles_out = self.getInputOutputFiles(lfiles_in, lfiles_out=lfiles_out, dout=dout, suffix=suffix, file_str_remove=file_str_remove)

  for file_in, file_out in zip(lfiles_in, lfiles_out):
   self.resampleConvolveSpec(file_in, file_out, wmin=wmin, wmax=wmax, dw=dw, fwhm_in=fwhm_in, 
  	fwhm_out=fwhm_out, fmt=fmt, left=left, right=right, get_header=get_header, header=header,
	lheader=lheader, fmt_date=fmt_date, verbose=verbose)

 def getInputOutputFiles(self, lfiles_in, lfiles_out=None, dout=None, suffix=None, file_str_remove=None):
  if not isinstance(lfiles_in, list):
   lfiles_in = [lfiles_in]

  if lfiles_out is None:
   lfiles_out = lfiles_in
   if suffix is None and (dout is None or (dout is not None and dout == os.path.dirname(lfiles_in[0]))):
    sys.exit('>>> You need to specify a SUFFIX if a list with output names is NOT provided!')

  if lfiles_out is not None:
   if not isinstance(lfiles_out, list):
    lfiles_out = [lfiles_out]
   if len(lfiles_in) != len(lfiles_out):
    sys.exit('>>> Length of input (%i) and output (%i) files is different!' % (len(lfiles_in), len(lfiles_out)))

  if suffix is not None:
   lfiles_out = ['%s.%s' % (file_out, suffix) for file_out in lfiles_out]

  if dout is not None:
   lfiles_out = [os.path.join(dout, os.path.basename(file_out)) for file_out in lfiles_out]

  if file_str_remove is not None:
   lfiles_out = [file_out.replace(file_str_remove,'') for file_out in lfiles_out]

  return lfiles_out

 def readMilesFits(self, fits):
  flux, hdr = pyfits.getdata(fits, header=True)
  w0 = hdr['CRVAL1']
  dw = hdr['CDELT1']
  wave = w0 + dw * np.arange(flux.size)
  lhdr =  self.getMilesFitsHeader(hdr)
  return wave, flux, lhdr

 def getMilesFitsHeader(self, hdr, skey='MODEL INGREDIENTS', hkey='COMMENT'):
  lhdr = None
  if hkey in hdr:
   shdr = str(hdr[hkey]).replace('=','').replace('#','').replace("'",'')
   if skey in shdr:
    shdr = shdr[shdr.find(skey):]
   lhdr = shdr.split('\n')
   lhdr = [line.strip() for line in lhdr if len(line.strip()) > 0]
  return lhdr

 def readASCII(self, filetxt, metakey='comments'):
  t = astread(filetxt)
  lhdr = self.getMilesASCIIHeader(t.meta[metakey]) if metakey in t.meta else None
  return t['col1'], t['col2'], lhdr

 def getMilesASCIIHeader(self, lhdr, skey='MODEL INGREDIENTS'):
  shdr = '\n'.join(lhdr)
  if skey in shdr:
   lhdr = shdr[shdr.find(skey):].split('\n')
  lhdr = [line.strip() for line in lhdr if len(line.strip()) > 0]
  return lhdr

 def getFWHM(self, lhdr, skey='Spectral resolution:'):
  try:
   return float(' '.join(lhdr).split(skey)[1].split()[0])
  except:
   return None

 def resampleConvolveSpec(self, file_in, file_out, wmin=None, wmax=None, dw=None, 
	fwhm_in=None, fwhm_out=None, fmt='%.1f  %.6e', left=0.0, right=0.0, 
	get_header=True, header=True, lheader=True, verbose=True,
	fmt_date='RGB@IAA - %s'):
  bfile_in =  os.path.basename(file_in)

  try:
   wave, flux, lhdr = self.readMilesFits(file_in)
  except:
   wave, flux, lhdr = self.readASCII(file_in)

  fwhm_file = self.getFWHM(lhdr)

  if fwhm_in is not None and fwhm_file is not None and fwhm_in != fwhm_file:
   print ('>>> WARNING: Input FWHM (%s) and file FWHM (%s) NOT equal! ("%s")' % (fwhm_in, fwhm_file, bfile_in))

  if fwhm_in is None:
   fwhm_in = fwhm_file

  if fwhm_out is not None and fwhm_in is None:
   sys.exit('>>> Could NOT find Spectral resolution (FWHM) in file "%s"!' % bfile_in)

  lhdr = lhdr if lheader else None

  resampleConvolveSpec(wave, flux, file_out, file_in=bfile_in, fwhm_in=fwhm_in, fwhm_out=fwhm_out, wmin=wmin, 
	wmax=wmax, dw=dw, get_header=get_header, header=header, lheader=lhdr, verbose=True, fmt=fmt, 
	left=left, right=right, fmt_date=fmt_date)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class getMassHLR(object):
 def __init__(self,fits,table=None,figname=None,savefig=False,dfig=None,show=False,basic=False,tprint=False,
              fill_value=0.0,plot_mask=False,dim_mask={0.0: np.nan}):
  self.nfits = fits
  self.rfits = os.path.basename(fits)
  self.table = table
  self.fill_value = fill_value
  self.K = fitsDataCube(fits)
  self.shape = self.K.qZones.shape
  self.galaxyName = self.K.header['PNAME']
  self.kalifaID = self.K.header['PREFIX']
  self.califaID = int(self.K.header['PREFIX'].replace('K',''))
  self.otable = None
  self.t      = None
  self.mask   = None
  self.figname = figname
  self.external_ellipse_params = 0 if table is None else 1
  if figname is None:
   self.figname = '%s_Ellipse.png' % self.galaxyName
  if dfig is not None:
   self.figname = os.path.join(dfig,os.path.basename(self.figname))

  self.getTableProp(table)
  self.getMassProp()
  self.getEllipse()
  self.getMass(basic=basic)
  if tprint:
   self.printMass()
  if show or savefig:
   self.plotEllipse(show=show,savefig=savefig,plot_mask=plot_mask,dim_mask=dim_mask)

 def getMassProp(self):
  self.mass2D_SD               = self.K.zoneToYX(self.K.Mcor__z, extensive=True, surface_density=True, fill_value=self.fill_value) * self.K.parsecPerPixel**2
  self.mass2D_SD_filled        = self.K.fillImage(self.mass2D_SD, mode='hollow')
  self.mass_per_pix            = self.K.Mcor__z/self.K.zoneArea_pix
  # Extensive = True --> Weight with qSignal instead of zoneArea == mass2D_SD
  self.mass_per_pix__yx        = self.K.zoneToYX(self.mass_per_pix,extensive=False, surface_density=False,fill_value=self.fill_value)
  self.mass_per_pix__yx_filled = self.K.fillImage(self.mass_per_pix__yx, mode='hollow')

  self.integrated_mass       = self.K.integrated_Mcor
  self.total_mass            = self.K.Mcor__z.sum()
  self.total_mass_SD         = self.mass2D_SD.sum()
  self.total_mass_SD_filled  = self.mass2D_SD_filled.sum()
  self.total_mass__yx        = self.mass_per_pix__yx.sum()
  self.total_mass__yx_filled = self.mass_per_pix__yx_filled.sum()

 def getMass(self,basic=True):
  if basic is None:
   lbasic = [True,False]
  else:
   lbasic = [basic]
  for basic in lbasic:
   self.setEllipse(basic=basic)
   self.getEllipseMask()
   self.getMassRegion(basic=basic)

 def getMassRegion(self,basic=True):
  if self.mask is not None:
   attr = 'basic_' if basic else ''
   setattr(self,'%s%s' % (attr,'mass_SD_region'), self.mass2D_SD[self.mask].sum())
   setattr(self,'%s%s' % (attr,'mass_SD_region_filled'), self.mass2D_SD_filled[self.mask].sum())
   setattr(self,'%s%s' % (attr,'mass_region'), self.mass_per_pix__yx[self.mask].sum())
   setattr(self,'%s%s' % (attr,'mass_region_filled'), self.mass_per_pix__yx_filled[self.mask].sum())

 def printMass(self):
  lkeys = ['total_mass','integrated_mass','region']
  for key in self.__dict__:
   if any([subkey in key for subkey in lkeys]):
    print '%25s: %g' % (key, self.__dict__[key])

 def getTableProp(self, table=None):
  if table is None:
   table = self.table
  if table is None:
   return
  idx = table['name'] == self.galaxyName
  if table['name'][idx].size < 1:
   print ('Empty table for object "%s" (cube: "%s")' % (self.galaxyName,self.rfits))
   self.external_ellipse_params = 0
   return
  self.otable = table[idx]
  self.t = lambda: None # Empty object-function
  for key in self.otable.colnames:
   setattr(self.t,key,self.otable[key].data[0])

 def getEllipse(self):
  self.qSignal_fill, self.mask_fill = self.K.fillImage(self.K.qSignal, mode='hollow')
  # PA: Position angle in radians, counter-clockwise relative to the positive X axis
  pa_rad, ba = self.K.getEllipseParams(self.qSignal_fill,mask=self.mask_fill)
  self.K.setGeometry(pa_rad, ba)
  self.HLR_pix_fill  = self.K.getHLR_pix(fill=True)
  self.HLR_pix       = self.K.getHLR_pix(fill=False)
  self.basic_pa_rad  = pa_rad
  self.basic_pa      = 180. * pa_rad / np.pi
  self.basic_ba      = ba
  self.table_HLR     = self.t.Reff_ell if self.t is not None else None
  self.table_ba      = self.t.EPS_OUT if self.t is not None else None
  self.table_pa      = 90.0 + self.t.PA_OUT if self.t is not None else None
  self.table_pa_rad  = self.table_pa * np.pi / 180. if self.t is not None else None
  #print 'PA: %f | PA_OUT: %f | ba: %f | BA_OUT: %f' % (self.basic_pa,self.t.INCL,ba,self.t.EPS_OUT)
  #print 'REF: %f | HLR: %f | HLR_fill: %f' % (self.t.Reff_ell,self.K.HLR_pix, self.HLR_pix_fill)

 def setEllipse(self,basic=True):
  if self.t is None and not basic:
   print ('*** Forced "basic = True" since galaxy "%s" is NOT present in external table ***' % self.galaxyName)
   basic = True
   self.external_ellipse_params = 0
  if basic:
   self.pa_rad = self.basic_pa_rad
   self.pa     = self.basic_pa
   self.ba     = self.basic_ba
   self.major  = self.HLR_pix_fill
   self.minor  = self.major*self.ba
  else:
   self.pa_rad = self.table_pa_rad
   self.pa     = self.table_pa
   self.ba     = self.table_ba
   self.major  = self.table_HLR
   self.minor  = self.major*self.ba
  self.K.setGeometry(self.pa_rad, self.ba, HLR_pix=self.major)

 def getEllipseMask(self):
  import pyregion
  sregion = 'image;ellipse(%f,%f,%f,%f,%f)' % (self.K.x0+1,self.K.y0+1,self.major,self.minor,self.pa)
  region = pyregion.parse(sregion)
  self.mask = region.get_mask(shape=self.shape)

 def plotEllipse(self,figname=None,color='y',bcolor=None,show=False,savefig=False,plot_mask=False,dim_mask=None):
  from matplotlib.patches import Ellipse
  import matplotlib.pyplot as plt
  if figname is None:
   figname = self.figname
  if bcolor is None:
   bcolor = color
  fig = plt.figure()
  ax = fig.add_subplot(111)
  image = self.K.qSignal
  if dim_mask is not None:
   for key in dim_mask:
    image[image == key] = dim_mask[key]
  if self.mask is not None and plot_mask:
   image[self.mask] = np.nan
  cax = ax.imshow(image,interpolation='nearest')
  cbar = fig.colorbar(cax)
  # Basic Ellipse
  semi_major = self.HLR_pix_fill
  semi_minor = semi_major*self.basic_ba
  major = 2.*semi_major
  minor = 2.*semi_minor
  angle = self.basic_pa
  color = bcolor
  self.ellip_basic = Ellipse(xy=(self.K.x0,self.K.y0),width=major,height=minor,angle=angle,fill=False,linewidth=3,ls='dashed',color=color)
  ax.add_artist(self.ellip_basic)
  # Computed Ellipse
  semi_major = self.table_HLR
  semi_minor = semi_major*self.table_ba
  major = 2.*semi_major
  minor = 2.*semi_minor
  angle = self.table_pa
  color = color
  self.ellip = Ellipse(xy=(self.K.x0,self.K.y0),width=major,height=minor,angle=angle,fill=False,linewidth=3,color=color)
  ax.add_artist(self.ellip)
  ax.legend([self.ellip_basic,self.ellip],['Pycasso','Table'])
  ax.set_title(self.galaxyName,fontsize=20)
  if figname is not None and savefig:
   fig.savefig(figname)
  if show:
   plt.show()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class tableMassHLR(object):

 def __init__(self,lfits,dprop,lkeys,ntable,verbose=True,tfmt='ascii.fixed_width',
	fmt='%.6g',delimiter=None,**kwargs):

  self.table = None
  for i,fits in enumerate(lfits,1):
   if verbose:
    print ('>>> %i/%i (Superfits: %s)' % (i,len(lfits),os.path.basename(fits)))
   cobj = getMassHLR(fits,**kwargs)
   dprop = self.fillDict(dprop,cobj,lkeys)

  self.table = self.writeTable(dprop,lkeys=lkeys,ntable=ntable,tfmt=tfmt,fmt=fmt,delimiter=delimiter)

 def fillDict(self,dprop,obj,lkeys):
  for key in lkeys:
   if key in dprop and key in obj.__dict__:
    dprop[key].append(getattr(obj,key))
  return dprop

 def writeTable(self,dprop,lkeys=None,ntable=None,fmt='%.6g',tfmt='ascii.fixed_width',delimiter=None):
  dprop = {key: dprop[key] for key in dprop if len(dprop[key]) > 0}
  table = Table(dprop)
  if lkeys is not None:
   lkeys = [key for key in lkeys if key in dprop]
   table = table[lkeys]
  for key in table.colnames:
   if isinstance(table[key].data[0],np.float):
    table[key].format = fmt
  if ntable is not None:
   table.write(ntable,format=tfmt,delimiter=delimiter)
  return table
# ---------------------------------------------------------------------------
