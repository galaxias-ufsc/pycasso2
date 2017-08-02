'''
Created on 24/06/2015

@author: andre
'''

import sys
if sys.version[0] == '2':
    from ConfigParser import SafeConfigParser as ConfigParser
else:
    from configparser import ConfigParser


__all__ = ['default_config_path', 'get_config', 'save_config']

default_config_path = 'pycasso.cfg'


def get_config(configfile='pycasso.cfg'):
    config = ConfigParser()
    config.read(configfile)
    return config


def save_config(cfg, configfile):
    with open(configfile, 'w') as cfp:
        cfg.write(cfp)


def parse_slice(sl):
    if sl is None:
        return None
    yy, xx = sl.split(',')
    y1, y2 = yy.split(':')
    y1 = int(y1)
    y2 = int(y2)
    x1, x2 = xx.split(':')
    x1 = int(x1)
    x2 = int(x2)
    assert(x1 < x2)
    assert(y1 < y2)
    return slice(y1, y2, 1), slice(x1, x2, 1)
