'''
Created on 24/06/2015

@author: andre
'''

from ConfigParser import ConfigParser

__all__ = ['default_config_path', 'get_config', 'save_config']

default_config_path = 'data/diving3d.cfg'

def get_config(configfile='diving3d.cfg'):
    config = ConfigParser()
    config.read(configfile)
    return config

def save_config(cfg, configfile):
    with open(configfile, 'w') as cfp:
        cfg.write(cfp)
