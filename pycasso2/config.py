'''
Created on 24/06/2015

@author: andre
'''


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
