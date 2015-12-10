'''
Created on 22/06/2015

@author: andre
'''

_unused = 0x0001
no_data = 0x0002
bad_pix = 0x0004
ccd_gap = 0x0008
telluric = 0x0010

d3d_screw = 0x1000

starlight_masked_pix = 0x0100
starlight_failed_run = 0x00200

# Compound flags
no_obs = no_data | bad_pix | ccd_gap | telluric | d3d_screw
no_starlight = starlight_masked_pix | starlight_failed_run
