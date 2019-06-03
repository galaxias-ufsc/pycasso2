'''
Created on 22/06/2015

@author: andre
'''

_unused = 0x0001
no_data = 0x0002
bad_pix = 0x0004
ccd_gap = 0x0008
telluric = 0x0010
seg_has_badpixels = 0x0020
low_sn = 0x0040
overlapping_spaxel = 0x0080

starlight_masked = 0x0100
starlight_failed_run = 0x0200
starlight_no_data = 0x0400
starlight_clipped = 0x0800

d3d_screw = 0x1000

# Compound flags
no_obs = no_data | bad_pix | ccd_gap | d3d_screw | overlapping_spaxel
before_starlight = no_obs | telluric | low_sn
no_starlight = starlight_no_data | starlight_failed_run
