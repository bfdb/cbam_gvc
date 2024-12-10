# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:28:31 2024

@author: bfdeboer
"""

import cfg
import utils as ut

# Parse the raw text version of EXIOBASE.
d_eb = ut.parse_eb_raw_txt()

# Calculate the Leontief inverse.
d_eb[cfg.s_li] = ut.calc_li(d_eb)

# Dump processed version of EXIOBASE to pickle.
ut.dump_eb_proc_pkl(d_eb)
