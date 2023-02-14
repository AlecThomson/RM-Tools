#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Logging module for RM-Tools"""

import logging

# Create logger
logging.captureWarnings(True)
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

# Create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)


# Create formatter
formatter = logging.Formatter("%(levelname)s %(module)s - %(funcName)s: %(message)s")

# Add formatter to ch
ch.setFormatter(formatter)

# Add ch to logger
logger.addHandler(ch)
