##Abundance Matching

This repository contains code to do galaxy abundance matching. 

Warning all code in this repository is a work in progress.  There are bugs, and things 
are rapidly changing.

AM.py contains code to non-parametrically solve for the stellar mass halo 
mass(SMHM) relation.  AM.AM() takes a galaxy abundance function, a halo abundance 
function, and a scatter model, and calculates the stellar mass halo mass relation that 
reproduces the input galaxy abundance function.

abundance.py contains code to calculate and fit abundance functions.

SMHM_model.py contains objects to parametrize the SMHM relation.

make_mocks.py contains code to 'paint' galaxies onto a halo catalogue given P(gal | halo).

&copy; Copyright 2015, Duncan Campbell