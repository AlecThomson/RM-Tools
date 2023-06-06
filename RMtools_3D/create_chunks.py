#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 13:25:30 2019

This code will divide a FITS cube into individual chunks.
To minimize problems with how to divide the cube, it will convert the image
plane into a 1D list of spectra.
Then the file will divided into smaller files, with fewer pixels, in order to
be run through RM synthesis and CLEAN.
A separate routine will re-assemble the individual chunks into a combined file again.
This code attempts to minimize the memory profile: in principle it should never
need more memory than the size of a single chunk, and perhaps not even that much.

Divide a FITS cube into small pieces for memory-efficient RM synthesis.
Files will be created in running directory.
WARNING: ONLY WORKS ON FIRST HDU, OTHERS WILL BE LOST.

@author: cvaneck
May 2019
"""

import numpy as np
import argparse
from astropy.io import fits
import os.path as path
from math import ceil, floor, log10
from tqdm.auto import trange, tqdm
import multiprocessing as mp

def cli():

    parser = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("infile", metavar="filename.fits",
                        help="FITS cube containing data")
    parser.add_argument("Nperchunk", metavar="N_pixels",
                        help="Number of pixels per chunk", type=int)
    parser.add_argument("-v", dest="verbose", action="store_true",
                        help="Verbose [False].")
    parser.add_argument("-p", dest="prefix", default=None,
                        help="Prefix of output files [filename]")

    args = parser.parse_args()

    Nperchunk=int(args.Nperchunk)

    if not path.exists(args.infile):
        raise Exception('Input file not found!')

    prefix = path.splitext(args.infile)[0] if not args.prefix else args.prefix

    return main(
        infile=args.infile,
        Nperchunk=Nperchunk,
        verbose=args.verbose,
        prefix=prefix,
    )




def _worker(i, prefix, prntcode, split, new_header, pbar):
    """This is the worker function for the multiprocessing version of the code.
    It is a separate function so that it can be pickled.
    """
    filename=('{}.C{'+prntcode+'}.fits').format(prefix,i)
    fits.writeto(filename, split, new_header,overwrite=True)
    pbar.update(1)

def main(
    infile: str,
    Nperchunk: int,
    verbose: bool = False,
    prefix: str = None,
):
    with fits.open(infile,memmap=True, mode="denywrite") as hdul:
        hdu = hdul[0]
        header = hdu.header
        data = hdu.data

    x_image=header['NAXIS1']
    y_image=header['NAXIS2']
    Npix_image=x_image*y_image

    num_chunks=ceil(Npix_image/Nperchunk)
    digits=floor(log10(num_chunks))+1
    prntcode = f":0{digits}d"

    if verbose:
        print(('Chunk name set to "{}.C{'+prntcode+'}.fits"').format(prefix,0))
        print('File will be divided into {} chunks'.format(num_chunks))

    base_idx_arr=np.arange(Nperchunk)

    new_header=header.copy()
    new_header['NAXIS2']=1
    new_header['NAXIS1']=Nperchunk
    new_header['OLDXDIM']=x_image
    new_header['OLDYDIM']=y_image

    splits = np.array_split(data, num_chunks, axis=-1)

    for i, split in enumerate(tqdm(splits, desc="Writing chunks", disable=not verbose)):
        filename=('{}.C{'+prntcode+'}.fits').format(prefix,i)
        fits.writeto(filename, split, new_header,overwrite=True)


if __name__ == "__main__":
    cli()


