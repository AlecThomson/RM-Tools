#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:10:26 2019

This code reassembles chunks into larger files. This is useful for assembling
output files from 3D RM synthesis back into larger cubes.

@author: cvaneck
"""

import argparse
import asyncio
import os.path as path
import re
from datetime import datetime
from glob import glob
from math import ceil, floor, log10
from typing import Coroutine, List, Tuple

import numpy as np
from astropy.io import fits
from tqdm.asyncio import tqdm, trange


def main():
    """This function will assemble a large FITS file or cube from smaller chunks."""

    descStr = """
    Assemble a FITS image/cube from small pieces. The new image will be created
    in the running directory.
    Supply one of the chunk files (other files will be identified by name pattern).
    Output name will follow the name of the input chunk, minus the '.C??.'
    """

    parser = argparse.ArgumentParser(
        description=descStr, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "chunkname", metavar="chunk.fits", help="One of the chunks to be assembled"
    )
    parser.add_argument(
        "-f",
        dest="output",
        default=None,
        help="Specify output file name [basename of chunk]",
    )
    parser.add_argument(
        "-o",
        dest="overwrite",
        action="store_true",
        help="Overwrite existing file? [False].",
    )

    args = parser.parse_args()

    if args.output == None:
        output_filename = ".".join(
            [
                x
                for x in args.chunkname.split(".")
                if not (x.startswith("C") and x[1:].isnumeric())
            ]
        )
    else:
        output_filename = args.output

    asyncio.run(
        assemble(
            chunkname=args.chunkname,
            output_filename=output_filename,
            overwrite=args.overwrite,
        )
    )


async def worker(
    i: int,
    chunk: np.ndarray,
    large_hdul: fits.HDUList,
    startx: int,
    stopx: int,
    starty: int,
    stopy: int,
) -> Coroutine[None]:
    print(f"{datetime.utcnow()} Worker {i} starting...")
    await asyncio.to_thread(
        update_and_write, large_hdul, chunk, startx, stopx, starty, stopy
    )
    print(f"{datetime.utcnow()} Worker {i} done")


def update_and_write(
    large_hdul: fits.HDUList,
    chunk: np.ndarray,
    startx: int,
    stopx: int,
    starty: int,
    stopy: int,
) -> None:
    large_hdul[0].data[
        ...,
        starty:stopy,
        startx:stopx,
    ] = chunk
    large_hdul.flush(verbose=True)


async def assemble(
    chunkname: str,
    output_filename: str,
    overwrite: bool = False,
) -> Coroutine[None]:
    # Get all the chunk filenames. Missing chunks will break things!
    filename = re.search("\.C\d+\.", chunkname)
    chunkfiles = glob(
        chunkname[0 : filename.start()] + ".C*." + chunkname[filename.end() :]
    )
    chunkfiles.sort()

    old_header = fits.getheader(chunkfiles[0])
    x_dim = old_header["OLDXDIM"]
    y_dim = old_header["OLDYDIM"]
    Nperchunk = old_header["NAXIS1"]
    Npix_image = x_dim * y_dim
    num_chunks = ceil(Npix_image / Nperchunk)
    Ndim = old_header["NAXIS"]

    if (Ndim != 4) and (Ndim != 3) and (Ndim != 2):
        raise Exception(
            "Right now this code only supports FITS files with 2-4 dimensions!"
        )

    new_header = old_header.copy()
    del new_header["OLDXDIM"]
    del new_header["OLDYDIM"]
    new_header["NAXIS1"] = x_dim
    new_header["NAXIS2"] = y_dim

    # Create blank file:
    new_header.tofile(output_filename, overwrite=overwrite)

    # According to astropy, this is how to create a large file without needing it in memory:
    shape = tuple(
        new_header["NAXIS{0}".format(ii)] for ii in range(1, new_header["NAXIS"] + 1)
    )
    with open(output_filename, "rb+") as fobj:
        fobj.seek(
            len(new_header.tostring())
            + (np.product(shape) * np.abs(new_header["BITPIX"] // 8))
            - 1
        )
        fobj.write(b"\0")

    chunks = []
    for f in chunkfiles:
        chunks.append(fits.getdata(f, memmap=True))

    tasks = []
    with fits.open(output_filename, mode="update", memmap=True) as large_hdul:
        large = large_hdul[0]
        large_data = large.data
        large_shape = large_data.shape
        print(f"{large_shape=}")
        y_full = False
        x_full = False
        startx = 0
        starty = 0
        for i, chunk in enumerate(chunks):
            chunk_shape = chunk.shape
            # Calculate the start and stop indices for the chunk
            stopx = startx + chunk_shape[-1]
            stopy = starty + chunk_shape[-2]
            # Fill in the large array with the chunk data
            print(f"{starty}:{stopy}")
            print(f"{startx}:{stopx}")
            task = asyncio.create_task(
                worker(i, chunk, large_hdul, startx, stopx, starty, stopy)
            )
            # Check if the chunk filled the x or y dimension
            x_full = stopx == large_shape[-1]
            y_full = stopy == large_shape[-2]
            # Update the start indices for the next chunk
            startx = 0 if x_full else stopx
            starty = 0 if y_full else stopy

            tasks.append(task)

        await tqdm.gather(*tasks)


if __name__ == "__main__":
    main()
