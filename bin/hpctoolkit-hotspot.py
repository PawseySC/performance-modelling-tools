#!/usr/bin/env python
#
# Copyright 2022 Pawsey Supercomputing Centre
#
# hpctoolkit-hotspot.py
#
#   Convert an HPC Toolkit database into a hotspot ("flat") profile.
#   Optionally outputs a CSV and/or png figure.
#
#
# Adapted from https://hatchet.readthedocs.io/en/latest/analysis_examples.html#generating-a-flat-profile
#
# Authors:
#
#   Joe Schoonover, Fluid Numerics LLC (joe@fluidnumerics.com)
#
# Getting help:
#
#   python3 hpctoolkit-hotspot.py --help
#
# Additional Documentation :
#
#
# /////////////////////////// #

import argparse
import hatchet as ht
import matplotlib.pyplot as plt


def cliParse():
  """
  Parse command line interface options with argparse
  and return args.
  """

  parser = argparse.ArgumentParser(
    description='Convert an HPC Toolkit database into a hotspot profile.')

  parser.add_argument('dbpath', type=str,
    help='Full path to HPC toolkit database.')

  parser.add_argument('--csv',  action='store_true',
                    help='Write flat profile to CSV (if set)')

  parser.add_argument('--png',  action='store_true',
                    help='Create bar plot of flat profile and save to png (if set)')

  parser.add_argument('--odir', type=str, default="./",
                    help='Path to output directory.')

  args = parser.parse_args()
  
  return args


def main():

  args = cliParse()

  dbname = args.dbpath.split("/")[-1]

  # If the output directory does not end in "/"
  # append "/"
  odir = args.odir
  if args.odir[-1] != "/":
    odir = args.odir+"/"

  # Read in HPCToolkit database.
  gf = ht.GraphFrame.from_hpctoolkit(args.dbpath)

  # Drop all index levels in the DataFrame except ``node``.
  gf.drop_index_levels()

  # Group DataFrame by ``name`` column, compute sum of all rows in each
  # group. This shows the aggregated time spent in each function.
  grouped = gf.dataframe.groupby('name').sum()

  # Sort DataFrame by ``time`` column in descending order.
  sorted_df = grouped.sort_values(by=['time'],
                                ascending=False)

  if args.csv :
    sorted_df.to_csv(odir+dbname+'.csv')

  # TO DO : bar plot, then csv

if __name__ == '__main__':
  main()
