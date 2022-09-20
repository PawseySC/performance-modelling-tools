#!/usr/bin/env python
#
# Copyright 2022 Pawsey Supercomputing Centre
#
# rocprof-memcpy-report.py
#
#   Parse HIPMemcpy calls from a rocprof trace profile (results.json) 
#   and print a report similar to nvprof --print-gpu-trace, showing
#   start time, runtime, total amount of data moved, direction, and duration
#   in the order they occured
#
# Authors:
#
#   Joe Schoonover, Fluid Numerics LLC (joe@fluidnumerics.com)
#
# Getting help:
#
#   python3 rocprof-memcpy-report.py --help
#
# Additional Documentation :
#
#
# /////////////////////////// #

import json
import argparse
from os.path import exists
import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def cliParse():
  """
  Parse command line interface options with argparse
  and return args.
  """

  parser = argparse.ArgumentParser(
    description='Combine mixbench-report json output with rocprof-irm.csv to create IRM plot and summary report')

  parser.add_argument('trace', type=str,
    help='Full path to rocprof trace profile')

  args = parser.parse_args()
  
  return args

def convertFromNS( valueNS ):
   """
     Takes in a value reported in nanoseconds
     and returns a floating point value between [0,1000)
     with the appropriate units
   """

   units = ["n","u","m","","K","M","G"]
   value = valueNS
   ndiv = 0
   for k in range(0,7):
     if abs(value) >= 1000.0:
       value = value/1000.0
       ndiv += 1    
     else:
       break

   return value, units[ndiv];

def convertFromBytes( valueBytes ):
   """
     Takes in a value reported in bytes
     and returns a floating point value between [0,1000)
     with the appropriate units
   """

   units = ["","K","M","G"]
   value = valueBytes
   ndiv = 0
   for k in range(0,3):
     if abs(value) >= 1000.0:
       value = value/1000.0
       ndiv += 1    
     else:
       break

   return value, units[ndiv];

def main():

  args = cliParse()

  dataset = []

  if exists(args.trace) : 
    
    with open(args.trace,'r') as f:
      traceProfile = json.load(f)

    print('hipMemcpy (Trace)\n')
    print('Begin     Duration    Size    Bandwidth \n')
    print('=========================================== \n')
    getStartTime = True
    for event in traceProfile["traceEvents"]:
        if "name" in event.keys():

          # get the first start time to adjust start time relative to this value  
          if getStartTime:
            if event["ph"] == "X":
              # Get start time in Ns
              firstBeginNS = float(event["args"]["BeginNs"])
              getStartTime = False

          if event["name"] == "hipMemcpy":
            # Parse size from args
            eventArgs = event["args"]["args"].split(" ")
            sizeBytes = float(eventArgs[3].split('(')[-1].replace(')',''))
            Bytes, bytesUnits = convertFromBytes( sizeBytes )

            # Get duration in Ns
            durationNS = float(event["args"]["DurationNs"])
            duration, durationUnits = convertFromNS(durationNS)

            # Get start time in Ns
            beginNS = float(event["args"]["BeginNs"]) - firstBeginNS
            begin, beginUnits = convertFromNS(beginNS)

            # Calculate bandwidth (GB/s)
            bandwidth = sizeBytes/durationNS
            print(f'{begin:>6.2f}{beginUnits:>1}s  {duration:>6.2f}{durationUnits:>1}s  {Bytes:>6.2f}{bytesUnits:>1}B  {bandwidth:>6.4f} GB/s \n')


  else:
    print(f"File not found : {args.trace}")
    sys.exit(1)



if __name__ == '__main__':
  main()
