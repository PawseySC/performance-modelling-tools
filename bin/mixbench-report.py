#!/usr/bin/env python
#
# Copyright 2022 Pawsey Supercomputing Centre
#
# mixbench-report.py
#
#   Convert ekondis/mixbench output into CSV or json output.
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
#   python3 mixbench-report.py --help
#
# Additional Documentation :
#
#
# /////////////////////////// #

import json
import argparse
from os.path import exists

def cliParse():
  """
  Parse command line interface options with argparse
  and return args.
  """

  parser = argparse.ArgumentParser(
    description='Convert mixbench stdout text file into CSV or json format')

  parser.add_argument('mblog', type=str,
    help='Full path to the mixbench stdout text file.')

  parser.add_argument('--csv',  action='store_true',
                    help='Write flat profile to CSV (if set)')

  parser.add_argument('--json',  action='store_true',
                    help='Create bar plot of flat profile and save to png (if set)')

  args = parser.parse_args()
  
  return args

def parseClockRateGHz(line):
  clockRateTxt = line.split(':')[-1].strip().split(' ')
  clockRate = clockRateTxt[0]
  clockUnits = clockRateTxt[1]
  if clockUnits == 'MHz' :
    return {'clockRateGHz':float(clockRate)/1000.0}
  elif clockUnits == 'GHz' :
    return {'clockRateGHz':float(clockRate)}
  else:
    print(f"Warning - units {clockUnits} not configured for reporting")
    return {'clockRateGHz':float(clockRate)}


def parseWarpSize(line):
  txt = line.split(':')[-1].strip()
  return {'warpSize':int(txt)}

def parseSPs(line):
  txt = line.split(':')[-1].strip().split(' ')
  return {'totalSPs':int(txt[0])}

def parseCSVData(csvLines):

  data = {'singlePrecision':{'arithmeticIntensity':[],
                             'executionTime':[],
                             'GFLOPS':[],
                             'bandwidth':[]},
          'doublePrecision':{'arithmeticIntensity':[],
                             'executionTime':[],
                             'GFLOPS':[],
                             'bandwidth':[]},
          'integer':{'arithmeticIntensity':[],
                     'executionTime':[],
                     'GIOPS':[],
                     'bandwidth':[]}}

  for line in csvLines:
      sline = line.split(',')
      if len(sline) > 1:
        data['singlePrecision']['arithmeticIntensity'].append(float(sline[1]))
        data['singlePrecision']['executionTime'].append(float(sline[2]))
        data['singlePrecision']['GFLOPS'].append(float(sline[3]))
        data['singlePrecision']['bandwidth'].append(float(sline[4]))
  
        data['doublePrecision']['arithmeticIntensity'].append(float(sline[5]))
        data['doublePrecision']['executionTime'].append(float(sline[6]))
        data['doublePrecision']['GFLOPS'].append(float(sline[7]))
        data['doublePrecision']['bandwidth'].append(float(sline[8]))
  
        data['integer']['arithmeticIntensity'].append(float(sline[9]))
        data['integer']['executionTime'].append(float(sline[10]))
        data['integer']['GIOPS'].append(float(sline[11]))
        data['integer']['bandwidth'].append(float(sline[12]))

  return data


def main():

  args = cliParse()

  dataset = {}
  if exists(args.mblog) : 
    
    with open(args.mblog,'r') as f:
      log = f.read().split('\n')

    k = 0
    for line in log:

        if 'GPU clock rate' in line:
          dataset.update(parseClockRateGHz(line))

        elif 'WarpSize' in line:
          dataset.update(parseWarpSize(line))

        elif 'Total SPs' in line:
          dataset.update(parseSPs(line))

        elif 'CSV' in line:
          dataset.update(parseCSVData(log[k+3:-1]))
          break
        k+=1

  else:
    print(f"File not found : {args.mblog}")

  # Post process calculations
  dataset.update({'computeUnits':dataset['totalSPs']/dataset['warpSize']})
  dataset.update({'gipsPeak':dataset['computeUnits']*dataset['clockRateGHz']})

  if args.json :
      ofile = args.mblog+'.json'
      with open(ofile,'w') as f:
          json.dump(dataset,f)

#  print(dataset)

if __name__ == '__main__':
  main()
