#!/usr/bin/python

import json
import uproot
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea import processor
import pickle

import argparse
import warnings


def main(args):

    # read samples to submit
    with open(args.fileset, 'r') as f:
        files = json.load(f)[args.sample]
    fileset = {}
    fileset[args.sample] = ["root://cmsxrootd.fnal.gov/"+ f for f in files[args.starti:args.endi]]

    # define processor
    if args.processor == "hww":
        from boostedhiggs.hwwprocessor import HwwProcessor
        p = HwwProcessor(year=args.year, jet_arbitration='met', el_wp="wp80")
    else:
        warnings.warn('Warning: no processor declared')
        return

    print(fileset)

    if args.condor:
        uproot.open.defaults['xrootd_handler'] = uproot.source.xrootd.MultithreadedXRootDSource

        executor = processor.FuturesExecutor(compression=1, status=True, workers=3)

        run = processor.Runner(executor=executor,savemetrics=True,chunksize=10000,schema=NanoAODSchema)

        out,metrics = run(fileset,'Events',processor_instance=p)

        print(f"Metrics: {metrics}")

    filehandler = open(f'outfiles/{args.year}_{args.sample}_{args.starti}-{args.endi}.hist', 'wb')
    pickle.dump(out, filehandler)
    filehandler.close()

if __name__ == "__main__":
    # e.g. 
    # inside a condor job: python run.py --year 2017 --processor hww --condor --starti 0 --endi 1 --fileset metadata.json --sample GluGluHToWWToLNuQQ_M125_TuneCP5_PSweight_13TeV-powheg2-jhugen727-pythia8
    # inside a dask job:  python run.py --year 2017 --processor hww --dask --fileset metadata.json --sample GluGluHToWWToLNuQQ_M125_TuneCP5_PSweight_13TeV-powheg2-jhugen727-pythia8

    parser = argparse.ArgumentParser()
    parser.add_argument('--year',       dest='year',       default='2017',       help="year", type=str)
    parser.add_argument('--starti',     dest='starti',     default=0,            help="start index of files", type=int)
    parser.add_argument('--endi',       dest='endi',       default=-1,           help="end index of files", type=int)
    parser.add_argument("--processor",  dest="processor",  default="hww",        help="HWW processor", type=str)
    parser.add_argument("--condor",     dest="condor",     action="store_true",  default=False, help="Run with condor")
    parser.add_argument("--dask",       dest="dask",       action="store_true",  default=False, help="Run with dask")
    parser.add_argument("--fileset",    dest="fileset",    default=None,         help="Fileset", required=True)
    parser.add_argument('--sample',     dest='sample',     default=None,         help='sample name', required=True)
    args = parser.parse_args()

    main(args)
