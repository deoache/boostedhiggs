import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")

import os
import argparse
import hist as hist2
import _pickle as cPickle

"""
For normal stack plots:
    python plot_stack.py --hname met_kin --haxis mt_lepmet --year 2017 --channel hadel --idir /uscms/home/docampoh/nobackup/boostedhiggs/hists/ --odir plots_Sep2
For cutflow stack plots:
    python plot_stack.py --hname cutflow --haxis cut --year 2017 --channel hadel --idir /uscms/home/docampoh/nobackup/boostedhiggs/hists/ --odir plots_Sep2
"""

def main(args):

    input_dir = args.idir + '/' + args.channel + '/' + args.hist_name

    map_proc = {
        "hww": "GluGluHToWWToLNuQQ",
        "qcd": "QCD",
        "tt_semileptonic": "TTToSemiLeptonic",
        "tt_hadronic": "TTToHadronic",
        "tt_dileptonic": "TTTo2L2Nu",
        "wjets": "WJetsToLNu",
        "zjets": "DYJetsToLL",
        "st": "ST"
    }
    
    paths_by_proc = {}    
    if args.channel=="hadmu":
        paths_by_proc['data'] = f"{input_dir}/muon_{args.hist_name}.pkl"
        data_key = "SingleMuon"
    elif args.channel=="hadel":
        paths_by_proc['data'] = f"{input_dir}/electron_{args.hist_name}.pkl"
        data_key = "SingleElectron"
    else:
        print('Not a valid data channel')
        exit

    map_proc['data'] = data_key
    paths_by_proc['hww'] = f"{input_dir}/hww_{args.hist_name}.pkl"

    mc_procs = ["tt_semileptonic", "tt_hadronic", "tt_dileptonic", "qcd", "wjets", "zjets", "st"]
    for mc in mc_procs:
        paths_by_proc[mc] = f"{input_dir}/{mc}_{args.hist_name}.pkl"

    hists_by_proc = {}
    for key,path in paths_by_proc.items():
        with open(path, "rb") as f:
            pklf = cPickle.load(f)
        hists_by_proc[key] = pklf[map_proc[key]][args.hist_name].project(args.hist_axis)

    print(f"generating {args.channel}_{args.hist_axis} plot")

    if args.hist_name == "cutflow":
        from utils import plot_cutflow
        plot_cutflow(
            data=hists_by_proc["data"],
            sig=hists_by_proc["hww"],
            bkg=[hists_by_proc["tt_hadronic"],
                 hists_by_proc["tt_dileptonic"],
                 hists_by_proc["st"],
                 hists_by_proc["zjets"],
                 hists_by_proc["tt_semileptonic"],
                 hists_by_proc["wjets"],
                 hists_by_proc["qcd"],],
            bkg_labels=[r"$t\bar{t}$ hadronic",
                        r"$t\bar{t}$ dileptonic",
                        r"Single-t",
                        r"$Z(ll)$",
                        r"$t\bar{t}$ semileptonic",
                        r"$W(l\nu)$",
                        "QCD"], 
            region=args.channel,
            odir=args.odir,
            year=args.year,
        )
    else:
        from utils import plot_stack
        plot_stack(
            data=hists_by_proc["data"],
            sig=hists_by_proc["hww"],
            bkg=[hists_by_proc["tt_hadronic"],
                 hists_by_proc["tt_dileptonic"],
                 hists_by_proc["st"],
                 hists_by_proc["zjets"],
                 hists_by_proc["tt_semileptonic"],
                 hists_by_proc["qcd"],
                 hists_by_proc["wjets"]],
            bkg_labels=[r"$t\bar{t}$ hadronic",
                        r"$t\bar{t}$ dileptonic",
                        r"Single-t",
                        r"$Z(ll)$",
                        r"$t\bar{t}$ semileptonic",
                        "QCD",
                        r"$W(l\nu)$"],
            axis_name=args.hist_axis,
            region=args.channel,
            odir=args.odir,
            year=args.year,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hname',      dest='hist_name',  default=None,  help="hist name",                         type=str,   required=True)
    parser.add_argument('--haxis',      dest='hist_axis',  default=None,  help="hist axis to project",              type=str,   required=True)
    parser.add_argument('--year',       dest='year',       default=None,  help="year",                              type=str,   required=True)
    parser.add_argument('--channel',    dest='channel',    default=None,  help="channel (hadmu or hadel)",          type=str,   required=True)
    parser.add_argument('--idir',       dest='idir',       default=None,  help="directory with pickled histograms", type=str,   required=True)
    parser.add_argument('--odir',       dest='odir',       default=None,  help="output directory with plots",       type=str,   required=True)

    args = parser.parse_args()

    os.system(f'mkdir -p {args.odir}')

    main(args)
