import os
import argparse
import _pickle as cPickle
import gc
import time
import json
import copy

import hist as hist2
from coffea import processor

def read_hists(sample, path):
    files = os.listdir(path)

    # make sample dictionary
    # each key must be different for samples with different cross section
    # must also match the sample in xsecs.json..?
    hww = {
        "GluGluHToWWToLNuQQ": [file for file in files if "GluGluHToWWToLNuQQ" in file],
    }
    qcd = {
        #"QCD_HT300to500": [file for file in files if "QCD_HT300to500" in file],
        "QCD_HT500to700": [file for file in files if "QCD_HT500to700" in file],
        "QCD_HT700to1000": [file for file in files if "QCD_HT700to1000" in file],
        "QCD_HT1000to1500": [file for file in files if "QCD_HT1000to1500" in file],
        "QCD_HT1500to2000": [file for file in files if "QCD_HT1500to2000" in file],
        "QCD_HT2000toInf": [file for file in files if "QCD_HT2000toInf" in file]
    }
    tt_semileptonic = {
        "TTToSemiLeptonic": [file for file in files if "TTToSemiLeptonic" in file],
    }
    tt_hadronic = {
        "TTToHadronic": [file for file in files if "TTToHadronic" in file],
    }
    tt_dileptonic = {
        "TTTo2L2Nu": [file for file in files if "TTTo2L2Nu" in file],
    }
    wjets = {
        "WJetsToLNu_HT-200To400": [file for file in files if "WJetsToLNu_HT-200To400" in file],
        "WJetsToLNu_HT-400To600": [file for file in files if "WJetsToLNu_HT-400To600" in file],
        "WJetsToLNu_HT-600To800": [file for file in files if "WJetsToLNu_HT-600To800" in file],
        "WJetsToLNu_HT-800To1200": [file for file in files if "WJetsToLNu_HT-800To1200" in file],
        "WJetsToLNu_HT-1200To2500": [file for file in files if "WJetsToLNu_HT-1200To2500" in file],
    }
    st = {
        "ST_s-channel_4f_leptonDecays": [file for file in files if "ST_s-channel_4f_leptonDecays" in file],
        "ST_t-channel_eleDecays": [file for file in files if "ST_t-channel_eleDecays" in file],
        "ST_t-channel_muDecays": [file for file in files if "ST_t-channel_muDecays" in file],
        "ST_tW_antitop_5f_inclusiveDecays": [file for file in files if "ST_tW_antitop_5f_inclusiveDecays" in file],
        "ST_tW_top_5f_inclusiveDecays": [file for file in files if "ST_tW_top_5f_inclusiveDecays" in file],
    }
    zjets = {
        #"DYJetsToLL_Pt-50To100": [file for file in files if "DYJetsToLL_Pt-50To100" in file],
        "DYJetsToLL_Pt-100To250": [file for file in files if "DYJetsToLL_Pt-100To250" in file],
        "DYJetsToLL_Pt-250To400": [file for file in files if "DYJetsToLL_Pt-250To400" in file],
        "DYJetsToLL_Pt-400To650": [file for file in files if "DYJetsToLL_Pt-400To650" in file],
        "DYJetsToLL_Pt-650ToInf": [file for file in files if "DYJetsToLL_Pt-650ToInf" in file],
    }
    singleElectron = {
        "SingleElectron": [file for file in files if "SingleElectron" in file]
    }
    singleMuon = {
        "SingleMuon": [file for file in files if "SingleMuon" in file]
    }

    samples_dics = {
        "hww": hww,
        "tt_semileptonic": tt_semileptonic,
        "tt_hadronic": tt_hadronic,
        "tt_dileptonic": tt_dileptonic,
        "qcd": qcd,
        "wjets": wjets,
        "st": st,
        "zjets": zjets,
        "electron": singleElectron,
        "muon": singleMuon,
    }

    return samples_dics[sample]


def load_hists(sample_dic, histogram, region, path):
    """load and accumulate ouput dics by sample, histogram and region"""
    
    hists = {key:[] for key in sample_dic}
    
    for key, hist in sample_dic.items():
        print(f"{key}")
        for h in hist:
            with open(f"{path}/{h}", "rb") as f:
                # disable garbage collector
                gc.disable()
                
                # load and save histograms by region
                H = cPickle.load(f)
                k = [key for key in H][0]
                histos = dict()
                
                try:
                    histos[histogram] = H[k][histogram][{"region":region}]
                except:
                    print('did not find histogram')
                    gc.enable()
                    continue

                histos["sumw"] = H[k]["sumw"]                

                hists[key].append(histos)
         
                # enable garbage collector again
                gc.enable()

    for key in sample_dic:
        sample_dic[key] = processor.accumulate(hists[key])
    
    return sample_dic


def scale_hists(sample_dic, xsec_path, lumi):
    """scale histograms to cross section"""

    with open(xsec_path) as f:
        xsecs = json.load(f)

    out = []
    for sample in sample_dic:
        
        hists = sample_dic[sample]
        sumw = sample_dic[sample]["sumw"]
        
        try:
            xsec = eval(xsecs[sample])
        except:
            xsec = xsecs[sample]

        weight = (xsec * lumi) / sumw
        hists = copy.deepcopy(hists)

        for h in hists.values():
            if isinstance(h, hist2.Hist):
                h *= weight
                
        out.append(hists)
    
    print(sample_dic)
    print(sample)
    if len(out) == 1:
        return {sample.split("_")[0]: out[0]}
    else:
        return {sample.split("_")[0]: processor.accumulate(out)}


def main(args):
    # loading and accumulating histograms
    sample_dic = read_hists(args.sample, args.hpath)
    sample_dic = load_hists(sample_dic, args.histogram, args.region, args.hpath)

    # removing None histograms
    sample_dic = {key:val for key,val in sample_dic.items() if val is not None}

    # scaling to xsecs
    if args.sample not in ["electron", "muon"]:
        output = scale_hists(sample_dic, args.xsecs, args.lumi)
    else:
        output = sample_dic

    output_path = "hists"

    os.system(f'mkdir -p {output_path}/{args.region}/{args.histogram}')
    with open(f'{output_path}/{args.region}/{args.histogram}/{args.sample}_{args.histogram}.pkl', 'wb') as f:
        cPickle.dump(output, f, protocol=-1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--hpath",     dest="hpath",     default=None,      type=str,   help="path to histograms",         required=True)
    parser.add_argument("--sample",    dest="sample",    default=None,      type=str,   help="sample to process (eg hww)", required=True)
    parser.add_argument("--histogram", dest="histogram", default=None,      type=str,   help="histograms to process",      required=True)
    parser.add_argument("--region",    dest="region",    default=None,      type=str,   help="region to process",          required=True)
    parser.add_argument("--lumi",      dest="lumi",      default=None,      type=float, help="integrated luminosity",      required=True)
    parser.add_argument("--xsecs",     dest="xsecs",     default=None,      type=str,   help="path to cross sections",     required=True)
    args = parser.parse_args()

    main(args)
