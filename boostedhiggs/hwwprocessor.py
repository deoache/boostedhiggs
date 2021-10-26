import numpy as np
import awkward as ak
import hist as hist2
import json
from copy import deepcopy

from coffea import processor
from coffea.nanoevents.methods import candidate, vector
from coffea.analysis_tools import Weights, PackedSelection

from boostedhiggs.corrections import (
    corrected_msoftdrop,
    add_pdf_weight,
    add_pileup_weight,
    add_leptonSFs,
    lumiMasks,
    is_overlap,
)
from boostedhiggs.utils import (
    getParticles,
    match_HWWlepqq,
)

import logging
logger = logging.getLogger(__name__)

class HwwProcessor(processor.ProcessorABC):
    def __init__(self, year="2017", jet_arbitration='met', el_wp="wp80"):
        self._year = year
        self._jet_arbitration = jet_arbitration
        self._el_wp = el_wp
        
        self._triggers = {
            2016: {
                'e': [
                    "Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
                    "Ele115_CaloIdVT_GsfTrkIdT",
                    "Ele15_IsoVVVL_PFHT600",
                ],
                'mu': [
                    "Mu50",
                    "Mu55",
                    "Mu15_IsoVVVL_PFHT600",
                ],
            },
            2017: {
                'e': [
                    #"Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
                    "Ele115_CaloIdVT_GsfTrkIdT",
                    #"Ele15_IsoVVVL_PFHT600",
                    "Ele35_WPTight_Gsf",
                ],
                'mu': [
                    "Mu50",
                    #"Mu15_IsoVVVL_PFHT600",
                    "IsoMu27",
                ],
            },
            2018: {
                'e': [
                    "Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
                    "Ele115_CaloIdVT_GsfTrkIdT",
                    "Ele15_IsoVVVL_PFHT600",
                ],
                'mu': [
                    "Mu50",
                    "Mu15_IsoVVVL_PFHT600",
                ],
            }
        }
        self._triggers = self._triggers[int(self._year)]
        print( self._triggers )

        self._json_paths = {
            '2016': "data/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt",
            '2017': "data/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt",
            '2018': "data/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt",
        }

        self._metfilters = [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "HBHENoiseFilter",
            "HBHENoiseIsoFilter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
        ]
        
        # WPs for btagDeepFlavB (UL)
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
        self._btagWPs = {
            '2016': {
                'loose': 0.0614,
                'medium': 0.3093,
                'tight': 0.7221,
            },
            '2017': {
                'loose': 0.0532,
                'medium': 0.3040,
                'tight': 0.7476,
            },
            '2018': {
                'loose': 0.0490,
                'medium': 0.2783,
                'tight': 0.7100,
            },
        }

        self.make_output = lambda: {
            'sumw': 0.,
            'cutflow': hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.IntCategory([0, 1], name="cut", label='Cut index', growth=True),
                hist2.axis.IntCategory([0, 2, 4, 6, 8], name='genflavor', label='gen flavor'),
                hist2.storage.Weight(),
            ),
            'signal_kin': hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.IntCategory([0, 2, 4, 6, 8], name='genflavor', label='gen flavor'),
                hist2.axis.IntCategory([0, 1, 4, 6, 9], name='genHflavor', label='higgs matching'),
                hist2.axis.IntCategory([0, 1, 2, 3, 4], name='nprongs', label='Jet nprongs'),
                hist2.storage.Weight(),
            ),
            "jet_kin": hist2.Hist(
                hist2.axis.StrCategory([], name='region', growth=True),
                hist2.axis.Regular(30, 200, 1000, name='jetpt', label=r'Jet $p_T$ [GeV]'),
                hist2.axis.Regular(30, 15, 200, name="jetmsd", label="Jet $m_{sd}$ [GeV]"),
                hist2.axis.Regular(25, -20, 0, name="jetrho", label=r"Jet $\rho$"),
                hist2.axis.Regular(20, 0, 1, name="btag", label="Jet btag (opphem)"),
                hist2.storage.Weight(),
            ),
            "lep_kin": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.Regular(25, 0, 1, name="lepminiIso", label="lep miniIso"),
                hist2.axis.Regular(25, 0, 1, name="leprelIso", label="lep Rel Iso"),
                hist2.axis.Regular(40, 10, 800, name='lep_pt', label=r'lep $p_T$ [GeV]'),
                hist2.axis.Regular(30, 0, 5, name="deltaR_lepjet", label="$\Delta R(l, Jet)$"),
                hist2.storage.Weight(),
            ),
            "met_kin": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.Regular(30, 0, 500, name="met", label=r"$p_T^{miss}$ [GeV]"),
                hist2.axis.Regular(20, 0, 300, name="mt_lepmet", label=r"$m_{T}$ [GeV]"),
                hist2.storage.Weight(),
            ),
            "jet_lep": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.Regular(30, 20, 200, name="jetlep_msd", label="(Jet - Lep) mass (SD) [GeV]"),
                hist2.axis.Regular(30, 20, 200, name="jetlep_mass", label="(Jet - Lep) mass [GeV]"),
            ),
            "higgs_kin": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.Regular(50, 10, 1000, name='matchedHpt', label=r'matched H $p_T$ [GeV]'),
                hist2.axis.Variable(
                    [10,35,60,85,110,135,160,185,210,235,260,285,310,335,360,385,410,450,490,530,570,615,665,715,765,815,865,915,965],
                    name='genHpt', 
                    label=r'genH $p_T$ [GeV]',
                ),
                hist2.storage.Weight(),
            ),
            "jet_lep_kin": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.Regular(30, 15, 200, name="jetlepmass", label="(Jet - lep) $m$ [GeV]"),
                hist2.axis.Regular(30, 15, 200, name="jetlepmsd", label="(Jet - lep) $m_{sd}$ [GeV]"),
                hist2.storage.Weight(),
            ),
            "mt_lepmet_cutflow": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.IntCategory([], name='cut', label='Cut index', growth=True),
                hist2.axis.Regular(20, 0, 300, name="mt_lepmet", label=r"$m_{T}$ [GeV]"),
                hist2.storage.Weight(),
            ),
            "met_cutflow": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.IntCategory([], name='cut', label='Cut index', growth=True),
                hist2.axis.Regular(30, 0, 500, name="met", label=r"$p_T^{miss}$ [GeV]"),
                hist2.storage.Weight(),
            ),
            "btag_cutflow": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.IntCategory([], name='cut', label='Cut index', growth=True),
                hist2.axis.Regular(20, 0, 1, name="btag", label="Jet btag (opphem)"),
                hist2.storage.Weight(),
            ),
            "lepminiIso_cutflow": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.IntCategory([], name='cut', label='Cut index', growth=True),
                hist2.axis.Regular(25, 0, 1, name="lepminiIso", label="lep miniIso"),
                hist2.storage.Weight(),
            ),
            "leprelIso_cutflow": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.IntCategory([], name='cut', label='Cut index', growth=True),
                hist2.axis.Regular(25, 0, 1, name="leprelIso", label="lep Rel Iso"),
                hist2.storage.Weight(),
            ),

        }
        
    def process(self, events):
        dataset = events.metadata['dataset']
        selection = PackedSelection()
        isRealData = not hasattr(events, "genWeight")
        nevents = len(events)
        weights = Weights(nevents, storeIndividual=True)
        
        output = self.make_output()
        if not isRealData:
            output['sumw'] = ak.sum(events.genWeight)
            
        # overlap removal
        #if isRealData:
        #    overlap_removal = is_overlap(events,dataset,self._triggers['e']+self._triggers['mu'],self._year)
        #else:
        #    overlap_removal = np.ones(len(events), dtype='bool')

        # trigger
        triggers = {}
        for channel in ["e","mu"]:
            # apply trigger to both data and MC
            #if isRealData:
            trigger = np.zeros(len(events), dtype='bool')
            for t in self._triggers[channel]:
                if t in events.HLT.fields:
                    trigger = trigger | events.HLT[t]
            triggers['trigger'+channel] = trigger
            del trigger
            #else:
            #    selection.add('trigger'+channel, np.ones(nevents, dtype='bool'))
            
        # lumi mask
        lumimask = np.ones(len(events), dtype='bool')
        if isRealData:
            lumimask = lumiMasks[self._year](events.run, events.luminosityBlock)

        # MET filters
        metfilters = np.ones(nevents, dtype='bool')
        for mf in self._metfilters:
            if mf in events.Flag.fields:
                metfilters = metfilters & events.Flag[mf]
        # only for data: 
        if isRealData:
             metfilters = metfilters & events.Flag["eeBadScFilter"]
        # only for 2017 and 2018:
        if self._year=="2017" or self._year=="2018":
            if "ecalBadCalibFilter" in events.Flag.fields:
                metfilters = metfilters & events.Flag["ecalBadCalibFilter"]

        # muons
        goodmuon = (
            (events.Muon.pt > 25)
            & (abs(events.Muon.eta) < 2.4)
            & events.Muon.mediumId
        )
        nmuons = ak.sum(goodmuon, axis=1)
        lowptmuon = (
            (events.Muon.pt > 10)
            & (abs(events.Muon.eta) < 2.4)
            & events.Muon.looseId
        )
        nlowptmuons = ak.sum(lowptmuon, axis=1)
        
        # electrons
        goodelectron = (
                (events.Electron.pt > 25)
                & (abs(events.Electron.eta) < 2.5)
        )
        if self._el_wp == "wp80":
            goodelectron = ( goodelectron 
                             & (events.Electron.mvaFall17V2noIso_WP80)
                         )
        elif self._el_wp == "wp90":
            goodelectron = ( goodelectron
                             & (events.Electron.mvaFall17V2noIso_WP90)
                         )
        elif self._el_wp == "wpl":
            goodelectron = ( goodelectron
                             & (events.Electron.mvaFall17V2noIso_WPL)
                         )
        else:
            raise RuntimeError("Unknown working point")
        nelectrons = ak.sum(goodelectron, axis=1)
        lowptelectron = (
            (events.Electron.pt > 10)
            & (abs(events.Electron.eta) < 2.5)
            & (events.Electron.cutBased >= events.Electron.LOOSE)
        )
        nlowptelectrons = ak.sum(lowptelectron, axis=1)

        # taus
        goodtau = (
            (events.Tau.pt > 20)
            & (abs(events.Tau.eta) < 2.3)
            & (events.Tau.idAntiEle >= 8)
            & (events.Tau.idAntiMu >= 1)
            #& ak.all(events.Tau.metric_table(events.Muon[goodmuon]) > 0.4, axis=2)
            #& ak.all(events.Tau.metric_table(events.Electron[goodelectron]) > 0.4, axis=2)
        )
        ntaus = ak.sum(goodtau, axis=1)
            
        # concatenate leptons and select leading one
        goodleptons = ak.concatenate([events.Muon[goodmuon], events.Electron[goodelectron]], axis=1)
        candidatelep = ak.firsts(goodleptons[ak.argsort(goodleptons.pt)])
        candidatelep_p4 = ak.zip(
            {
                "pt": candidatelep.pt,
                "eta": candidatelep.eta,
                "phi": candidatelep.phi,
                "mass": candidatelep.mass,
                "charge": candidatelep.charge,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )

        # missing transverse energy
        met = events.MET

        # transverse mass of lepton and MET
        mt_lep_met = np.sqrt(
            2.*candidatelep_p4.pt*met.pt*(ak.ones_like(met.pt) - np.cos(candidatelep_p4.delta_phi(met)))
        )
        
        # fatjets
        fatjets = events.FatJet
        fatjets["msdcorr"] = corrected_msoftdrop(fatjets)
        fatjets["qcdrho"] = 2 * np.log(fatjets.msdcorr / fatjets.pt)
        
        candidatefj = fatjets[
            (fatjets.pt > 200)
            & (abs(fatjets.eta) < 2.5)
            & fatjets.isTight
        ]
        dphi_met_fj = abs(candidatefj.delta_phi(met))
        dr_lep_fj = candidatefj.delta_r(candidatelep_p4)

        if self._jet_arbitration == 'pt':
            candidatefj = ak.firsts(candidatefj)
        elif self._jet_arbitration == 'met':
            candidatefj = ak.firsts(candidatefj[ak.argmin(dphi_met_fj,axis=1,keepdims=True)])
        elif self._jet_arbitration == 'lep':
            candidatefj = ak.firsts(candidatefj[ak.argmin(dr_lep_fj,axis=1,keepdims=True)])
        else:
            raise RuntimeError("Unknown candidate jet arbitration")

        # lepton isolation
        # check pfRelIso04 vs pfRelIso03
        lep_miniIso = candidatelep.miniPFRelIso_all
        lep_relIso = candidatelep.pfRelIso03_all
        mu_iso = ( ((candidatelep.pt < 55.) & (lep_relIso < 0.25)) | 
                     ((candidatelep.pt >= 55.) & (lep_miniIso < 0.1)) )
        el_iso = ( ((candidatelep.pt < 120.) & (lep_relIso < 0.25)) |
                   ((candidatelep.pt >= 120.) & (lep_miniIso < 0.1)) )

        # leptons within fatjet
        lep_in_fj = ak.fill_none(candidatefj.delta_r(candidatelep_p4) < 0.8,False)
        
        # lepton and fatjet mass
        candidatefj_p4 = ak.zip(
            {
                "pt": candidatefj.pt,
                "eta": candidatefj.eta,
                "phi": candidatefj.phi,
                "mass": candidatefj.msdcorr,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=candidate.behavior,
        )
        lep_fj_m = (candidatefj - candidatelep_p4).mass
        lep_fj_msd = (candidatefj_p4 - candidatelep_p4).mass

        # jets
        jets = events.Jet
        jets = jets[
            (jets.pt > 30) 
            & (abs(jets.eta) < 2.5) 
            & jets.isTight
        ]
        dphi_jet_fj = abs(jets.delta_phi(candidatefj))
        dr_jet_fj = abs(jets.delta_r(candidatefj))
        
        # b-jets
        bjets_ophem = ak.max(jets[dphi_jet_fj > np.pi / 2].btagDeepFlavB, axis=1)
        # bjets_ophem = ak.max(jets[dphi_jet_fj > np.pi / 2].btagDeepB,axis=1) # 0.4941 medium 2017 WP

        # match HWW semi-lep dataset
        if "HWW" in dataset:
            hWWlepqq_flavor,hWWlepqq_matched,hWWlepqq_nprongs,matchedH,genH,iswlepton,iswstarlepton = match_HWWlepqq(events.GenPart,candidatefj)
            matchedH_pt = ak.firsts(matchedH.pt)
            genH_pt = ak.firsts(genH.pt)
        else:
            hWWlepqq_flavor = ak.zeros_like(candidatefj.pt) 
            hWWlepqq_matched = ak.zeros_like(candidatefj.pt)
            hWWlepqq_nprongs = ak.zeros_like(candidatefj.pt)
            matchedH = ak.zeros_like(candidatefj.pt)
            matchedH_pt = ak.zeros_like(candidatefj.pt)
            genH = ak.zeros_like(candidatefj.pt)
            genH_pt = ak.zeros_like(candidatefj.pt)
            iswlepton = ak.ones_like(candidatefj.pt, dtype=bool)
            iswstarlepton = ak.ones_like(candidatefj.pt, dtype=bool)
        
        # add weights for MC
        if not isRealData:
            weights.add("genweight", events.genWeight)
            if "LHEPdfWeight" in events.fields:
                add_pdf_weight(weights, events.LHEPdfWeight)
            else:
                add_pdf_weight(weights, None)
            add_pileup_weight(weights, events.Pileup.nPU, self._year)
            logger.debug("Weight statistics: %r" % weights.weightStatistics)
            
        # regions
        for channel in ["e","mu"]:
            selection.add('trigger'+channel, triggers['trigger'+channel] )
        selection.add('lumimask',lumimask)
        selection.add('metfilters',metfilters)
        selection.add("iswlepton", iswlepton)
        selection.add("iswstarlepton", iswstarlepton)
        selection.add("fjacc", (candidatefj.pt > 200) & (abs(candidatefj.eta) < 2.5) & (candidatefj.qcdrho > -6.) & (candidatefj.qcdrho < -1.4) )
        selection.add("fjmsd", candidatefj.msdcorr > 15.)
        selection.add("lepinfj", lep_in_fj)
        selection.add('onemuon', (nmuons == 1) & (nlowptmuons <= 1) & (nelectrons == 0) & (nlowptelectrons == 0) & (ntaus == 0))
        selection.add('oneelectron', (nelectrons == 1) & (nlowptelectrons <= 1) & (nmuons == 0) & (nlowptmuons == 0) & (ntaus == 0))
        selection.add('muonkin', (candidatelep.pt > 30.) & abs(candidatelep.eta < 2.4))
        selection.add('electronkin', (candidatelep.pt > 40.) & abs(candidatelep.eta < 2.4))
        selection.add("muoniso",mu_iso)
        selection.add("electroniso",el_iso)
        selection.add("btag_ophem_med", bjets_ophem < self._btagWPs[self._year]['medium'])
        selection.add("met20", met.pt > 20.)
        selection.add("mtlepmet", mt_lep_met < 80.)

        regions = {
            #"hadel_noiso": ["triggere", "metfilters", "lumimask", "oneelectron", "fjacc", "fjmsd", "btag_ophem_med", "met20", "mtlepmet"],
            #"hadmu_noiso": ["triggermu", "metfilters", "lumimask", "onemuon", "fjacc", "fjmsd", "btag_ophem_med", "met20", "mtlepmet"],
            "hadel": ["triggere", "metfilters", "lumimask", "oneelectron", "fjacc", "fjmsd", "btag_ophem_med", "met20", "lepinfj", "mtlepmet", "electroniso"],
            "hadmu": ["triggermu", "metfilters", "lumimask", "onemuon", "fjacc", "fjmsd", "btag_ophem_med", "met20", "lepinfj", "mtlepmet", "muoniso"],
            #"noselection": []
        }

        if "HWW" in dataset:
            keys = regions.keys()
            for key in keys:
                if key=="noselection": continue
                regions["%s_iswlepton"%key] = regions[key] + ["iswlepton"]
                regions["%s_iswstarlepton"%key] = regions[key] + ["iswstarlepton"]

        # make dictionary of weights for different regions
        weights_dict = {}
        for region in regions.keys():
            weights_dict[region] = deepcopy(weights)
            # add region dependent weights for MC
            if not isRealData:
                if "hadel" in region:
                    add_leptonSFs(weights_dict[region],  candidatelep, self._year, "elec")
                if "hadmu" in region:
                    add_leptonSFs(weights_dict[region], candidatelep, self._year, "muon")
                    
        # function to normalize arrays after a cut or selection
        def normalize(val, cut=None):
            if cut is None:
                ar = ak.to_numpy(ak.fill_none(val, np.nan))
                return ar
            else:
                ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
                return ar
        
        def fill(region):
            selections = regions[region]
            cut = selection.all(*selections)
            weights_region = weights_dict[region]
            
            output['signal_kin'].fill(
                region=region,
                genflavor=normalize(hWWlepqq_flavor, cut),
                genHflavor=normalize(hWWlepqq_matched, cut),
                nprongs=normalize(hWWlepqq_nprongs, cut),
                weight = weights_region.weight()[cut],
            )
            output["jet_kin"].fill(
                region=region,
                jetpt=normalize(candidatefj.pt, cut),
                jetmsd=normalize(candidatefj.msdcorr, cut),
                jetrho=normalize(candidatefj.qcdrho, cut),
                btag=normalize(bjets_ophem, cut),
                weight=weights_region.weight()[cut],
            )
            output['lep_kin'].fill(
                region=region,
                lepminiIso=normalize(lep_miniIso, cut),
                leprelIso=normalize(lep_relIso, cut),
                lep_pt=normalize(candidatelep.pt, cut),
                deltaR_lepjet=normalize(candidatefj.delta_r(candidatelep_p4), cut),
                weight=weights_region.weight()[cut],
            )
            output["met_kin"].fill(
                region=region,
                met=normalize(met.pt, cut),
                mt_lepmet=normalize(mt_lep_met, cut),
                weight=weights_region.weight()[cut],
            )
            output["jet_lep"].fill(
                region=region,
                jetlep_msd=normalize(lep_fj_msd, cut),
                jetlep_mass=normalize(lep_fj_m,cut),
                weight=weights_region.weight()[cut],
            )
            if "HWW" in dataset:
                output['higgs_kin'].fill(
                    region=region,
                    matchedHpt=normalize(matchedH_pt, cut),
                    genHpt=normalize(genH_pt, cut),
                    weight=weights_region.weight()[cut],
                )
                
            # cutflow
            allcuts = set([])
            cut = selection.all(*allcuts)
            output["cutflow"].fill(
                region=region,
                cut=0,
                genflavor=normalize(hWWlepqq_flavor, cut),
                weight=weights_region.weight()[cut],
            )
            output["mt_lepmet_cutflow"].fill(
                region=region,
                cut=0,
                mt_lepmet=normalize(mt_lep_met, cut),
                weight=weights_region.weight()[cut],
            )
            output["met_cutflow"].fill(
                region=region,
                cut=0,
                met=normalize(met.pt, cut),
                weight=weights_region.weight()[cut],
            )
            output["btag_cutflow"].fill(
                region=region,
                cut=0,
                btag=normalize(bjets_ophem, cut),
                weight=weights_region.weight()[cut],
            )
            output["lepminiIso_cutflow"].fill(
                region=region,
                cut=0,
                lepminiIso=normalize(lep_miniIso, cut),
                weight=weights_region.weight()[cut],
            )
            output["leprelIso_cutflow"].fill(
                region=region,
                cut=0,
                leprelIso=normalize(lep_relIso, cut),
                weight=weights_region.weight()[cut],
            )
            for i, cut in enumerate(regions[region]):
                allcuts.add(cut)
                cut = selection.all(*allcuts)
                output["cutflow"].fill(
                    region=region,
                    cut=i + 1,
                    genflavor=normalize(hWWlepqq_flavor, cut),
                    weight=weights_region.weight()[cut],
                )
                output["mt_lepmet_cutflow"].fill(
                    region=region,
                    cut=i + 1,
                    mt_lepmet=normalize(mt_lep_met, cut),
                    weight=weights_region.weight()[cut],
                )
                output["met_cutflow"].fill(
                    region=region,
                    cut=i + 1,
                    met=normalize(met.pt, cut),
                    weight=weights_region.weight()[cut],
                )
                output["btag_cutflow"].fill(
                    region=region,
                    cut=i + 1,
                    btag=normalize(bjets_ophem, cut),
                    weight=weights_region.weight()[cut],
                )
                output["lepminiIso_cutflow"].fill(
                    region=region,
                    cut=i + 1,
                    lepminiIso=normalize(lep_miniIso, cut),
                    weight=weights_region.weight()[cut],
                )
                output["leprelIso_cutflow"].fill(
                    region=region,
                    cut=i + 1,
                    leprelIso=normalize(lep_relIso, cut),
                    weight=weights_region.weight()[cut],
                )
                
        for region in regions:
            if isRealData:
                if "SingleMuon" in dataset and "hadel" in region: continue
                if "SingleElectron" in dataset and "hadmu" in region: continue
            fill(region)

        return {dataset: output}
            
    def postprocess(self, accumulator):
        return accumulator
