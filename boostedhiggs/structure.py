import numpy as np
import awkward as ak
from uproot_methods import TVector2Array, TLorentzVectorArray


def buildevents(df, fatjet='FatJet', usemask=False):
    events = ak.Table()

    if 'genWeight' in df:
        events['genWeight'] = df['genWeight']
        events['Pileup_nPU'] = df['Pileup_nPU']

        events['genpart'] = ak.JaggedArray.fromcounts(
            df['nGenPart'],
            ak.Table.named(
                'particle',
                p4=TLorentzVectorArray.from_ptetaphim(
                    df['GenPart_pt'],
                    df['GenPart_eta'],
                    df['GenPart_phi'],
                    df['GenPart_mass'],
                ),
                pdgId=df['GenPart_pdgId'],
                genPartIdxMother=df['GenPart_genPartIdxMother'],
                statusFlags=df['GenPart_statusFlags'],
            ),
        )

    events['fatjets'] = ak.JaggedArray.fromcounts(
        df[f'n{fatjet}'],
        ak.Table.named(
            'fatjet',
            p4=TLorentzVectorArray.from_ptetaphim(
                df[f'{fatjet}_pt'],
                df[f'{fatjet}_eta'],
                df[f'{fatjet}_phi'],
                df[f'{fatjet}_mass'],
            ),
            # msoftdrop=ak.MaskedArray(df[f'{fatjet}_msoftdrop'] <= 0, df[f'{fatjet}_msoftdrop']) if usemask else np.maximum(1e-5, df[f'{fatjet}_msoftdrop']),
            msoftdrop=df[f'{fatjet}_msoftdrop'],
            area=df[f'{fatjet}_area'],
            n2=df[f'{fatjet}_n2b1'],
            jetId=df[f'{fatjet}_jetId'],
            lsf3=df[f'{fatjet}_lsf3'],
        ),
    )

    events['jets'] = ak.JaggedArray.fromcounts(
        df['nJet'],
        ak.Table.named(
            'jet',
            p4=TLorentzVectorArray.from_ptetaphim(
                df['Jet_pt'],
                df['Jet_eta'],
                df['Jet_phi'],
                df['Jet_mass'],
            ),
            deepcsvb=df['Jet_btagDeepB'],
            hadronFlavor=df['Jet_hadronFlavour'],
            jetId=df['Jet_jetId'],
        ),
    )

    events['met'] = TVector2Array.from_polar(
        df['MET_pt'],
        df['MET_phi']
    )

    events['electrons'] = ak.JaggedArray.fromcounts(
        df['nElectron'],
        ak.Table.named(
            'electron',
            p4=TLorentzVectorArray.from_ptetaphim(
                df['Electron_pt'],
                df['Electron_eta'],
                df['Electron_phi'],
                df['Electron_mass'],
            ),
            cutBased=df['Electron_cutBased'],
        ),
    )

    events['muons'] = ak.JaggedArray.fromcounts(
        df['nMuon'],
        ak.Table.named(
            'muon',
            p4=TLorentzVectorArray.from_ptetaphim(
                df['Muon_pt'],
                df['Muon_eta'],
                df['Muon_phi'],
                df['Muon_mass'],
            ),
            sip3d=df['Muon_sip3d'],
            dxy=df['Muon_dxy'],
            dz=df['Muon_dz'],
            mvaId=df['Muon_mvaId'],
            looseId=df['Muon_looseId'],
            pfRelIso04_all=df['Muon_pfRelIso04_all'],
            miniPFRelIso_all=df['Muon_miniPFRelIso_all'], 
        ),
    )

    events['taus'] = ak.JaggedArray.fromcounts(
        df['nTau'],
        ak.Table.named(
            'tau',
            p4=TLorentzVectorArray.from_ptetaphim(
                df['Tau_pt'],
                df['Tau_eta'],
                df['Tau_phi'],
                df['Tau_mass'],
            ),
            idDecayMode=df['Tau_idDecayMode'],
        ),
    )

    return events
