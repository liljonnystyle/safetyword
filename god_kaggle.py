
from GodsPipe import pipeline

df = pd.read_csv('data/training.csv')
df = df.set_index('EventId')
for i in xrange(30):
    df = df[df.iloc[:,i] >= -998]
cols = ['DER_mass_transverse_met_lep','DER_pt_h','DER_deltaeta_jet_jet','DER_mass_jet_jet','DER_prodeta_jet_jet',
        'DER_pt_tot','DER_pt_ratio_lep_tau','DER_met_phi_centrality','DER_lep_eta_centrality','PRI_tau_pt','PRI_jet_num']

goddf = df[cols]
