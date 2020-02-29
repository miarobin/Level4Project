import numpy as np

n_processes = 500000
n_jet = 4
joe_mom = np.load('mom_4jet_500000.npy', allow_pickle=True, encoding='bytes')

me = np.zeros(n_processes)
mom = np.zeros((n_processes, n_jet, 4))
for i in tqdm(range(n_processes)):
    me[i], mom[i] = matrix2py.get_value(joe_mom[i],alphas,nhel)
np.save('LO_mom_{}jet_{}'.format(n_jet, n_processes), mom)
np.save('LO_me_{}jet_{}'.format(n_jet, n_processes), me)