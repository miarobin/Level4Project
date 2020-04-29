import numpy as np
import DataPreprocessing
import time

def split(cutoff, mandel_str, mom):
    largeS = []
    smallS = []
    mandel_vars = DataPreprocessing.mandel_creation(mandel_str, mom)
    for i, data_point in enumerate(np.transpose(mandel_vars)):
        if(min(data_point) > cutoff):
            largeS.append(i)
        else:
            smallS.append(i)

    return(largeS, smallS)


def datasetSplit(mom_filename, me_filename, cutoff, mandel_str):
    me_raw = np.load(me_filename, allow_pickle=True) #Matrix elements
    mom_raw = np.load(mom_filename, allow_pickle=True) #4-momenta of inputs

    largeS, smallS = split(cutoff, mandel_str, mom_raw)

    mom_small = mom_raw[smallS]
    me_small = me_raw[smallS]

    mom_large = mom_raw[largeS]
    me_large = me_raw[largeS]

    return (mom_large, me_large, mom_small, me_small)

def saveDatasetSplit(mom_filename, me_filename, cutoff, mandel_str):
    mom_large, me_large, mom_small, me_small = datasetSplit(mom_filename, me_filename, cutoff, mandel_str)
    np.save('MG_mom_largeS_{}'.format(len(mom_large)), mom_large)
    np.save('MG_me_largeS_{}'.format(len(mom_large)), me_large)

    np.save('MG_mom_smallS_{}'.format(len(mom_small)), mom_large)
    np.save('MG_me_smallS_{}'.format(len(mom_small)), me_large)

#mandel_str = ['1,3','1,4','2,3','2,4','1,2,3','1,2,4','2,3,4','1,3,4']
#mom_large, me_large, mom_small, me_small = datasetSplit('MG_mom_1000000.npy', 'MG_me_1000000.npy', 50000, mandel_str)

def smallestS(me_filename, mom_filename, mandel_index):
    all_mandel = ['1,2','1,3','2,3']
    #all_mandel = ['1,2','1,3','1,4','2,3','2,4','3,4']

    me, mom = DataPreprocessing.npy(me_filename, mom_filename, [])
    
    tic = time.perf_counter()
    mandel_vars = DataPreprocessing.mandel_creation(all_mandel, mom)
    smallest_indices = []
    for index, datapoint in enumerate(mandel_vars.T):
        if np.argmin(datapoint) == mandel_index:
            smallest_indices.append(index)
    toc = time.perf_counter()
    print(f"Split ran in {toc - tic:0.4f} seconds")
    return me[smallest_indices], mom[smallest_indices]



