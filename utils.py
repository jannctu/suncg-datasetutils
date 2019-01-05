import numpy as np


def inBatchIndex(ind,batch_size):
    batchindex = np.linspace(batch_size * ind, (batch_size * (ind + 1))-1, num=batch_size)
    return batchindex.astype(np.int64)
