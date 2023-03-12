import numpy as np
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def run(long[::1] signals, double[::1] h, double[::1] l, double[::1] SL, double[::1] TP, int direction):
    cdef:
        int outRows = signals.shape[0]
        int hlRows = h.shape[0]

        long[::1] out = np.zeros((outRows), dtype=np.int64) # output, same lenght as signals

        int s
        int i
        long[::1] entryIdx = np.add(signals,1)
        int idx

    for s in range(outRows):
        idx = entryIdx[s] # start from the next candle after signal
        if direction==1:
            for i in range(idx,hlRows):
                if h[i] >= TP[s]: out[s] = i;break
                if l[i] <= SL[s]: out[s] = i;break
            
        #for shorts
        if direction==-1:
            for i in range(idx,hlRows):
                if h[i] >= SL[s]: out[s] = i;break
                if l[i] <= TP[s]: out[s] = i;break

    return np.asarray(out)