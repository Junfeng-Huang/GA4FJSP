
import numpy as np

class Job:
    def __init__(self, process) -> None:
        self.process = np.array(process, dtype=np.float32)
        #self.m_num = self.process.shape[-1]
        self.operations_num = len(process)

        self.o_avail_m = []
        for i in range(self.operations_num):
            self.o_avail_m.append(self._avail_m(i))

    def get_o_m_time(self, o_, m):
        return self.process[o_][m]

    def get_avail_m(self, operation):
        return self.o_avail_m[operation]

    def get_avail_m_time(self, operation):
        return self.process[operation][self.get_avail_m(operation)]

    def _avail_m(self, operation):
        avail = []
        for m, t in enumerate(self.process[operation]):
            if t != np.inf:
                avail.append(m)
        return avail

    
