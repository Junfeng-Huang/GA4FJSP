
from machine import Machine, MachineProcess
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools
from job import Job
from encode import Encode
from decode import Decode
import numpy as np
# np.random.seed(0)

class GA:
    def __init__(self, workpieces, pop_size=300, mixs=0.2, gs=0.4, ls=0.2, rs=0.2) -> None:
        """
        workpieces:嵌套三层列表.第一层表示工件;第二层表示每个工件的工序;第三层表示工序在
                   各个机器上的处理时间，若某工序不能在某机器上处理，时间表示为inf。
        """

        self.m_num = len(workpieces[0][0])
        #self.mp = MachineProcess(self.m_num)
        
        self.jobs = []   # readonly
        self.jobs_num = len(workpieces)
        self.total_operations = 0
        for p in workpieces:
            self.jobs.append(Job(p))
            self.total_operations += len(p)

        self.chs_len = self.total_operations
        self.pop_size = pop_size

        self.encode = Encode(self.jobs, self.m_num, pop_size, mixs, gs, ls, rs)
        self.decode = Decode(self.jobs, self.chs_len)
        
        self.best_fit_value = float("inf")
        self.best_arrange = None
    
    def run(self, iterations=10, k=3, p_c=0.8, p_v=0.5, p_m=0.01):
        """
        种群初始化（染色体编码，种群生成）-> 染色体解码 -> 评价 -> 选择 -> 交叉 -> 变异
        """
        pop = self.encode.initialize()

        mp = MachineProcess(self.m_num)

        for it in range(iterations):
            print("{} iterations".format(it))
            fitness = []
            for i in range(len(pop)):
                chs = pop[i]
                self.decode.decode4fjsp(chs, mp)
                fit_value = mp.get_spend_time()
                fitness.append(fit_value)
                if fit_value < self.best_fit_value:
                    self.best_fit_value = fit_value
                    self.best_arrange = mp.get_arrange()
                mp.reset()
            pop = self.select(pop, fitness, k=k)
            print("best fit value:", self.best_fit_value)
            pop[:,0:self.chs_len] = self.machine_cross(pop[:,0:self.chs_len], p_c)
            pop[:,self.chs_len:] = self.operation_cross(pop[:,self.chs_len:], p_c)
            
            self.machine_mutation(p_m, pop)
            # opearion_mutation对于全是单工序的工件问题可以不开启。
            # self.operation_mutation(p_m, pop)

    def machine_cross(self, ms, p_c):
        cross_ms = np.zeros_like(ms, dtype=np.int32)
        pop_range = [i for i in range(self.pop_size)]
        chs_range = [i for i in range(self.chs_len)]
        for i in range(0, (self.pop_size//2)*2, 2): 
            parents_index = np.random.choice(pop_range, 2, replace=False)
            parents = ms[parents_index]
            if np.random.rand() < p_c:
                r = np.random.randint(1, self.chs_len, size=1)
                r_site = np.random.choice(chs_range, r, replace=False,)
                parents[0][r_site], parents[1][r_site] = parents[1][r_site], parents[0][r_site]
            cross_ms[[i,i+1]] = parents
        if self.pop_size != (self.pop_size//2)*2:
            cross_ms[-1] = ms[-1]
        return cross_ms

    def operation_cross(self, os, p_c):
        cross_os = np.zeros_like(os, dtype=np.int32)
        pop_range = [i for i in range(self.pop_size)]
        job_range = [i for i in range(self.jobs_num)]

        for i in range(0, (self.pop_size//2)*2, 2):
            parents_index = np.random.choice(pop_range, 2, replace=False)
            parents = os[parents_index]
            # print(parents[0])
            # print(parents[1])
            if np.random.rand() < p_c:
                r = np.random.randint(1, self.jobs_num, size=1)
                r_jobs = np.random.choice(job_range, size=r, replace=False)
                # print("---",r_jobs)
                r_site1 = True
                r_site2 = True
                for job in r_jobs:
                    r_site1 = r_site1 & (parents[0]!=job)
                    r_site2 = r_site2 & (parents[1]!=job )
                parents[0][r_site1], parents[1][r_site2] = parents[1][r_site2], parents[0][r_site1]
                # print([int(r) for r in r_site1])
                # print([int(r) for r in r_site2])
                # print(parents[0])
                # print(parents[1])
            cross_os[[i, i+1]] = parents
        if self.pop_size != (self.pop_size//2)*2:
            cross_os[-1] = os[-1]
        return cross_os

    def machine_mutation(self, p_m, pop):

        def m_site2o_(o_nums, m_site):
            total_o_num = 0
            for j_index, o_num in enumerate(o_nums):
                total_o_num += o_num
                if total_o_num > m_site: # 不能等于，从0开始的前闭后开
                    return j_index, m_site - total_o_num + o_num
            raise Exception("convert error")
        mutation_pop_size = int(self.pop_size * p_m)
        mutation_pop_index = np.random.choice(range(self.pop_size),
                                        size = mutation_pop_size,
                                        replace=False)
        o_nums = []
        for job in self.jobs:
            o_nums.append(job.operations_num)
        for i in range(mutation_pop_size):
            r = np.random.randint(1, self.chs_len, size=1)
            r_sites = np.random.choice(range(self.chs_len), size=r, replace=False)
            chs_index = mutation_pop_index[i]
            for r_s in r_sites:
                o_site = m_site2o_(o_nums, r_s)
                avail_m_time = self.jobs[o_site[0]].get_avail_m_time(o_site[1])
                pop[chs_index][r_s] = avail_m_time.argmin()

    def operation_mutation(self, p_m, pop):
        """
        一种做法是生成随机r个位置的所有邻域排序，再一一评估，但对于r有限制，O(r!)
        此处暂不考虑其他做法。
        """
        mutation_pop_size = int(self.pop_size * p_m)
        mutation_pop_index = np.random.choice(range(self.pop_size),
                                        size = mutation_pop_size,
                                        replace=False)
        mp = MachineProcess(self.m_num)
        for index in mutation_pop_index:
            fitness = []
            chs = pop[index]
            # r = np.random.randint(1, self.chs_len, size=1)
            r = np.random.randint(2, 5, size=1)
            r_site = np.random.choice(range(self.chs_len, 2*self.chs_len), size=r, replace=False)
            arange_r_site = list(itertools.permutations(r_site,len(r_site)))
            for a_r_s in arange_r_site:
                a_r_s = list(a_r_s)
                temp_chs = chs.copy()
                temp_chs[r_site] = temp_chs[a_r_s]
                self.decode.decode4fjsp(temp_chs, mp)
                fitness.append(mp.get_spend_time())
                mp.reset()
            mutant_gene_site = fitness.index(min(fitness))
            pop[index][r_site] = pop[index][list(arange_r_site[mutant_gene_site])]

        
    def select(self, pop, fitness, k=3):
        new_pop = np.zeros_like(pop, dtype=np.int32)
        fitness = np.array(fitness, dtype=np.float32)
        pop_range = [i for i in range(self.pop_size)]
        for i in range(self.pop_size):
            random_k = np.random.choice(pop_range, size=k, replace=False)
            new_pop[i] = pop[random_k[np.argmin(fitness[random_k])]]
        return new_pop

    def plot(self,):
        plt.figure()
        ax = plt.subplot(111)
        ax.set_xlim((0,95))
        ax.set_xticks(range(0,95))
        ax.set_ylim((0,11))
        ax.set_yticks([0.4,1.4,2.4,3.4,4.4,5.4,6.4,7.4,8.4,9.4,10.4])
        ax.set_yticklabels(["A","B","C","D","E","F","G","H","I","J","K"])
        ax.set_ylabel("Device")
        ax.set_xlabel("Time")
        best_range = self.best_arrange
        j = 0
        for r in best_range:
            m_No, arange, fit_value = r
            for i in range(len(arange)) :
                ax.add_patch(patches.Rectangle(
                    (arange[i][0], j),
                    width=arange[i][1] - arange[i][0],
                    height=0.8,
                    fill=False,
                    linewidth=1,
                ))
                ax.text(arange[i][0]+(arange[i][1] - arange[i][0])/2, j+0.3,s=str(m_No[i][0]))
            j+=1
            print(m_No)
            print("__________")
        plt.show()