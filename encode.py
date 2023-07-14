from ast import Lambda
import numpy as np
import random

class Encode:
    def __init__(self, jobs, m_num, pop=300, mixs=0.2, gs=0.4, ls=0.2, rs=0.2):
        self.pop = pop
        self.mixs_size = int(mixs * pop)
        self.gs_size = int(gs * pop)
        self.ls_size = int(ls * pop)
        self.rs_size = int(rs * pop)
        assert pop == self.mixs_size + self.gs_size + self.ls_size + self.rs_size, "Error of proportion"

        self.jobs = jobs
        self.jobs_num = len(self.jobs)
        self.chs_len = 0
        for j in self.jobs:
            self.chs_len += j.operations_num

        self.m_num = m_num

        self.accumulation_job_operations = []
        s = 0
        for i in range(self.jobs_num):
            self.accumulation_job_operations.append(s)
            s += self.jobs[i].operations_num
        
    def initialize(self):
        mixs_chs = self.mixs_initial()
        gs_chs = self.gs_initial()
        ls_chs = self.ls_initial()
        rs_chs = self.rs_initial()

        return np.vstack((mixs_chs, gs_chs, ls_chs, rs_chs))

    def mixs_initial(self):
        """
        mix select:结合不同机器的处理时间和相应机器的总处理时间，选择机器
        和原生mix算法相比，这个算法没有先对仅能在特定机器上运行的工件进行预分配。
        这种情况比较少，而且容易使得某一类型工件在特定机器的排列太紧，是否改变还需斟酌。
        """
        def choose(avail_m, o_time, machine_total_time):
            o_avail_time = o_time[avail_m]
            avail_total_time = machine_total_time[avail_m]
            sort_info = sorted(zip(range(len(avail_m)), o_avail_time, avail_total_time),
                              key=lambda x:0.5*x[0]*x[0]+x[1])
            return sort_info[0][0]
        # 表示机器选择的种群染色体,每条染色体下标表示按
        ms = self.chs_matrix(size=self.mixs_size)
        # 表示工序选择的种群染色体
        os = self.chs_matrix(size=self.mixs_size)
        for i in range(self.mixs_size):
            os_random = self.os_random_gen()
            os[i] = os_random
            machine_total_time = np.array([0.0 for _ in range(self.m_num)], dtype=np.float32)
            job_index = [i for i in range(self.jobs_num)]
            random.shuffle(job_index)
            for j in job_index:
                job = self.jobs[j]
                for o_ in range(job.operations_num):
                    o_time = job.process[o_]
                    avail_m = job.get_avail_m(o_)
                    relative_m_index = choose(avail_m, o_time, machine_total_time)

                    m_index = avail_m[relative_m_index]
                    machine_total_time[m_index] += o_time[m_index]
                    site = self.ms_site(j, o_)
                    ms[i][site] = relative_m_index
        return np.hstack((ms, os))

    def gs_initial(self):
        """
        global select:以全局机器负载时间最小为目标分配工序，此处未考虑工序先后（间隔）问题，
                      仅仅是将每个机器上的工序时间加和。
        """

        # 表示机器选择的种群染色体,每条染色体下标表示按
        ms = self.chs_matrix(size=self.gs_size)
        # 表示工序选择的种群染色体
        os = self.chs_matrix(size=self.gs_size)
        for i in range(self.gs_size):
            os_random = self.os_random_gen()
            os[i] = os_random
            machine_total_time = np.array([0.0 for _ in range(self.m_num)], dtype=np.float32)
            job_index = [i for i in range(self.jobs_num)]
            random.shuffle(job_index)
            for j in job_index:
                job = self.jobs[j]
                for o_ in range(job.operations_num):
                    spend_time = machine_total_time + job.process[o_]
                    avail_m = job.get_avail_m(o_)
                    select_time = spend_time[avail_m]
                    relative_m_index = select_time.argmin()

                    m_index = avail_m[relative_m_index]
                    machine_total_time[m_index] += spend_time[m_index]
                    site = self.ms_site(j, o_)
                    ms[i][site] = relative_m_index
        return np.hstack((ms, os))

    def ls_initial(self):
        ms = self.chs_matrix(size=self.ls_size)
        os = self.chs_matrix(size=self.ls_size)
        for i in range(self.ls_size):
            os_random = self.os_random_gen()
            os[i] = os_random
            job_index = [i for i in range(self.jobs_num)]
            random.shuffle(job_index)
            for j in job_index:
                job = self.jobs[j]
                machine_total_time = np.array([0 for _ in range(self.m_num)], dtype=np.float32)
                for o_ in range(job.operations_num):
                    spend_time = machine_total_time + job.process[o_]
                    avail_m = job.get_avail_m(o_)
                    select_time = spend_time[avail_m]
                    relative_m_index = select_time.argmin()

                    m_index = avail_m[relative_m_index]
                    machine_total_time[m_index] += spend_time[m_index]
                    site = self.ms_site(j, o_)
                    ms[i][site] = relative_m_index
        return np.hstack((ms, os))

    def rs_initial(self):
        ms = self.chs_matrix(size=self.rs_size)
        os = self.chs_matrix(size=self.rs_size)
        for i in range(self.rs_size):
            os_random = self.os_random_gen()
            os[i] = os_random
            job_index = [i for i in range(self.jobs_num)]
            #random.shuffle(job_index)
            for j in job_index:
                job = self.jobs[j]
                for o_ in range(job.operations_num):
                    avail_m = job.get_avail_m(o_)
                    relative_m_index = np.random.randint(0, len(avail_m))
                    site = self.ms_site(j, o_)
                    ms[i][site] = relative_m_index
        return np.hstack((ms, os))

    def chs_matrix(self, size):
        return np.zeros([size, self.chs_len], dtype=np.int32)

    def os_random_gen(self, rand=True):
        os = np.zeros([self.chs_len,], dtype=np.int32)
        i = 0
        for j in range(self.jobs_num):
            job = self.jobs[j]
            for _ in range(job.operations_num):
                os[i] = j
                i += 1
        if rand:
            np.random.shuffle(os)
        return os

    def ms_site(self, j, o_):
        return self.accumulation_job_operations[j] + o_
