
import copy

class Machine:
    def __init__(self, lable) -> None:
        self.label = lable
        self.load_operations = [] # 保存的是元组，第一个元素是工件号，第二个是工序
        self.time_windows = []
        self.last_end = 0

    def reset(self):
        self.load_operations = []
        self.time_windows = []  
        self.last_end = 0

    def add_operation(self, operation, m_t, pre_operation_end_time):
        operation_end_time = 0.
        if len(self.load_operations) == 0:
            self.load_operations.append(operation)
            time_window = (pre_operation_end_time, pre_operation_end_time + m_t)
            self.time_windows.append(time_window)
            operation_end_time = pre_operation_end_time + m_t
        else:
            free_windows = self.get_free_windows()
            for free_w in free_windows:
                t = max(free_w[0], pre_operation_end_time)
                if t + m_t < free_w[1]: # 不能<=因为时间窗口是前闭后开[start, end)
                    operation_end_time = t + m_t
                    time_window = (t, operation_end_time)
                    insert_index = 0
                    for w_index, w in enumerate(self.time_windows): # 可bs优化
                        if w[1] == free_w[0]:
                            insert_index = w_index + 1
                            break
                    self.load_operations.insert(insert_index, operation)
                    self.time_windows.insert(insert_index, time_window)
                    break
        if self.last_end < operation_end_time:
            self.last_end = operation_end_time
        return operation_end_time

    def get_free_windows(self,) -> list:
        free_windows = []
        pre_w = self.time_windows[0]
        for w in self.time_windows[1:]:
            free_w = [pre_w[1], w[0]]
            free_windows.append(free_w)
            pre_w = w
        free_w = [pre_w[1], float("inf")]
        free_windows.append(free_w)
        return free_windows


class MachineProcess:
    def __init__(self, m_num) -> None:
        self.m_num = m_num
        self.m = [] 
        for label in range(self.m_num):
            self.m.append(Machine(label))
        self._index = 0

    def reset(self):
        for i in range(self.m_num):
            self.m[i].reset()
        self._index = 0

    def get_spend_time(self):
        spend_time = float("-inf")
        for i in range(self.m_num):
            if self.m[i].last_end > spend_time:
                spend_time = self.m[i].last_end
        return spend_time

    def get_arrange(self) -> list:
        """
        返回list：[(load_operations, time_windows, last_end),]
        """
        result = []
        for i in range(self.m_num):
            load_operations = copy.deepcopy(self.m[i].load_operations)
            time_windows = copy.deepcopy(self.m[i].time_windows)
            last_end = self.m[i].last_end
            arrange = (load_operations, time_windows, last_end)
            result.append(arrange)
        return result

    def __len__(self):
        return len(self.m)

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < self.m_num:
            r = self.m[self._index]
            self._index += 1
            return r
        else:
            raise StopIteration

    def __getitem__(self, key):
        return self.m[key]
