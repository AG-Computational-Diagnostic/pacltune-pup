
import os
import subprocess
import time
import copy
import psutil
from threading import Thread
from threading import Event
from tabulate import tabulate

def query_gpus_utils():
    process = subprocess.Popen(["nvidia-smi",
                                "--query-gpu=name,index,utilization.gpu,utilization.memory",
                                "--format=csv,noheader,nounits"],
                                stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    str_out=stdout.decode('utf-8')
    int_utils=[ [int(gpu_stat) for gpu_stat in gpu_str.strip('\r').split(', ')[1:]] for gpu_str in str_out.split('\n') if gpu_str != '']
    gpus_name=[ gpu_str.strip('\r').split(', ')[0] for gpu_str in str_out.split('\n') if gpu_str != '']
    return [ [gpu_name] + gpu_utils for gpu_name, gpu_utils in zip(gpus_name, int_utils)]

def query_other_perfs():
    cpu_util=psutil.cpu_percent()
    mem_util=psutil.virtual_memory().percent
    if hasattr(psutil.cpu_times(), 'iowait'): # Linux only
        io_wait=psutil.cpu_times().iowait
    else:
        io_wait='// Linux only //'
    return cpu_util, mem_util, io_wait

def update_single_avg(old_value, new_value, old_count):
    return old_value + (new_value - old_value) / (old_count + 1)

def update_single_min(old_value, new_value):
    if old_value > new_value:
        return new_value
    else:
        return old_value

def update_single_max(old_value, new_value):
    if old_value < new_value:
        return new_value
    else:
        return old_value

# old_stats and new_stats are list of form [min, avg, max]
def update_stats(old_stats, new_value, old_count):
    new_stats = [update_single_min(old_stats[0], new_value),
                 update_single_avg(old_stats[1], new_value, old_count),
                 update_single_max(old_stats[2], new_value)]
    return new_stats

# old_gpus_sums, new_gpus_values and new new_gpus_sums are list of form [[GPU_NAME, GPU_ID, % GPU, % GPU MEM], ...]
def update_gpus_sums(old_gpus_sums, new_gpus_values, old_count, sum):
    new_gpus_sums = copy.deepcopy(old_gpus_sums)
    if sum not in {'min', 'max', 'avg'}:
        raise Exception('Choosen summary (sum=' + str(sum) +') is not valid.')
    update_call_str = 'update_single_' + sum + '(old_gpus_sums, new_gpu_val'
    if sum in {'min', 'max'}:
        update_call_str = update_call_str + ')'
    else:
        update_call_str = update_call_str + ', old_count=' + str(old_count) + ')'
    for i, new_gpu_values in enumerate(new_gpus_values):
        if i != new_gpu_values[1]:
            raise Exception('i=' + str(i) + ' is not equal GPU_ID=' + str(new_gpu_values[1]))
        if old_gpus_sums[i][0] != new_gpu_values[0]:
            raise Exception('GPU Name from previous quries (=' + old_gpus_sums[i][0] + ') is not equal GPU Name from new query (=' + str(new_gpu_values[0]) + ')')
        new_gpus_sums[i] = [new_gpu_values[0]] + [i] + [eval(update_call_str) for old_gpus_sums, new_gpu_val in zip(old_gpus_sums[i][2:], new_gpu_values[2:])]
    return new_gpus_sums

def test_nvidia_smi():
    return_code=os.system('nvidia-smi -L')
    if return_code == 0:
        return None
    else:
        return ' // ERROR in nvidia-smi -L; no Nvidia GPUs? //'

class ResMonitor(Thread):
    def __init__ (self, query_interval=1): # query_interval is in seconds
        Thread.__init__(self)
        self.exit_event=Event()
        self.query_interval=query_interval
        self.query_count=0
        self.gpus_mins=None
        self.gpus_avgs=None
        self.gpus_maxs=None
        self.cpu_util=None
        self.mem_util=None
        self.iowait=None
        self.query_duration_avg=None
        self.nv_smi_error=test_nvidia_smi()
        self.start()
    def exit(self,wait_for_exit=True):
        self.exit_event.set()
        if wait_for_exit:
            self.join()
        return self.report()
    def run(self):
        while not self.exit_event.isSet():
            start_query = time.time()
            if self.nv_smi_error == None:
                new_gpus_values = query_gpus_utils()
            new_cpu_util, new_mem_util, new_io_wait = query_other_perfs()
            end_query = time.time()
            new_query_duration = end_query - start_query
            if self.query_count == 0:
                if self.nv_smi_error == None:
                    self.gpus_mins = new_gpus_values
                    self.gpus_avgs = new_gpus_values
                    self.gpus_maxs = new_gpus_values
                self.cpu_util= 3 * [new_cpu_util]
                self.mem_util= 3 * [new_mem_util]
                self.iowait= 3 * [new_io_wait]
                self.query_duration_avg = new_query_duration
            else:
                if self.nv_smi_error == None:
                    self.gpus_mins=update_gpus_sums(old_gpus_sums=self.gpus_mins,
                                                    new_gpus_values=new_gpus_values,
                                                    old_count=self.query_count,
                                                    sum='min')
                    self.gpus_avgs=update_gpus_sums(old_gpus_sums=self.gpus_avgs,
                                                    new_gpus_values=new_gpus_values,
                                                    old_count=self.query_count,
                                                    sum='avg')
                    self.gpus_maxs=update_gpus_sums(old_gpus_sums=self.gpus_maxs,
                                                    new_gpus_values=new_gpus_values,
                                                    old_count=self.query_count,
                                                    sum='max')
                self.cpu_util=update_stats(self.cpu_util, new_cpu_util, self.query_count)
                self.mem_util=update_stats(self.mem_util, new_mem_util, self.query_count)
                if self.iowait[0] != '// Linux only //':
                    self.iowait=update_stats(self.iowait, new_io_wait, self.query_count)
                self.query_duration_avg = update_single_avg(self.query_duration_avg, new_query_duration, self.query_count)
            self.query_count+=1
            time.sleep(self.query_interval)
    def report(self):
        print('############ Ressource Overview #############')
        if self.is_alive():
            print('// MONITOR IS STILL RUNNING //')
        print('CPU & MEM: -----------------------------')
        tab_list=[['CPU'] + self.cpu_util, ['MEM'] + self.mem_util]
        print(tabulate(tab_list, headers=['', 'MIN(%)', 'AVG(%)', 'MAX(%)'], tablefmt='orgtbl', floatfmt=".2f"))
        print('IO: ------------------------------------')
        if self.iowait[0] != '// Linux only //':
            tab_list=[['Wait'] + self.iowait]
            print(tabulate(tab_list, headers=['', 'MIN', 'AVG', 'MAX'], tablefmt='orgtbl'))
        else:
            print(self.iowait[0])
        print('GPUs: =======================================')
        if self.nv_smi_error == None:
            print('MIN: ----------------------------------------')
            print(tabulate(self.gpus_mins, headers=['', 'ID', 'GPU(%)', 'Mem(%)'], tablefmt='orgtbl', floatfmt=".2f"))
            print('AVG: ----------------------------------------')
            print(tabulate(self.gpus_avgs, headers=['', 'ID', 'GPU(%)', 'Mem(%)'], tablefmt='orgtbl', floatfmt=".2f"))
            print('MAX: ----------------------------------------')
            print(tabulate(self.gpus_maxs, headers=['', 'ID', 'GPU(%)', 'Mem(%)'], tablefmt='orgtbl', floatfmt=".2f"))
            print('=============================================')
        else:
            print(self.nv_smi_error)
        print('Avg. query time:   ' + str(round(self.query_duration_avg, 4)) + 's')
        print('Number of queries: ' + str(self.query_count))
