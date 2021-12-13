#!/usr/bin/python3.6

import os
import sys
import re
import subprocess
from random import randrange
from math import floor

stats = {
}

test = 0

iter = 1
#imageH = [250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]
#imageH_min = 250
#imageH_max = 4000
#imageW = [5, 125, 255, 513, 1025, 2049, 4098, 9169]
#imageH = [9169]
#imageW = [9169]
#kernelL = [1, 3, 5, 9, 13, 33, 65, 123, 251]
mat_dims = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
#mat_dims = [32768, 65536]
#mat_dims = [64, 1024]
#kernelL = [123, 251]
# imageH = [1025]
# imageW = [1025]
#kernelL = [13, 33, 65, 123, 251]
block_size = [4, 8, 16, 32, 64]
#block_size = [8, 16]

rand_dims = 0

if(test):
    iter = 1
    imageW = [4,128]
    imageH = [4,128]
    kernelL = [10,100000]



#measures = ["rgflops", "cgflops", "gflops", "gmem_read", "tmem_read", "gpu_time", "cpu_time"]
measures = ["gpu_time-rl", "cpu_time", "gpu_gflops-rl", "cpu_gflops", "gpu_gflops-ll", "gpu_time-ll", "gpu_time-rl_no_tensor", "gpu_gflops-rl_no_tensor"]

perf_pattern = re.compile(r'GFlop\/s\s=\s+([0-9\.e\-\+]+)')

gputime_pattern = re.compile(r'factor time \(s\) = ([0-9\.e\-\+]+)')
cputime_pattern = re.compile(r'cpu time \(ns\):\s+([0-9\.e\-\+]+)')
cpuperf_pattern = re.compile(r'GFlop\/s:\s+([0-9\.e\-\+]+)')

fail_pattern = re.compile(r'fail')

for measure in measures:
    stats[measure] = {}
    for size in block_size:
        stats[measure][size] = {}
        for m in mat_dims:
            stats[measure][size][m] = {}


with open("fails.txt", 'w') as failfile:
    for i in range(rand_dims):
        mat_dims.append(randrange(512))
    for size in block_size:
        for m in mat_dims:
                #for width in imageH:
                    for i in range(1,iter+1):
                        gpuperf_val = ""
                        gputime_val = ""

                        cpuperf_val = ""
                        cputime_val = ""

                        print("Running block_size {} for matrix size {}".format(size, m))
                        res = 0
                        #perf_proc = subprocess.run("../convolution2D/x64/Release/convolution2D_{}.exe -i={:d} -j={:d} -k={:d}".format(block_size, height, width, length), shell=True, stdout=subprocess.PIPE)
                        perf_proc = subprocess.run("./tensor_lu --nb {:d} --m {:d}".format(size, m), shell=True, stdout=subprocess.PIPE)

                        perf_string = perf_proc.stdout.decode('UTF-8')

                        print("String: {}".format(perf_string))

                        perf = re.search(perf_pattern, perf_string)
                        if perf:
                            gpuperf_val = perf.group(1)
                        print("Parsed gpuperf_val: {}".format(gpuperf_val))

                        gputime = re.search(gputime_pattern, perf_string)
                        if gputime:
                            gputime_val = gputime.group(1)
                        print("Parsed gpu_time read vals: {}".format(gputime_val))

                        stats["gpu_gflops-rl"][size][m][i] = gpuperf_val
                        stats["gpu_time-rl"][size][m][i] = gputime_val

                        perf_proc = subprocess.run("./tensor_lu_no_tensor --nb {:d} --m {:d}".format(size, m), shell=True, stdout=subprocess.PIPE)

                        perf_string = perf_proc.stdout.decode('UTF-8')

                        print("String: {}".format(perf_string))

                        perf = re.search(perf_pattern, perf_string)
                        if perf:
                            gpuperf_val = perf.group(1)
                        print("Parsed gpuperf_val: {}".format(gpuperf_val))

                        gputime = re.search(gputime_pattern, perf_string)
                        if gputime:
                            gputime_val = gputime.group(1)
                        print("Parsed gpu_time read vals: {}".format(gputime_val))

                        stats["gpu_gflops-rl_no_tensor"][size][m][i] = gpuperf_val
                        stats["gpu_time-rl_no_tensor"][size][m][i] = gputime_val
                        
                        # perf_proc = subprocess.run("./cpu_impl/run {:d} {:d}".format(m, size), shell=True, stdout=subprocess.PIPE)
                        # perf_string = perf_proc.stdout.decode('UTF-8')

                        # cpuperf = re.search(cpuperf_pattern, perf_string)
                        # if cpuperf:
                        #     cpuperf_val = cpuperf.group(1)
                        # print("Parsed cpuperf_val: {}".format(cpuperf_val))

                        # cputime = re.search(cputime_pattern, perf_string)
                        # if cputime:
                        #     cputime_val = cputime.group(1)
                        # print("Parsed cpu_time read vals: {}".format(cputime_val))

                        # perf_proc = subprocess.run("../remifa_testing/remifa/build_2/tests/testing_lu_gpu --algo=remifa-ll --nb {:d} --m {:d} --diagdom --out-upd=tc32 --in-upd=tc32".format(size, m), shell=True, stdout=subprocess.PIPE)

                        # perf_string = perf_proc.stdout.decode('UTF-8')

                        # print("String: {}".format(perf_string))

                        # perf = re.search(perf_pattern, perf_string)
                        # if perf:
                        #     gpuperf_val = perf.group(1)
                        # print("Parsed gpuperf_val: {}".format(gpuperf_val))

                        # gputime = re.search(gputime_pattern, perf_string)
                        # if gputime:
                        #     gputime_val = gputime.group(1)
                        # print("Parsed gpu_time read vals: {}".format(gputime_val))



                        stats["gpu_gflops-ll"][size][m][i] = gpuperf_val
                        stats["gpu_time-ll"][size][m][i] = gputime_val
                        stats["cpu_time"][size][m][i] = cputime_val
                        stats["cpu_gflops"][size][m][i] = cpuperf_val

    with open("stats_cpu_gpu_times_final.csv",'w') as fopen:
        fopen.write("statistic,,,,,")
        for measure in measures:
            fopen.write("{},".format(measure))
            for i in range(1,iter):
                fopen.write(",")
        fopen.write("\n")
        fopen.write("block_size,m,")
        for measure in measures:
            for i in range(1,iter+1):
                fopen.write("iter{},".format(i))
        fopen.write("\n")
        for size in block_size:
            for m in mat_dims:
                    #for width in imageW:
                        fopen.write("{},".format(size))
                        fopen.write("{},".format(m))
                        #fopen.write("{},".format(height * height))
                        for measure in measures:
                            for itr in range(1,iter+1):
                                fopen.write("{},".format(stats[measure][size][m][itr]))
                        fopen.write("\n")
        fopen.write("\n")