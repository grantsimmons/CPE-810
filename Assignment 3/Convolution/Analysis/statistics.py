#!/usr/bin/python3.6

import os
import sys
import re
import subprocess
from random import randrange

stats = {
}

test = 0

iter = 1
#imageH = [250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]
imageH_min = 250
imageH_max = 4000
#imageW = [5, 125, 255, 513, 1025, 2049, 4098, 9169]
#imageH = [9169]
#imageW = [9169]
#kernelL = [1, 3, 5, 9, 13, 33, 65, 123, 251]
kernelL = [123]
#kernelL = [123, 251]
# imageH = [1025]
# imageW = [1025]
#kernelL = [13, 33, 65, 123, 251]
block_size = [8]

rand_dims = 0

if(test):
    iter = 1
    imageW = [4,128]
    imageH = [4,128]
    kernelL = [10,100000]



#measures = ["rgflops", "cgflops", "gflops", "gmem_read", "tmem_read", "gpu_time", "cpu_time"]
measures = ["gpu_time", "cpu_time"]

perf_pattern = re.compile(r'total\sGFLOPS:\s+([0-9\.]+)')
rperf_pattern = re.compile(r'rows\sGFLOPS:\s+([0-9\.]+)')
cperf_pattern = re.compile(r'columns\sGFLOPS:\s+([0-9\.]+)')
gmem_reads_pattern = re.compile(r'total\sGMEM_READS:\s+([0-9]+)')
tmem_reads_pattern = re.compile(r'total\sTMEM_READS:\s+([0-9]+)')
gmem_writes_pattern = re.compile(r'total\sGMEM_WRITES:\s+([0-9]+)')

gputime_pattern = re.compile(r'GPU_TIME:\s+([0-9\.]+)')
cputime_pattern = re.compile(r'CPU_TIME:\s+([0-9\.]+)')

fail_pattern = re.compile(r'fail')

for measure in measures:
    stats[measure] = {}
    for size in block_size:
        stats[measure][size] = {}
        for length in kernelL:
            stats[measure][size][length] = {}
            for height in range(imageH_min,imageH_max,50):
                stats[measure][size][length][height] = {}
                #for width in imageW:
                stats[measure][size][length][height][height] = {}


with open("fails.txt", 'w') as failfile:
    for i in range(rand_dims):
        kernelL.append(randrange(512))
    for size in block_size:
        for length in kernelL:
            for height in range(imageH_min,imageH_max,50):
                #for width in imageH:
                    for i in range(1,iter+1):
                        perf_val = ""
                        rperf_val = ""
                        cperf_val = ""
                        gmem_reads_val = ""
                        gmem_reads_with_atomics_val = ""
                        gmem_reads_per_element_val = ""
                        gmem_writes_val = ""
                        gmem_writes_per_element_val = ""
                        gputime_val = ""
                        cputime_val = ""

                        print("Running block_size {} for length {}, width {}, height {}".format(size, length, height, height))
                        res = 0
                        #perf_proc = subprocess.run("../convolution2D/x64/Release/convolution2D_{}.exe -i={:d} -j={:d} -k={:d}".format(block_size, height, width, length), shell=True, stdout=subprocess.PIPE)
                        perf_proc = subprocess.run("../convolution2D/x64/Release/convolution2D.exe -i={:d} -j={:d} -k={:d} -b={:d}".format(height, height, length, size), shell=True, stdout=subprocess.PIPE)
                        perf_string = perf_proc.stdout.decode('UTF-8')

                        print("String: {}".format(perf_string))

                        perf = re.search(perf_pattern, perf_string)
                        if perf:
                            perf_val = perf.group(1)
                        print("Parsed perf_val: {}".format(perf_val))

                        rperf = re.search(rperf_pattern, perf_string)
                        if rperf:
                            rperf_val = rperf.group(1)
                        print("Parsed rperf_val: {}".format(rperf_val))

                        cperf = re.search(cperf_pattern, perf_string)
                        if cperf:
                            cperf_val = cperf.group(1)
                        print("Parsed cperf_val: {}".format(cperf_val))

                        gmem_reads = re.search(gmem_reads_pattern, perf_string)
                        if gmem_reads:
                            gmem_reads_val = gmem_reads.group(1)
                        print("Parsed gmem read vals: {}".format(gmem_reads_val))

                        tmem_reads = re.search(tmem_reads_pattern, perf_string)
                        if tmem_reads:
                            tmem_reads_val = tmem_reads.group(1)
                        print("Parsed tmem read vals: {}".format(tmem_reads_val))

                        gputime = re.search(gputime_pattern, perf_string)
                        if gputime:
                            gputime_val = gputime.group(1)
                        print("Parsed gpu_time read vals: {}".format(gputime_val))
                        
                        cputime = re.search(cputime_pattern, perf_string)
                        if cputime:
                            cputime_val = cputime.group(1)
                        print("Parsed cpu_time read vals: {}".format(cputime_val))
                        # gmem_writes = re.search(gmem_writes_pattern, perf_string)
                        # if gmem_writes:
                        #     gmem_writes_val = gmem_writes.group(1)
                        # print("Parsed gmem write vals: {}".format(gmem_writes_val))

                        # stats["gflops"][size][length][height][width][i] = perf_val
                        # stats["rgflops"][size][length][height][width][i] = rperf_val
                        # stats["cgflops"][size][length][height][width][i] = cperf_val
                        #stats["gmem_read"][size][length][height][width][i] = gmem_reads_val
                        #stats["tmem_read"][size][length][height][width][i] = tmem_reads_val
                        stats["gpu_time"][size][length][height][height][i] = gputime_val
                        stats["cpu_time"][size][length][height][height][i] = cputime_val

    with open("stats_cpu_gpu_times_final.csv",'w') as fopen:
        fopen.write("statistic,,,,,")
        for measure in measures:
            fopen.write("{},".format(measure))
            for i in range(1,iter):
                fopen.write(",")
        fopen.write("\n")
        fopen.write("block_size,length,height,width,elements,")
        for measure in measures:
            for i in range(1,iter+1):
                fopen.write("iter{},".format(i))
        fopen.write("\n")
        for size in block_size:
            for length in kernelL:
                for height in range(imageH_min,imageH_max,50):
                    #for width in imageW:
                        fopen.write("{},".format(size))
                        fopen.write("{},".format(length))
                        fopen.write("{},".format(height))
                        fopen.write("{},".format(height))
                        fopen.write("{},".format(height * height))
                        for measure in measures:
                            for itr in range(1,iter+1):
                                fopen.write("{},".format(stats[measure][size][length][height][height][itr]))
                        fopen.write("\n")
        fopen.write("\n")

