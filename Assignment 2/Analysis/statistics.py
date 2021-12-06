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
num_bins = [4,8,16,32,64,128,256,512]
block_widths = [4,8,16,32,64,128,256,512]
vec_dims = [1,10,100,1000,10000,100000,1000000,10000000]

rand_dims = 0

if(test):
    iter = 1
    block_widths = [4,128]
    num_bins = [4,128]
    vec_dims = [10,100000]



measures = ["gflops", "gmem_read", "gmem_read_with_atomic", "gmem_read_per_elem", "gmem_write", "gmem_write_per_elem"]

perf_pattern = re.compile(r'GFLOPS:\s+([0-9\.]+)')
gmem_reads_pattern = re.compile(r'Global\sMemory\sReads:\s+([0-9]+)\s+\(Including atomic operations\):\s([0-9\.]+)\s+\(Per Element\):\s([0-9\.]+)', re.MULTILINE)
gmem_writes_pattern = re.compile(r'Global Memory Writes: ([0-9\.]+)\s+\(Per Element\): ([0-9.]+)', re.MULTILINE)
fail_pattern = re.compile(r'FAIL')

for measure in measures:
    stats[measure] = {}
    for arr_size in vec_dims:
        stats[measure][arr_size] = {}
        for bin_count in num_bins:
            stats[measure][arr_size][bin_count] = {}
            for size in block_widths:
                stats[measure][arr_size][bin_count][size] = {}


with open("fails.txt", 'w') as failfile:
    for i in range(rand_dims):
        vec_dims.append(randrange(512))
    for arr_size in vec_dims:
        for bin_count in num_bins:
            for size in block_widths:
                for i in range(1,iter+1):
                    perf_val = ""
                    gmem_reads_val = ""
                    gmem_reads_with_atomics_val = ""
                    gmem_reads_per_element_val = ""
                    gmem_writes_val = ""
                    gmem_writes_per_element_val = ""

                    print("Running block size {} for array size {}, bin count {}".format(size, arr_size, bin_count))
                    res = 0
                    perf_proc = subprocess.run("./x64/Debug/Histogram_{}.exe -block_width={:d} -vec_dim={:d}".format(bin_count, size, arr_size), shell=True, stdout=subprocess.PIPE)
                    perf_string = perf_proc.stdout.decode('UTF-8')

                    print("String: {}".format(perf_string))

                    perf = re.search(perf_pattern, perf_string)
                    if perf:
                        perf_val = perf.group(1)
                    print("Parsed perf_val: {}".format(perf_val))

                    gmem_reads = re.search(gmem_reads_pattern, perf_string)
                    if gmem_reads:
                        gmem_reads_val = gmem_reads.group(1)
                        gmem_reads_with_atomics_val = gmem_reads.group(2)
                        gmem_reads_per_element_val = gmem_reads.group(3)
                    print("Parsed gmem read vals: {},{},{}".format(gmem_reads_val, gmem_reads_with_atomics_val, gmem_reads_per_element_val))

                    gmem_writes = re.search(gmem_writes_pattern, perf_string)
                    if gmem_writes:
                        gmem_writes_val = gmem_writes.group(1)
                        gmem_writes_per_element_val = gmem_writes.group(2)
                    print("Parsed gmem write vals: {},{}".format(gmem_writes_val, gmem_writes_per_element_val))

                    stats["gflops"][arr_size][bin_count][size][i] = perf_val
                    stats["gmem_read"][arr_size][bin_count][size][i] = gmem_reads_val
                    stats["gmem_read_with_atomic"][arr_size][bin_count][size][i] = gmem_reads_with_atomics_val
                    stats["gmem_read_per_elem"][arr_size][bin_count][size][i] = gmem_reads_per_element_val
                    stats["gmem_write"][arr_size][bin_count][size][i] = gmem_writes_val
                    stats["gmem_write_per_elem"][arr_size][bin_count][size][i] = gmem_writes_per_element_val

    with open("stats.csv",'w') as fopen:
        fopen.write("statistic,,,")
        for measure in measures:
            fopen.write("{},".format(measure))
            for i in range(1,iter):
                fopen.write(",")
        fopen.write("\n")
        fopen.write("arr_size,bin_count,block_size,")
        for measure in measures:
            for i in range(1,iter+1):
                fopen.write("iter{},".format(i))
        fopen.write("\n")
        for asize in vec_dims:
            for bins in num_bins:
                for bsize in block_widths:
                    fopen.write("{},".format(asize))
                    fopen.write("{},".format(bins))
                    fopen.write("{},".format(bsize))
                    for measure in measures:
                        for itr in range(1,iter+1):
                            fopen.write("{},".format(stats[measure][asize][bins][bsize][itr]))
                    fopen.write("\n")
        fopen.write("\n")

