#!/usr/bin/python3.6

import os
import sys
import re
import subprocess
from random import randrange

stats = {
}

iter = 1

#dims = [1,2,4,8,16,32,64,128,256,512,1024]
#dims = [3,82,164,360,374,403,467]
#dims = [3, 61, 125, 259, 509]
#dims = [3,82,164,360,374,403,467,1101,2049,5000]
dims = [64]
dims_outer = [64]
rand_dims = 0

#block_size = [2,4,8,16,32]
block_size = [16,32]

pattern = re.compile(r'([\d.]+)\sGFlop')
pattern_fail = re.compile(r'FAIL')

with open("fails.txt", 'w') as failfile:
    for i in range(rand_dims):
        dims.append(randrange(512))
    print(dims)
    for size in block_size:
        stats[size] = {}
        for dim1 in dims: #controls output.height
            for dim2 in dims_outer: #controls outer matrix dimension
                #for dim3 in dims: #controls output.width
                    #key = "{}x{}:{}x{}".format(dim2,dim1,dim1,dim3)
                    key = "{}x{}:{}x{}".format(dim2,dim1,dim1,dim2)
                    stats[size][key] = []
                    for i in range(0,iter):
                        print("Running {} for block size {}".format(key, size))
                        res = 0
                        #perf_proc = subprocess.run("./MulMatrix/x64/Debug/MulMatrix_{}.exe -hA={:d} -wA={:d} -hB={:d} -wB={:d}".format(size,dim2,dim1,dim1,dim3), shell=True, stdout=subprocess.PIPE)
                        perf_proc = subprocess.run("./MulMatrix/x64/Debug/MulMatrix_{}.exe -hA={:d} -wA={:d} -hB={:d} -wB={:d}".format(size,dim2,dim1,dim1,dim2), shell=True, stdout=subprocess.PIPE)
                        perf_string = perf_proc.stdout.decode('UTF-8')
                        print(perf_string)
                        perf = re.search(pattern, perf_string).group(1)
                        fail = re.match(pattern_fail, perf_string)
                        if(fail):
                            print("Fail: {}, {}".format(size, key))
                        print("Performance: {}".format(perf))
                        #run command
                        stats[size][key].append(perf)

        with open("stats_{}.csv".format(size),'w') as fopen:
            fopen.write("Size: {}\n".format(size))
            for dim in stats[size]:
                print(stats[size][dim])
                fopen.write("{},".format(dim))
                for i in stats[size][dim]:
                    fopen.write("{},".format(i))
                fopen.write("\n")
