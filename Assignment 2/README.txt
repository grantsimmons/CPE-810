Assignment 2: CUDA-based Histogram Computation

Author: Grant Simmons (gsimmons@stevens.edu)

Description:
This program implements a privatized histogram computation GPU kernel to accelerate the histogram 
computation for data sets of arbitrary size.

Project Structure:
	Histogram/ - Contains source code, solution files
		x64/Debug/ - Contains compiled binaries such as Histogram.exe and Histogram_4.exe
                                                                            (BinCount = 4)

	Analysis/ - Contains performance analysis data in CSV format under various conditions
                Contains script used to generate CSV data
                Contains graphs of performance and memory operations for 
                    changing system parameters

Compiling (Visual Studio):
To compile the program, open the Histogram.sln file in Visual Studio. Additional linker configuration
may be required to compile on a new system. Be sure to include the header files from the CUDA examples.
These are found in NVIDIA Corporation\CUDA Samples\v11.4\common\inc. You may need to additionally include
the header files from the GPU Computing Toolkit, found in NVIDIA GPU Computing Toolkit\CUDA\v11.4\include.

Compiling from the command line:
On Windows, this should be a simple matter of running the following command
from the Histogram\ directory adjacent to this file in the tree structure:
    
    nvcc.exe kernel.cu -Iinc -o Histogram.exe

On Linux, run the following command:

    nvcc kernel.cu -Iinc -o Histogram.out

Usage:
To run the program, either call the provided "./statistics.py" script which will call the executable and
collect performance statistics, or you can call the executable file directly with the following command:

	Windows:
	x64\Debug\Histogram.exe -vec_dim=<Input array length> -bin_count=<Number of histogram bin intervals>

If the file has been compiled from the command line, simply call:

    Windows:
    Histogram.exe -vec_dim=<Input array length> -bin_count=<Number of histogram bin intervals>

    Linux:
    ./Histogram.out -vec_dim=<Input array length> -bin_count=<Number of histogram bin intervals>
