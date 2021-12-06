Assignment 3: CUDA-based 2D Convolution Computation

Author: Grant Simmons (gsimmons@stevens.edu)

Description:
This program implements a texture memory-based 2D convolution computation GPU kernel to accelerate the convolution 
computation for data sets of arbitrary size.

Project Structure:
	convolution2D/ - Contains source code, solution files
		x64/Release/ - Contains compiled binaries such as convolution2D.exe

	Analysis/ - Contains performance analysis data in CSV format under various conditions
                Contains script used to generate CSV data
                Contains graphs of performance and memory operations for 
                    changing system parameters

Compiling (Visual Studio):
To compile the program, open the convolution2D.sln file in Visual Studio. Additional linker configuration
may be required to compile on a new system. Be sure to include the header files from the CUDA examples.
These are found in NVIDIA Corporation\CUDA Samples\v11.4\common\inc. You may need to additionally include
the header files from the GPU Computing Toolkit, found in NVIDIA GPU Computing Toolkit\CUDA\v11.4\include.

Compiling from the command line:
On Windows, this should be a simple matter of running the following command
from the convolution2D\ directory adjacent to this file in the tree structure:
    
    nvcc.exe kernel.cu main.cpp convolutionTexture_gold.cpp -Iinc -o convolution2D.exe

On Linux, run the following command:

    nvcc kernel.cu main.cpp convolutionTexture_gold.cpp -Iinc -o convolution2D

Usage:
To run the program, either call the provided "./statistics.py" script which will call the executable and
collect performance statistics, or you can call the executable file directly with the following command:

	Windows:
	x64\Release\convolution2D.exe -i=<dimX> -j=<dimY> -k=<dimK>

If the file has been compiled from the command line, simply call:

    Windows:
    convolution2D.exe -i=<dimX> -j=<dimY> -k=<dimK>

    Linux:
    ./convolution2D -i=<dimX> -j=<dimY> -k=<dimK>
