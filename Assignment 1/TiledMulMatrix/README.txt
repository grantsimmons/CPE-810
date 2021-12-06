Assignment 1: CUDA-based Tiled Matrix Multiplication

Author: Grant Simmons (gsimmons@stevens.edu)

Description:
This program implements a tiled matrix multiplication GPU kernel to accelerate the multiplication
of matrices of arbitrary dimensions.

Project Structure:
	MulMatrix/ - Contains source code, solution files
		x64/Debug/ - Contains compiled binaries such as MulMatrix.exe and MulMatrix_4.exe
										  (Block size = 4)
	results/ - Contains performance analysis data in CSV format under various conditions

	Assignment 1 Charts.xslx - A curated spreadsheet with graphics of analyzed performance data

	statistics.py - A performance metric gathering tool which calls and measures kernel performance

Compilation:
To compile the program, open the MulMatrix.sln file in Visual Studio. Additional linker configuration
may be required to compile on a new system. Be sure to include the header files from the CUDA examples.
These are found in NVIDIA Corporation\CUDA Samples\v11.4\common\inc. You may need to additionally include
the header files from the GPU Computing Toolkit, found in NVIDIA GPU Computing Toolkit\CUDA\v11.4\include.

Usage:
To run the program, either call the provided "./statistics.py" script which will call the executable and
collect performance statistics, or you can call the executable file directly with the following command:

	Windows:
	x64\Debug\MulMatrix.exe -hA=<Height of A> -wA=<Width of A> -hB=<Width of A> -wB=<Width of B>

This program has not been compiled for Linux, though I hope to include Linux compilation and usage
instructions in the future.