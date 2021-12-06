Assignment 1: CUDA-based Tiled Matrix Multiplication

Author: Grant Simmons (gsimmons@stevens.edu)

Description:
This program implements a tiled matrix multiplication GPU kernel to accelerate the multiplication
of matrices of arbitrary dimensions.

Project Structure:
	TiledMatrixMul/ - Contains source code, solution files
		x64/Debug/ - Contains compiled binaries such as TiledMatrixMul.exe and TiledMatrixMul_4.exe
										  (Block size = 4)
		inc/ - Contains any additional header files required for kernel compilation

	results/ - Contains performance analysis data in CSV format under various conditions

	Assignment 1 Charts.xslx - A curated spreadsheet with graphics of analyzed performance data

	statistics.py - A performance metric gathering tool which calls and measures kernel performance

Compilation:
To compile the program, open the MulMatrix.sln file in Visual Studio. A

Compiling from the command line:
On Windows, this should be a simple matter of running the following command
from the TiledMatrixMul\ directory adjacent to this file in the tree structure:
    
    nvcc.exe kernel.cu -Iinc -o TiledMatrixMul.exe

On Linux, run the following command 

    nvcc kernel.cu -Iinc -o TiledMatrixMul.out

Usage:
To run the program, either call the provided "./statistics.py" script which will call the executable and
collect performance statistics, or you can call the executable file directly with the following command:

	Windows:
	x64\Debug\TiledMatrixMul.exe

If the file has been compiled from the command line, simply call:

    Windows:
    TiledMatrixMul.exe [-i] <rowDimA>  <colDimA>  <rowDimB>

    Linux:
    ./TiledMatrixMul.out [-i] <rowDimA>  <colDimA>  <rowDimB>
