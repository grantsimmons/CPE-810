#include "solver.hh"
#include <stdlib.h>

int main(int argc, char** argv) {
    //Read Matrix
    int size = 8;
    int block_size = 8;

    double test_arr[] = {10, 4, -3, 4, 5, 9, 1, 9,
                        3, -9, 0, -9, 10, 2, 4, 6,
                        -3, -6, 26, 2, 1, 3, 7, 8,
                        3, -4, 8, 9, 1, 5, 12, 4,
                        -1, 2, 4, 7, 4, -2, 9, 2,
                        9, 1, 1, 9, 8, -4, 5, -2,
                        1, -2, -2, -3, 2, 4, 5, 4,
                        3, 4, -2, 3, 4, 2, 3, 5};
    
    size = 1000;
    block_size = 100;

	if(argc == 3) {
		size = atoi(argv[1]);
		block_size = atoi(argv[2]);
	}
	else {
		exit(1);
	}

	std::cout << "Size: " << size << ", Block Size: " << block_size << std::endl;
    float* test_arr_real = (float*) malloc(size * size * sizeof(float));

     float k = 0;
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            test_arr_real[i*size + j] = k++;
        }
    }

    // SerialMatrix<double> test(test_arr, size, size, block_size);
    SerialMatrix<float> test(test_arr_real, size, size, block_size);
    // test.print();

    Solver matrix_solver = Solver(test);
    matrix_solver.solve();

    // test.print();

    return 0;
}
