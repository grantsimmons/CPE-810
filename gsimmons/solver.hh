#include "serial_matrix.hh"
#include <deque>
#include <omp.h>
#include <functional>
#include <chrono>

#ifndef SOLVER_H_
#define SOLVER_H_

#define OMP_SUPPORT

template <typename T>
class Solver{
private:
    SerialMatrix<T>& matrix_;

    std::vector<std::vector<Block<T>*>> block_test_;

    const int kNumCategories = 4;

    int current_solve_level_;

public:
    Solver(SerialMatrix<T>& matrix) : matrix_(matrix), current_solve_level_(0){
        block_test_ = std::vector<std::vector<Block<T>*>>(kNumCategories);
        clearAndInitBlockVectors();
    }

    void routeBlock(Block<T>& this_block, category cat) {
        //std::cout << "Routing block" << std::endl;
        this_block.setCategory(cat);
        //std::cout << this_block.getCoordinate().row << std::endl;
        //std::cout << "Block coordinates: " << this_block.getCoordinate().row << ", " << this_block.getCoordinate().col << std::endl;

        //This isn't the best way, but it's easier than initializing a vector of reference_wrappers
        block_test_[cat].push_back(&this_block); //Guarantees Vector location
    }

    void clearAndInitBlockVectors() {
        //Initialize category 0
        //std::cout << "Clearing and initializing block category arrays" << std::endl;
        //std::cout << "Resizing block category arrays" << std::endl;
        block_test_[kRightLooking].clear();
        block_test_[kUpperSolve].clear();
        block_test_[kLowerSolve].clear();
        block_test_[kSchurComp].clear();
        //std::cout << "Finished resizing block arrays" << std::endl;
        
    }

    void initSolver() {
        int i = current_solve_level_;

        //Set category for diagonal element
        //std::cout << "Initializing solver" << std::endl;
        matrix_.getBlock(i,i).setCategory(kRightLooking);

        for(int block_row = i; block_row < matrix_.getBlockRows(); block_row++) {
            for(int block_col = i; block_col < matrix_.getBlockColumns(); block_col++) {
                //std::cout << "Calculating category for block " << block_row << ", " << block_col << std::endl;

                //Default to processed
                category block_category = kProcessed;
                if(block_row == i && block_col == i) {
                    block_category = kRightLooking;
                }
                else if (block_row == i) {
                    block_category = kUpperSolve;
                }
                else if (block_col == i) {
                    block_category = kLowerSolve;
                }
                else if (block_col > i && block_row > i) {
                    block_category = kSchurComp;
                }

                //std::cout << "Pushing block " << block_row << " " << block_col << " to block queue " << block_category << std::endl;
                routeBlock(matrix_.getBlock(block_row, block_col), block_category);


            }
        }
        //std::cout << "Checking array sizes..." << std::endl;

        if (block_test_[kRightLooking].size() != 1 
            || block_test_[kLowerSolve].size() != matrix_.getBlockColumns() - current_solve_level_ - 1
            || block_test_[kUpperSolve].size() != matrix_.getBlockRows() - current_solve_level_ - 1
            || block_test_[kSchurComp].size() != (matrix_.getBlockRows() - current_solve_level_ - 1) * (matrix_.getBlockColumns() - current_solve_level_ - 1)) {
            std::cout << "Cat 1 " << block_test_[kRightLooking].size() << std::endl;
            std::cout << "Cat 2 " << block_test_[kLowerSolve].size() << std::endl;
            std::cout << "Cat 3 " << block_test_[kUpperSolve].size() << std::endl;
            std::cout << "Cat 4 " << block_test_[kSchurComp].size() << std::endl;
            throw std::runtime_error("Error: Category queues are not sized properly!");
        }
    }

    void solve() {
        auto start = std::chrono::high_resolution_clock::now();

        while(current_solve_level_ < matrix_.getBlockColumns()) {
            //Clear vector
            clearAndInitBlockVectors();

            //Recalculate block categories
                //1, 2, and 3 move to 0
                //4 recalculated to 1, 2, 3, and 4
            initSolver();

            //Calculate Category 1 block
            Block<T>& thisDiagonalBlock = *(block_test_[kRightLooking][0]);
            thisDiagonalBlock.solve(kRightLooking);

            //Calculate Category 2 and 3 blocks (parallel)
#ifdef OMP_SUPPORT
            #pragma omp parallel
            {
                #pragma omp for nowait
#endif
                for(int i = 0; i < block_test_[kUpperSolve].size(); i++) {
                    block_test_[kUpperSolve][i]->solve(kUpperSolve, thisDiagonalBlock);
                }

#ifdef OMP_SUPPORT
                #pragma omp for nowait
#endif
                for(int i = 0; i < block_test_[kLowerSolve].size(); i++) {
                    block_test_[kLowerSolve][i]->solve(kLowerSolve, thisDiagonalBlock);
                }

#ifdef OMP_SUPPORT
            }
            #pragma omp barrier

            //Calculate Category 4 blocks (parallel)
            #pragma omp parallel for
#endif
            for(int threadid = 0; threadid < block_test_[kSchurComp].size(); threadid++) {
                Block<T>& this_block = *(block_test_[kSchurComp][threadid]);
                this_block.solve(kSchurComp, matrix_.getBlock(this_block.getCoordinate().row, current_solve_level_), matrix_.getBlock(current_solve_level_, this_block.getCoordinate().col));
            }

            //All blocks should be solved by this point

            current_solve_level_++;
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Execution time: " << duration.count() << std::endl;
    }
};

#endif