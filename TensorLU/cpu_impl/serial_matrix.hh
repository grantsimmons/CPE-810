#ifndef SERIAL_MATRIX_H_
#define SERIAL_MATRIX_H_


#include <iostream>
#include <stdexcept>
#include <vector>
#include <iomanip>

template <typename T>
class SerialMatrix; //Forward declare due to cyclical dependency

enum category {kRightLooking, kUpperSolve, kLowerSolve, kSchurComp, kProcessed};

template <typename T>
class Block {
private:
    typedef typename std::vector<T>::iterator BlockStart;
    const BlockStart block_start__;
    SerialMatrix<T>* parent__;
    const int block_size_squared__;
    const int block_size__;
    int category__;
    struct BlockCoord {int row, col;} block_coord__;

public:

    Block(BlockStart block_iter, int block_row, int block_col, SerialMatrix<T>* parent, int category = kSchurComp) : 
        block_start__(block_iter), parent__(parent), category__(category), block_coord__({block_row,block_col}),
        block_size__(parent->getBlockSize()), block_size_squared__(parent->getBlockSizeSquared()) {}
    
    Block() { //Default constructor required
    }

    SerialMatrix<T>& getParent() {
        return *(parent__);
    }

    auto getCoordinate() {
        return block_coord__;
    }

    int getBlockSize() {
        return block_size__;
    }

    int getBlockSizeSquared() {
        return block_size_squared__;
    }

    void setCategory(category cat) {
        category__ = cat;
    }

    void printBlock() {
        for(int i = 0; i < parent__->getBlockSizeSquared(); i++) {
            std::cout << *(block_start__ + i) << std::endl;
        }
    }

    T& getElement(int row, int col) {
        return *(block_start__ + row * block_size__ + col);
    }

    void solve(category cat) {
        //std::cout << "Conducting ";

        if(cat == kRightLooking) {
            //solve
            //std::cout << "Right looking ";
            //std::vector<T> new_vec(block_start__, block_start__ + block_size_squared__);
            for(int k = 0; k < block_size__; k++) {
                //U[i,i:] = A[i,i:]
                for(int i = k + 1; i < block_size__; i++) {
                    //L[i][k] = A[i][k]/U[k][k]
                    block_start__[i * block_size__ + k] = block_start__[i * block_size__ + k] / block_start__[k * block_size__ + k];
                }
                for(int j = k + 1; j < block_size__; j++) {
                    for(int i = k + 1; i < block_size__; i++) {
                        block_start__[i * block_size__ + j] -= block_start__[i * block_size__ + k] * block_start__[k * block_size__ + j];
                    }
                }
            }
        }
        else {
            throw std::runtime_error("Error");
        }
        
        //std::cout << "solve for block " << block_coord__.row << ", " << block_coord__.col << std::endl;
    }

    void solve(category cat, Block<T>& dep) {
        //std::cout << "Conducting ";
        switch (cat) {
            case kUpperSolve:
                // std::cout << "Upper ";

                for(int k = 0; k < block_size__; k++) {
                    for(int j = 0; j < block_size__; j++) {
                        for(int i = k + 1; i < block_size__; i++) {
                            block_start__[i * block_size__ + j] -= dep.getElement(i,k) * block_start__[k * block_size__ + j];
                        }
                    }
                }

                break;
            case kLowerSolve:
                // std::cout << "Lower ";

                for(int k = 0; k < block_size__; k++) {
                    for(int i = 0; i < block_size__; i++) {
                        block_start__[i * block_size__ + k] /= dep.getElement(k,k);
                    }
                    for(int j = k + 1; j < block_size__; j++) {
                        for(int i = 0; i < block_size__; i++) {
                            block_start__[i * block_size__ + j] -= block_start__[i * block_size__ + k] * dep.getElement(k,j);
                        }
                    }
                }

                break;
            default:
                throw std::runtime_error("Error");
                break;
        }
        // std::cout << "solve for block " << block_coord__.row << ", " << block_coord__.col << std::endl;
    }

    void solve(category cat, Block<T>& dep1, Block<T>& dep2) {
        // std::cout << "Conducting ";
        if(cat == kSchurComp) {
            //solve
            // std::cout << "Schur complement ";
            for(int k = 0; k < block_size__; k++) {
                for(int j = 0; j < block_size__; j++) {
                    for(int i = 0; i < block_size__; i++) {
                        block_start__[i * block_size__ + j] -= dep1.getElement(i,k) * dep2.getElement(k,j);
                    }
                }
            }

        }
        else {
            throw std::runtime_error("Error");
        }
        // std::cout << "solve for block " << block_coord__.row << ", " << block_coord__.col << std::endl;
    }
};

template <typename T>
class SerialMatrix {
private:

    //const int block_size_, block_cols_, block_rows_, elem_cols_, elem_rows_;
    
    std::vector<T> matrix_data_;

    std::vector<Block<T>*> block_accessors_;

    auto getBlockPointer(int block_row, int block_col) {
        //return std::next(matrix_data_.begin(), (block_row * block_cols_ + block_col) * (block_size_ << 1));
        return std::next(matrix_data_.begin(), (block_size_ * block_size_) * (block_row * block_cols_ + block_col)) ;
    }

public:

    const int block_size_, block_cols_, block_rows_, elem_cols_, elem_rows_;

    SerialMatrix(T* data, int rows, int cols, int block_size = 1) 
        : block_size_(block_size), block_cols_((cols-1) / block_size_ + 1), 
          block_rows_((rows-1) / block_size_ + 1), elem_cols_(cols), elem_rows_(rows) {
        //TODO: Round up counts
        //Check size
        if(rows != cols) {
            throw std::runtime_error("Only square matrices are supported");
        }
        if(elem_rows_ % block_size_ != 0 || elem_cols_ % block_size_ != 0) {
            throw std::runtime_error("Row (" + std::to_string(elem_rows_) + ") or column (" + std::to_string(elem_cols_) + ") dimension is not even multiple of block size (" + std::to_string(block_size_) + ")");
        }

        //Initialize matrix
        matrix_data_ = std::vector<T>(rows * cols);

        //Generate block serialization
        for(int row = 0; row < rows; row++) {
            for(int col = 0; col < cols; col++) {
                //std::cout << data[row * cols + col];
                //std::cout << "Setting " << data[row * cols + col] << " at ";
                (*this).at(row,col) = data[row * cols + col];
            }
        }

        //Initialize block accessors
        block_accessors_ = std::vector<Block<T>*>(block_rows_ * block_cols_);

        //Generate block accessors
        for(int block_row = 0; block_row < block_rows_; block_row++) {
            for(int block_col = 0; block_col < block_cols_; block_col++) {
                block_accessors_[block_row * block_cols_ + block_col] = new Block<T>(getBlockPointer(block_row, block_col), block_row, block_col, this);
                //std::cout << "Block index (" << block_row << ", " << block_col << ") value: " << block_accessors_[block_row * block_cols_ + block_col]->getElement(0,0) << ". Block Size Squared: " << block_accessors_[block_row * block_cols_ + block_col]->getBlockSizeSquared() << std::endl;
            }
        }
    }
    
    ~SerialMatrix() {
        // for(int block_row = 0; block_row < block_rows_; block_row++) {
        //     for(int block_col = 0; block_col < block_cols_; block_col++) {
        //         delete [] block_accessors_[block_row * block_cols_ + block_col];
        //         std::cout << "Block index (" << block_row << ", " << block_col << ") value: " << block_accessors_[block_row * block_cols_ + block_col]->getElement(0,0) << ". Block Size Squared: " << block_accessors_[block_row * block_cols_ + block_col]->getBlockSizeSquared() << std::endl;
        //     }
        // }
    }

    T& operator()(int row, int col) {
        return (*this).at(row, col);
    }

    T& at(int row, int col) {
        int block_row_coarse = row / block_size_;
        int block_row_fine = row % block_size_;
        
        int block_column_coarse = col / block_size_;
        int block_column_fine = col % block_size_;

        int matrix_index = ( block_size_ * block_size_ ) * ( block_cols_ * block_row_coarse + block_column_coarse ) //Coarse-grain block index
                           + block_size_ * block_row_fine + block_column_fine;                             //Fine-grain block index
        
        //std::cout << "Matrix index: " << matrix_index << ", BRC: " << block_row_coarse << ", BRF: " << block_row_fine << ", BCC: " << block_column_coarse << ", BCF: " << block_column_fine << std::endl;

        return matrix_data_[matrix_index];
    }

    int getBlockSize() {
        return block_size_;
    }

    int getBlockRows() {
        return block_rows_;
    }

    int getBlockColumns() {
        return block_cols_;
    }
    int getBlockSizeSquared() {
        return (block_size_ * block_size_);
    }

    Block<T>& getBlock(int row, int col) {
        if(row < block_rows_ && col < block_cols_)
            return *(block_accessors_[row * block_cols_ + col]);
        else
            throw std::runtime_error("Block index (" + std::to_string(row) + ", " + std::to_string(col) + ") out of bounds (" + std::to_string(block_rows_-1) + ", " + std::to_string(block_cols_-1) + ")");
    }

    std::vector<Block<T>>& getBlocks() {
        return block_accessors_;
    }

    void print() {
        for(int row = 0; row < elem_rows_; row++) {
            std::cout << std::endl;
            for(int col = 0; col < elem_cols_; col++) {
                std::cout << std::setw(10) << (*this).at(row, col);
            }
        }
        std::cout << std::endl;
        for(int row = 0; row < elem_rows_; row++) {
            std::cout << std::endl;
            for(int col = 0; col < elem_cols_; col++) {
               std::cout << std::setw(10) << matrix_data_[row * elem_cols_ + col];
            }
        }
        std::cout << std::endl;
    }

    void printBlock(auto it) {
        for(int i = 0; i < (block_size_ * block_size_); i++) {
            std::cout << *(it+i) << std::endl;
        }
    }
};

#endif
