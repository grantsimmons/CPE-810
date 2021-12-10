#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <tuple>
#include <stdarg.h>

template <class T>
class matrix {
private:

    bool debug;

    T** data;
    
    std::string name;

    int category = 4;

public:
    int num_row, num_col;

    int dprint(const char* format, ... ) {
        int ret = 0;
        va_list args;
        va_start(args, format);
        if(this->debug) {
            ret = vprintf(format, args);
        }
        va_end(args);
        return ret;
    }

    matrix(std::string name, const int num_row, const int num_col, bool debug = false)
        : name(name), num_row(num_row), num_col(num_col), debug(debug) {

        this->data = new T*[num_row];
        for(int row = 0; row < num_row; row++) {
            this->data[row] = new T[num_col];
            std::fill(this->data[row], this->data[row] + this->num_col, 0);
        }

    }

    matrix(std::string name, const int num_row, const int num_col, T* mat_1d, bool debug = false)
        : name(name), num_row(num_row), num_col(num_col), debug(debug) {

        this->data = new T*[num_row];
        for(int row = 0; row < num_row; row++) {
            this->data[row] = new T[num_col];
            int base_index = row * num_col;
            std::copy(mat_1d + base_index, mat_1d + base_index + num_col, this->data[row]);
        }

    }

    matrix(matrix* mat) : name(mat->name), num_row(mat->num_row), num_col(mat->num_col), debug(mat->debug) {
        //Perform deep copy of data
        this->data = new T*[mat->num_row];
        for(int row = 0; row < this->num_row; row++) {
            this->data[row] = new T[this->num_col];
            std::copy(mat->data[row], mat->data[row] + mat->num_col, this->data[row]);
        }
    }

    matrix(matrix& mat) : name(mat.name), num_row(mat.num_row), num_col(mat.num_col), debug(mat.debug) {
        this->data = new T*[mat.num_row];
        for(int row = 0; row < this->num_row; row++) {
            this->data[row] = new T[this->num_col];
            std::copy(mat.data[row], mat.data[row] + mat.num_col, this->data[row]);
        }
    }

    matrix(std::string filename, bool debug = false) : debug(debug), num_row(0), num_col(0){
        std::ifstream file(filename);
        //TODO: Check for file existence!
        int num_lines;

        // Ignore comments headers
        while (file.peek() == '%') file.ignore(2048, '\n');

        // Read number of rows and columns
        file >> this->num_row >> this->num_col >> num_lines;

        this->name = filename;

        // Create 2D array and fill with zeros'
        this->data = new T*[this->num_row];
        for(int row = 0; row < this->num_row; row++) {
            std::cout << row << std::endl;
            this->data[row] = new T[this->num_col];
            std::fill(this->data[row], this->data[row] + this->num_col, 0);
        }

        // fill the matrix with data

        for (int l = 0; l < num_lines; l++)
        {
            T data_point;
            int row, col;
            file >> row >> col >> data_point;
            this->data[row][col] = data_point;
        }

        file.close();
    }

    ~matrix() {
        for(int row = 0; row < this->num_row; row++) {
            delete this->data[row];
        }
        delete data;
    }

    matrix& operator*(matrix& mat) {
        if (this->num_col != mat.num_row) {
            this->print();
            mat.print();
            throw std::runtime_error("Matrix " + this->name + " column dimension (" + std::to_string(this->num_col) + ") does not equal " + mat.name + " row dimension (" + std::to_string(mat.num_row) + ")");
        }
        matrix* ret = new matrix(this->name + "*" + mat.name, this->num_row, mat.num_col);

        for(int row = 0; row < this->num_row; row++) {
            for(int col = 0; col < mat.num_col; col++) {
                T sum = 0;
                for(int idx = 0; idx < mat.num_row; idx++) { //FIXME: right?
                    sum += this->data[row][idx] * mat[idx][col];
                }
                (*ret)[row][col] = sum;
            }
        }
        return (*ret);
    }

    matrix& operator/(T& div) {
        matrix<T>* ret = new matrix(this->name + "/" + std::to_string(div), this->num_row, this->num_col);
        for(int row = 0; row < this->num_row; row++) {
            for(int col = 0; col < this->num_col; col++) {
                ret->data[row][col] = this->data[row][col] / div;
            }
        }
        return (*ret);
    }

    matrix<T>& operator-(matrix& mat) {
        if (this->num_col != mat.num_col || this->num_row != mat.num_row)
            throw std::runtime_error("Matrix " + this->name + " and " + mat.name + "dimensions not equal"); 
        matrix<T>* ret = new matrix<T>(this->name + "-" + mat.name, this->num_row, this->num_col);
        for(int row = 0; row < this->num_row; row++) {
            for(int col = 0; col < this->num_col; col++) {
                ret->data[row][col] = this->data[row][col] - mat[row][col];
            }
        }     
        return (*ret); 
    }

    void operator-=(matrix& mat) {
        if (this->num_col != mat.num_col || this->num_row != mat.num_row)
            throw std::runtime_error("Matrix " + this->name + " and " + mat.name + "dimensions not equal"); 
        for(int row = 0; row < this->num_row; row++) {
            for(int col = 0; col < this->num_col; col++) {
                this->data[row][col] -= mat[row][col];
            }
        }     
    }

    matrix<T>& operator+(matrix& mat) {
        if (this->num_col != mat.num_col || this->num_row != mat.num_row)
            throw std::runtime_error("Matrix " + this->name + " and " + mat.name + "dimensions not equal"); 
        matrix<T>* ret = new matrix<T>(this->name + "-" + mat.name, this->num_row, this->num_col);
        for(int row = 0; row < this->num_row; row++) {
            for(int col = 0; col < this->num_col; col++) {
                ret->data[row][col] = this->data[row][col] + mat[row][col];
            }
        }     
        return (*ret); 
    }

    void operator+=(matrix& mat) {
        if (this->num_col != mat.num_col || this->num_row != mat.num_row)
            throw std::runtime_error("Matrix " + this->name + " and " + mat.name + "dimensions not equal"); 
        for(int row = 0; row < this->num_row; row++) {
            for(int col = 0; col < this->num_col; col++) {
                this->data[row][col] += mat[row][col];
            }
        }     
    }

    void operator=(matrix& mat) {
        this->data = new T*[mat.num_row];
        for(int row = 0; row < this->num_row; row++) {
            this->data[row] = new T[this->num_col];
            std::copy(mat.data[row], mat.data[row] + mat.num_col, this->data[row]);
        }
    }

    void operator=(matrix* mat) {
        this->data = new T*[mat->num_row];
        for(int row = 0; row < this->num_row; row++) {
            this->data[row] = new T[this->num_col];
            std::copy(mat->data[row], mat->data[row] + mat->num_col, this->data[row]);
        }
    }

    T* operator[](int idx) {
        if(idx >= this->num_row)
            throw std::runtime_error("Matrix " + this->name + " Row index " + std::to_string(idx) + " out of bounds (num_row = " + std::to_string(this->num_row) + ")");
        return data[idx];
    }

    void set_name(std::string name) {
        this->name = name;
    }

    int get_category() {
        return category;
    }

    void set_category(int cat) {
        this->category = cat;
    }

    T get(int r, int c) {
        if(c < num_col && r < num_row) {
            dprint("Getting %d, %d: %f\n", r, c, this->data[r][c]); 
            return this->data[r][c];
        }
        else
            throw std::runtime_error("matrix::get(" + std::to_string(r) + ", " + std::to_string(c) + ") index out of bounds (" + std::to_string(num_row - 1) + ", " + std::to_string(num_col - 1) + ")");
    }

    void set(int r, int c, T val) {
        if(c < num_col && r < num_row) {
            dprint("Setting %d, %d: %f\n", r, c, val);
            this->data[r][c] = val;
        }
        else
            throw std::runtime_error("matrix::set(" + std::to_string(r) + ", " + std::to_string(c) + ") index out of bounds (" + std::to_string(num_row - 1) + ", " + std::to_string(num_col - 1) + ")");
    }

    void swap_rows(int src, int dest) {
        T* temp;
        temp = this->data[dest];
        this->data[dest] = this->data[src];
        this->data[src] = temp;
    }

    void print() {
        if(name != "")
            std::cout << std::endl << name << ": " << std::endl;
        for(int y = 0; y < this->num_row; y++) {
            for(int x = 0; x < this->num_col; x++) {
                std::cout << std::setw(12) << this->data[y][x] << " ";
            }
             std::cout << std::endl;
         }
    }

    void print_matlab() {
        std::cout << std::endl << "A = [";
        for(int y = 0; y < this->num_row; y++) {
            for(int x = 0; x < this->num_col; x++) {
                std::cout << " " << this->data[y][x];
            }
            if(y == this->num_row - 1) {
                std::cout << "]";
            }
            std::cout << ";\n";
        }
    }   

    std::tuple<matrix<T>&, matrix<T>&> solve_doolittle() {
        //Doolittle notes:
        //U: Determine rows from top to bottom
        //L: Determine columns from left to right
        #define A (*this)
        if(A.num_col != A.num_row)
            throw std::runtime_error("Matrix " + A.name + " is not square!");
            
        matrix* l = new matrix("L", this->num_row, this->num_col);
        #define L (*l)
        matrix* u = new matrix("U", this->num_row, this->num_col);
        #define U (*u)

        if(num_row == num_col)
            for(int i = 0; i < num_row; i++) L[i][i] = 1;
            //For the Doolittle algorithm, the L matrix is the unit triangular matrix

        int u_row = 0, l_col = 0;
        while(u_row < num_row && l_col < num_col) {
            dprint("Calculating row %d of U", u_row);
            for(int j = u_row; j < num_row; j++) { //Calculate row of U
                T sum = 0;
                dprint("Values:");
                for(int x = 0; x < u_row; x++) {
                    sum += U[x][j] * L[u_row][x];
                    dprint("%d: %d %d\nSum: %f", u_row, j, x, sum);
                }
                T val = A[u_row][j] - sum; 
                    // '/ L[u_row][u_row];' //By convention, this part should always be 1
                U[u_row][j] = val;
            }

            //std::cout << "Calculating column " << i << " of L" << std::endl;
            for(int j = l_col + 1; j < num_col; j++) { //Calculate column of L
                T sum = 0;
                for(int x = 0; x < l_col; x++) {
                    sum += L[j][x] * U[x][l_col];
                    dprint("%d: %d %d\nSum: %f", l_col, j, x, sum);
                }
                T val = (A[j][l_col] - sum) / U[l_col][l_col];
                L[j][l_col] = val;
            }

            u_row++; //Keeping these separate in case we want to support rectangular matrices
            l_col++;
        }

        return std::tuple<matrix<T>&,matrix<T>&>(U,L);
    }

    std::tuple<matrix<T>&, matrix<T>&>  solve_left_looking() {
        #define A (*this)
        if(A.num_col != A.num_row)
            throw std::runtime_error("Matrix " + A.name + " is not square!");
            
        matrix* l = new matrix("LL-L", this->num_row, this->num_col);
        #define L (*l)
        for(int i = 0; i < this->num_row; i++) {
            L[i][i] = 1;
        }
        matrix* u = new matrix("LL-U", this->num_row, this->num_col);
        #define U (*u)

        for(int i = 0; i < this->num_row; i++) {
            matrix<T> partial("placeholder",1,1);
            matrix<T> partial_prod("placeholder_prod",this->num_row-i-1,1);
            if(i > 0) {
                //We need to avoid the boundary case where i-1 becomes negative
                //Having num_col or num_row < 1 will create issues with the matrix operators
                //and it's easier to handle this issue here instead

                matrix<T> L11 = l->submatrix(0,0,i-1,i-1);                 //Matrix
                matrix<T> a12 = this->submatrix(0,i,i-1,i);                //Vector
                matrix<T> u12 = solve_triangle(L11, a12); //u12 = a12 / L11 (Vector)
            
                U.set_range(u12, 0, i);

                matrix<T> l21 = l->submatrix(i,0,i,i-1);                   //Vector
                partial = l21 * u12; //Must be initialized to 0            //Vector

                matrix<T> L31 = l->submatrix(i+1,0,l->num_row - 1,i-1);    //Matrix
                partial_prod = L31 * u12; // Must be initialized to 0      //Vector


            }

            if (partial.num_row != 1 && partial.num_col != 1)
                throw std::runtime_error("(" + partial.name + ") Partial dimensions are not (1,1). num_row: " + std::to_string(partial.num_row) + ", num_col: " + std::to_string(partial.num_col));

            matrix<T> partial_net = *(this->submatrix(i+1,i,this->num_row - 1,i)) - partial_prod;
            matrix<T> l32 = partial_net / U[i][i];

            U[i][i] = this->data[i][i] - partial[0][0]; //u22 = a22 / (l21 * u12)
            
            L.set_range(l32,i+1,i);
        }
        U.print();
        L.print();
        
        return std::tuple<matrix<T>&,matrix<T>&>(U,L);
    }

    std::tuple<matrix<T>&, matrix<T>&>  solve_right_looking() {
        #define A (*this)
        if(A.num_col != A.num_row)
            throw std::runtime_error("Matrix " + A.name + " is not square!");
            
        matrix* l = new matrix("RL-L", this->num_row, this->num_col);
        #define L (*l)
        for(int i = 0; i < this->num_row; i++) {
            L[i][i] = 1;
        }
        matrix* u = new matrix("RL-U", this->num_row, this->num_col);
        #define U (*u)

        T u11 = A[0][0];
        for(int j = 0 ; j < this->num_col; j++) {
            U[0][j] = A[0][j];
            L[j][0] = A[j][0] / u11;
        }

        //Effectively:
        //U.set_range(A.submatrix(0,0,0,A.num_col-1), 0,0);
        //L.set_range(A.submatrix(0,0,A.num_row-1,0) / u11, 0,0);
        //This specific implementation does not work due to unsupported
        //function signatures and operators

        matrix<T> l21 = L.submatrix(1,0,L.num_row-1,0);
        
        matrix<T> u12 = U.submatrix(0,1,0,U.num_col-1);
        
        matrix<T> product = l21 * u12;

        if(product.num_col >= 1 || product.num_row >= 1) {
            matrix<T> A22 = A.submatrix(1, 1, num_row-1, num_col-1);
            matrix<T> LU22(A22 - product);
            auto[u_temp, l_temp] = LU22.solve_right_looking();
            U.set_range(u_temp, 1, 1);
            L.set_range(l_temp, 1, 1);
        }

        return std::tuple<matrix<T>&,matrix<T>&>(U,L);
    }

    std::tuple<matrix<T>&, matrix<T>&>  solve_blocking(int block_size) {
        #define A (*this)
        if(A.num_col != A.num_row)
            throw std::runtime_error("Matrix " + A.name + " is not square!");
            
        matrix* l = new matrix("RL-L", this->num_row, this->num_col);
        #define L (*l)
        for(int i = 0; i < this->num_row; i++) {
            L[i][i] = 1;
        }
        matrix* u = new matrix("RL-U", this->num_row, this->num_col);
        #define U (*u)

        T u11 = A[0][0];
        for(int j = 0 ; j < this->num_col; j++) {
            U[0][j] = A[0][j];
            L[j][0] = A[j][0] / u11;
        }

        //Effectively:
        //U.set_range(A.submatrix(0,0,0,A.num_col-1), 0,0);
        //L.set_range(A.submatrix(0,0,A.num_row-1,0) / u11, 0,0);
        //This implementation does not work due to unsupporting functionsignatures and operators

        matrix<T> L_sub = L.submatrix(1,0,L.num_row-1,0);
        
        matrix<T> U_sub = U.submatrix(0,1,0,U.num_col-1);
        
        matrix<T> product = L_sub * U_sub;

        if(product.num_col >= 1 || product.num_row >= 1) {
            matrix<T> A_sub = A.submatrix(1, 1, num_row-1, num_col-1);
            matrix<T> LU22(A_sub - product);
            auto[u_temp, l_temp] = LU22.solve_right_looking();
            U.set_range(u_temp, 1, 1);
            L.set_range(l_temp, 1, 1);
        }

        return std::tuple<matrix<T>&,matrix<T>&>(U,L);
    }

    void set_range(matrix<T>& insert, int r, int c) {
        for(int ir = 0; ir < insert.num_row; ir++) {
            for(int ic = 0; ic < insert.num_col; ic++) {
                A[r+ir][c+ic] = insert[ir][ic];
            }
        }
    }

    matrix<T>* submatrix(int r_min, int c_min, int r_max,int c_max) {
        matrix<T>* res = new matrix<T>(this->name + " sub (" + std::to_string(r_min) + "," + std::to_string(c_min) + "," + std::to_string(r_max) + "," + std::to_string(c_max) + ")", 
                      (r_max-r_min) + 1, (c_max-c_min) + 1);
        for(int row = r_min; row <= r_max; row++) {
            for(int col = c_min; col <= c_max; col++) {
                res->data[row - r_min][col - c_min] = A[row][col];
            }
        }
        return res;
    }
};

template <typename T>
matrix<T> solve_triangle(matrix<T>& mat, matrix<T>& vec) {
    T sum;
    //TODO: add dimension checks here
    matrix<T> res("result", vec.num_row, vec.num_col);
    for(int i = 0; i < mat.num_row; i++) {
        sum = 0;
        for(int j = 0; j < i; j++) {
            sum += mat[i][j] * res[j][0];
        }
        res[i][0] = (vec[i][0] - sum) / mat[i][i];
        //res.print();
    }
    return res;
}

int main () {
    matrix<double> file_test("../../data/matrix/Mat8_8.mtx");
    double test_mat[] = { 3, 0, 0, 0, -1, 1, 0, 0, 3, -2, -1, 0, 1, -2, 6, 2};
    double test_arr[] = {5, 6, 4, 2};
    matrix<double> imm_test("imm_test", 4, 4, test_mat);
    matrix<double> imm_arr("imm_arr", 4, 1, test_arr);
    solve_triangle<double>(imm_test, imm_arr);
    file_test.print_matlab();
    auto[t1,t2] = file_test.solve_right_looking();
    t1.print();
    t2.print();
}