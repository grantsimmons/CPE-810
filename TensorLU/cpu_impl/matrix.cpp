#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#include <iomanip>

struct matrix {
    const int num_row, num_col;

    double* data;

    std::string name;

    matrix(std::string name, const int num_row, const int num_col) 
    : name(name), num_row(num_row), num_col(num_col) {
        this->data = new double[num_row * num_col];
        std::fill(this->data, this->data + this->num_row * this->num_col, 0);
        if(num_row != num_col) {
            std::cout << "ERROR: Matrix " << name << " is not square! (Dimensions: " << num_row << ", " << num_col << ")" << std::endl;
            std::cout << "    Only square matrices are supported at this time." << std::endl;
            exit(1);
        }
    }

    matrix(std::string name, const int num_row, const int num_col, double* data) 
    : name(name), num_row(num_row), num_col(num_col), data(data) {
        if(num_row != num_col) {
            std::cout << "ERROR: Matrix " << name << " is not square! (Dimensions: " << num_row << ", " << num_col << ")" << std::endl;
            std::cout << "    Only square matrices are supported at this time." << std::endl;
            exit(1);
        }
    }
    
    matrix(matrix* mat) : num_row(mat->num_row), num_col(mat->num_col) {
        //Perform deep copy of data
        //this->set_name(mat->name.append(" copy"));
        this->data = new double[num_row * num_col];
        std::copy(mat->data, mat->data + mat->num_row * mat->num_col, this->data);
    }

    ~matrix() {
        delete data;
    }

    void set_name(std::string name) {
        this->name = name;
    }

    double get(int r, int c) {
        if(c < num_col && r < num_row) {
            // if(name != "")
            //     std::cout << "(" << name << ") ";
            // std::cout << "Getting " << r << ", " << c << ": " << this->data[(r * this->num_col) + c] << std::endl;
            return this->data[(r * this->num_col) + c];
        }
        else {
            printf("matrix::get(%d,%d) index out of bounds (%d, %d)\n", r, c, num_row - 1, num_col - 1);
            exit(1);
        }
    }

    void set(int r, int c, double val) {
        if(c < num_col && r < num_row) {
            // if (name != "")
            //     std::cout << "(" << name << ") ";
            // std::cout << "Setting " << r << ", " << c << ": " << val << std::endl;
            this->data[(r * this->num_col) + c] = val;
        }
        else {
            printf("matrix::set(%d,%d) index out of bounds (%d, %d)", r, c, num_row - 1, num_col - 1);
            exit(1);
        }
    }

    void print() {
        if(name != "")
            std::cout << std::endl << name << ": " << std::endl;
        for(int y = 0; y < this->num_row; y++) {
            for(int x = 0; x < this->num_col; x++) {
                std::cout << std::setw(12) << this->data[(y * this->num_col) + x] << " ";
            }
            std::cout << std::endl;
        }
    }

    void print_matlab() {
        std::cout << std::endl << "A = [";
        for(int y = 0; y < this->num_row; y++) {
            for(int x = 0; x < this->num_col; x++) {
                std::cout << " " << this->data[(y * this->num_col) + x];
            }
            if(y == this->num_row - 1) {
                std::cout << "]";
            }
            std::cout << ";\n";
        }
    }

    void solve_doolittle() {
        //Doolittle notes:
        //U: Determine rows from top to bottom
        //L: Determine columns from left to right
        this->set_name("A");
        matrix* l = new matrix("L", this->num_row, this->num_col);
        matrix* u = new matrix("U", this->num_row, this->num_col);

        if(num_row == num_col)
            for(int i = 0; i < num_row; i++) l->set(i, i, 1);
            //For the Doolittle algorithm, the L matrix is the unit triangular matrix

        int u_row = 0, l_col = 0;
        while(u_row < num_row && l_col < num_col) {
            //std::cout << "Calculating row " << i << " of U" << std::endl;
            for(int j = u_row; j < num_row; j++) { //Calculate row of U
                double sum = 0;
                //std::cout << "Values: " << std::endl;
                for(int x = 0; x < u_row; x++) {
                    sum += u->get(x, j) * l->get(u_row, x);
                    //std::cout << i << " " << j << " " << x << std::endl;
                    //std::cout << "Sum: " << sum << std::endl;
                }
                double val = (this->get(u_row, j) - sum); 
                    // '/ l->get(u_row, u_row);' //By convention, this part should always be 1
                u->set(u_row, j, val);
            }

            //std::cout << "Calculating column " << i << " of L" << std::endl;
            for(int j = l_col + 1; j < num_col; j++) { //Calculate column of L
                double sum = 0;
                for(int x = 0; x < l_col; x++) {
                    sum += l->get(j, x) * u->get(x, l_col);
                    //std::cout << i << " " << j << " " << x << std::endl;
                    //std::cout << "Sum: " << sum << std::endl;
                }
                double val = (this->get(j, l_col) - sum) / u->get(l_col, l_col);
                l->set(j, l_col, val);
            }

            u_row++; //Keeping these separate in case we want to support rectangular matrices
            l_col++;
        }
        l->print();
        u->print();
    }
};

matrix* read_matrix(std::string filename) {
    std::ifstream file(filename);
    int num_row, num_col, num_lines;

    // Ignore comments headers
    while (file.peek() == '%') file.ignore(2048, '\n');

    // Read number of rows and columns
    file >> num_row >> num_col >> num_lines;

    // Create 2D array and fill with zeros
    matrix* mat = new matrix(filename, num_row, num_col);

    // fill the matrix with data
    for (int l = 0; l < num_lines; l++)
    {
         double data;
         int row, col;
         file >> row >> col >> data;
         mat->data[col + (row * num_col)] = data;
    }

    file.close();

    return mat;
}


int main () {
    matrix* mat = read_matrix("../data/matrix/Mat16_16_New.mtx");

    // double test_arr[] = {2, -1, -2, 
    //                     -4,  6,  3, 
    //                     -4, -2,  8};
    // matrix* test = new matrix("Test", 3, 3, test_arr);

    // test->solve_doolittle();

    mat->print();

    //mat->print_matlab();

    mat->solve_doolittle();
}