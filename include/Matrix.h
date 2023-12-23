#ifndef LU_DECOMPOSITION_MATRIX_H
#define LU_DECOMPOSITION_MATRIX_H

#include <iostream>
#include <stdexcept>
#include <omp.h>

class Matrix {
private:
    size_t size;
    double *data;
    const size_t blockSize = 32;

    inline size_t index(int row, int col) const {
        return row * size + col;
    }

public:
    Matrix(int sz);
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    ~Matrix();

    double& operator()(int row, int col);
    const double& operator()(int row, int col) const;
    size_t get_size() const;

    void input();
    Matrix multiply(const Matrix& other);
    void LU_Decomposition(Matrix& L, Matrix& U);

    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix);

};

#endif //LU_DECOMPOSITION_MATRIX_H
