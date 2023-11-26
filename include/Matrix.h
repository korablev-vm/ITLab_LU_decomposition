#ifndef LU_DECOMPOSITION_MATRIX_H
#define LU_DECOMPOSITION_MATRIX_H

#include <iostream>
#include <stdexcept>

using namespace std;

class Matrix {
private:
    size_t size;
    double *data; // Используем указатель на массив
    static const size_t blockSize = 32;

    inline size_t index(size_t row, size_t col) const {
        return row * size + col;
    }

public:
    double& operator()(int row, int col);
    const double& operator()(int row, int col) const;
    size_t get_size() const;
    Matrix(int sz);
    Matrix(const Matrix& other); // Конструктор копирования
    Matrix& operator=(const Matrix& other); // Оператор присваивания
    ~Matrix(); // Деструктор
    friend ostream& operator<<(ostream& os, const Matrix& matrix);
    void input();
    Matrix multiply(const Matrix& other);
    void LU_Decomposition(Matrix& L, Matrix& U);
};

#endif //LU_DECOMPOSITION_MATRIX_H
