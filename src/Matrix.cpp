#include "Matrix.h"
#include <cstring> // Для использования memset и memcpy

// Конструктор
Matrix::Matrix(int sz) : size(sz) {
    data = new double[sz * sz];
    memset(data, 0, sz * sz * sizeof(double));
}

// Конструктор копирования
Matrix::Matrix(const Matrix& other) : size(other.size) {
    data = new double[size * size];
    memcpy(data, other.data, size * size * sizeof(double));
}

// Оператор присваивания
Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        delete[] data;
        size = other.size;
        data = new double[size * size];
        memcpy(data, other.data, size * size * sizeof(double));
    }
    return *this;
}

// Деструктор
Matrix::~Matrix() {
    delete[] data;
}

// Оператор доступа (для изменения)
double& Matrix::operator()(int row, int col) {
    return data[index(row, col)];
}

// Оператор доступа (для чтения)
const double& Matrix::operator()(int row, int col) const {
    return data[index(row, col)];
}

// Получение размера матрицы
size_t Matrix::get_size() const {
    return size;
}

// Ввод данных в матрицу
void Matrix::input() {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            cin >> data[index(i, j)];
        }
    }
}

// Умножение матриц
Matrix Matrix::multiply(const Matrix& other) {
    if (size != other.size)
        throw invalid_argument("Matrix sizes do not match for multiplication.");

    Matrix result(size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                result(i, j) += (*this)(i, k) * other(k, j);
            }
        }
    }
    return result;
}

// LU-разложение (простая реализация)
void Matrix::LU_Decomposition(Matrix& L, Matrix& U) {
    if (L.size != size || U.size != size)
        throw invalid_argument("Matrix sizes do not match for LU decomposition.");

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (j < i)
                L(j, i) = 0;
            else {
                L(j, i) = (*this)(j, i);
                for (int k = 0; k < i; k++) {
                    L(j, i) = L(j, i) - L(j, k) * U(k, i);
                }
            }
        }
        for (int j = 0; j < size; ++j) {
            if (j < i)
                U(i, j) = 0;
            else if (j == i)
                U(i, j) = 1;
            else {
                U(i, j) = (*this)(i, j) / L(i, i);
                for (int k = 0; k < i; k++) {
                    U(i, j) = U(i, j) - ((L(i, k) * U(k, j)) / L(i, i));
                }
            }
        }
    }
}

// Оператор вывода
ostream& operator<<(ostream& os, const Matrix& matrix) {
    for (int i = 0; i < matrix.size; ++i) {
        for (int j = 0; j < matrix.size; ++j) {
            os << matrix(i, j) << " ";
        }
        os << endl;
    }
    return os;
}
