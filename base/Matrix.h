#ifndef LU_DECOMPOSITION_MATRIX_H
#define LU_DECOMPOSITION_MATRIX_H

#include <iostream>
#include <stdexcept>
#include <omp.h>
#include <cstring>
#include <algorithm>
#include <immintrin.h>


class Matrix {
private:
    size_t size;
    double* data;
    const size_t blockSize = 8;

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
    double* get_data() const;

    void input();
    Matrix multiply(const Matrix& other);
    void LU_Decomposition(Matrix& L, Matrix& U);

    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix);

};

#endif //LU_DECOMPOSITION_MATRIX_H


Matrix::Matrix(int sz) : size(sz) {
    data = new double[sz * sz];
    memset(data, 0, sz * sz * sizeof(double));
}

Matrix::Matrix(const Matrix& other) : size(other.size) {
    data = new double[size * size];
    memcpy(data, other.data, size * size * sizeof(double));
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        delete[] data;
        size = other.size;
        data = new double[size * size];
        memcpy(data, other.data, size * size * sizeof(double));
    }
    return *this;
}

Matrix::~Matrix() {
    delete[] data;
}

double& Matrix::operator()(int row, int col) {
    return data[index(row, col)];
}

const double& Matrix::operator()(int row, int col) const {
    return data[index(row, col)];
}

size_t Matrix::get_size() const {
    return size;
}

double* Matrix::get_data() const {
    return data;
}

void Matrix::input() {
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            std::cin >> data[index(i, j)];
        }
    }
}

Matrix Matrix::multiply(const Matrix& other) {
    if (size != other.size)
        throw std::invalid_argument("Matrix sizes do not match for multiplication.");
    Matrix result(size);
    for (size_t i = 0; i < size; ++i)
        for (size_t j = 0; j < size; ++j)
            for (size_t k = 0; k < size; ++k)
                result(i, j) += (*this)(i, k) * other(k, j);
    return result;
}

//void Matrix::LU_Decomposition(Matrix& L, Matrix& U) {
//    for (size_t i = 0; i < size; ++i) {
//        for (size_t j = 0; j < size; ++j) {
//            L(i, j) = (i == j) ? 1.0 : 0.0;
//            U(i, j) = 0.0;
//        }
//    }
//    for (size_t i = 0; i < size; i += blockSize) {
//        size_t limit = i + blockSize < size ? i + blockSize : size;
//
//        for (size_t j = i; j < limit; ++j) {
//            for (size_t k = i; k < j; ++k) {
//                __m256d sum_vec = _mm256_setzero_pd(); // Инициализация вектора суммы нулями
//                size_t l;
//                for (l = i; l + 4 <= k; l += 4) {
//                    __m256d L_vec = _mm256_loadu_pd(&L(j, l));
//                    __m256d U_vec = _mm256_loadu_pd(&U(l, k));
//                    __m256d mult_vec = _mm256_mul_pd(L_vec, U_vec);
//                    sum_vec = _mm256_add_pd(sum_vec, mult_vec);
//                }
//                double sum_array[4];
//                _mm256_storeu_pd(sum_array, sum_vec);
//                double sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
//                for (; l < k; ++l) { // Обработка оставшихся элементов, если k - i не кратно 4
//                    sum += L(j, l) * U(l, k);
//                }
//                L(j, k) = ((*this)(j, k) - sum) / U(k, k);
//            }
//            for (size_t k = j; k < limit; ++k) {
//                __m256d sum_vec = _mm256_setzero_pd();
//                size_t l;
//                for (l = i; l + 4 <= j; l += 4) {
//                    __m256d L_vec = _mm256_loadu_pd(&L(j, l));
//                    __m256d U_vec = _mm256_loadu_pd(&U(l, k));
//                    __m256d mult_vec = _mm256_mul_pd(L_vec, U_vec);
//                    sum_vec = _mm256_add_pd(sum_vec, mult_vec);
//                }
//                double sum_array[4];
//                _mm256_storeu_pd(sum_array, sum_vec);
//                double sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
//                for (; l < j; ++l) { // Обработка оставшихся элементов
//                    sum += L(j, l) * U(l, k);
//                }
//                U(j, k) = (*this)(j, k) - sum;
//            }
//        }
//    }
//}




void Matrix::LU_Decomposition(Matrix& L, Matrix& U) {
    // сделаем предварительную инициализацию L и U. L - единичная матрица, U - нулевая матрица
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            L(i, j) = (i == j) ? 1.0 : 0.0;
            U(i, j) = 0.0;
        }
    }

    for (size_t i = 0; i < size; i += blockSize) {
        // лимит - граница блока для предотвращения выхода за пределы матрицы
        size_t limit = i + blockSize < size ? i + blockSize : size;

        for (size_t j = i; j < limit; ++j) {
            for (size_t k = i; k < j; ++k) {
                double sum = 0.0;
                for (size_t l = i; l < k; ++l) {
                    sum += L(j, l) * U(l, k); // вычисление элемента L.
                }
                L(j, k) = ((*this)(j, k) - sum) / U(k, k); // обновление L с учетом U
            }
            // вычисление элементов верхней треугольной матрицы U
            for (size_t k = j; k < limit; ++k) {
                double sum = 0.0;
                for (size_t l = i; l < j; ++l) {
                    sum += L(j, l) * U(l, k); // вычисление элемента U
                }
                U(j, k) = (*this)(j, k) - sum; // обновление U
            }
        }

        // Параллельное обновляем элементы L и U за пределами блока.
#pragma omp parallel for collapse(2) schedule(static)
        for (size_t j = i + blockSize; j < size; ++j) {
            for (size_t k = i; k < limit; ++k) {
                if (k < j) { // Только для элементов ниже главной диагонали (для L).
                    double sum = 0.0;
                    for (size_t l = i; l < k; ++l) {
                        sum += L(j, l) * U(l, k); // суммирование для вычисления элемента L.
                    }
                    L(j, k) = ((*this)(j, k) - sum) / U(k, k); // обновление значения L
                }
                // обновление U не требуется, так как заполняется в предыдущих шагах
            }
        }
    }
}

std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    for (int i = 0; i < matrix.size; ++i) {
        for (int j = 0; j < matrix.size; ++j) {
            os << matrix.data[matrix.index(i, j)] << " ";
        }
        os << std::endl;
    }
    return os;
}


//void Matrix::LU_Decomposition(Matrix& L, Matrix& U) {
//
//    for (int i = 0; i < size; ++i) {
//        for (int j = 0; j < size; ++j) {
//            if (j < i)
//                L(j, i) = 0;
//            else {
//                L(j, i) = (*this)(j, i);
//                for (int k = 0; k < i; k++) {
//                    L(j, i) = L(j, i) - L(j, k) * U(k, i);
//                }
//            }
//        }
//        for (int j = 0; j < size; ++j) {
//            if (j < i)
//                U(i, j) = 0;
//            else if (j == i)
//                U(i, j) = 1;
//            else {
//                U(i, j) = (*this)(i, j) / L(i, i);
//                for (int k = 0; k < i; k++) {
//                    U(i, j) = U(i, j) - ((L(i, k) * U(k, j)) / L(i, i));
//                }
//            }
//        }
//    }
//}
