#include "Matrix.h"
#include <cstring>
#include <algorithm>

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

void Matrix::input() {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cin >> data[index(i, j)];
        }
    }
}

Matrix Matrix::multiply(const Matrix& other) {
    if (size != other.size)
        throw std::invalid_argument("Matrix sizes do not match for multiplication.");
    Matrix result(size);
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            for (int k = 0; k < size; ++k)
                result(i, j) += (*this)(i, k) * other(k, j);
    return result;
}

void Matrix::LU_Decomposition(Matrix& L, Matrix& U) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            L(i, j) = 0;
            U(i, j) = 0;
        }
        L(i, i) = 1;
    }
    for (int i = 0; i < size; i += blockSize) {
        for (int j = i; j < (i + blockSize < size ? i + blockSize : size); ++j) {
            for (int k = i; k < j; ++k) {
                L(j, k) = (*this)(j, k);
                for (int l = i; l < k; ++l) {
                    L(j, k) -= L(j, l) * U(l, k);
                }
                L(j, k) /= U(k, k);
            }
            for (int k = j; k < (i + blockSize < size ? i + blockSize : size); ++k) {
                U(j, k) = (*this)(j, k);
                for (int l = i; l < j; ++l) {
                    U(j, k) -= L(j, l) * U(l, k);
                }
            }
        }
#pragma omp parallel for
        for (int j = i + blockSize; j < size; ++j) {
            for (int k = i; k < (i + blockSize < size ? i + blockSize : size); ++k) {
                L(j, k) = (*this)(j, k);
                for (int l = i; l < k; ++l) {
                    L(j, k) -= L(j, l) * U(l, k);
                }
                L(j, k) /= U(k, k);
            }
#pragma omp parallel for
            for (int k = i; k < (i + blockSize < size ? i + blockSize : size); ++k) {
                for (int l = k + 1; l < size; ++l) {
                    (*this)(j, l) -= L(j, k) * U(k, l);
                }
            }
        }
    }
}

//void Matrix::LU_Decomposition(Matrix& L, Matrix& U) {
//    for (int i = 0; i < size; ++i) {
//        for (int j = 0; j < size; ++j) {
//            L(i, j) = (i == j) ? 1 : 0;
//            U(i, j) = 0;
//        }
//    }
//    for (int i = 0; i < size; i += blockSize) {
//        for (int j = i; j < (i + blockSize < size ? i + blockSize : size); ++j) {
//            for (int k = i; k < j; ++k) {
//                L(j, k) = (*this)(j, k);
//                for (int l = i; l < k; ++l) {
//                    L(j, k) -= L(j, l) * U(l, k);
//                }
//                L(j, k) /= U(k, k);
//            }
//            for (int k = j; k < (i + blockSize < size ? i + blockSize : size); ++k) {
//                U(j, k) = (*this)(j, k);
//                for (int l = i; l < j; ++l) {
//                    U(j, k) -= L(j, l) * U(l, k);
//                }
//            }
//        }
//        for (int j = i + blockSize; j < size; ++j) {
//            for (int k = i; k < (i + blockSize < size ? i + blockSize : size); ++k) {
//                L(j, k) = (*this)(j, k);
//                for (int l = i; l < k; ++l) {
//                    L(j, k) -= L(j, l) * U(l, k);
//                }
//                L(j, k) /= U(k, k);
//            }
//
//            for (int k = i; k < (i + blockSize < size ? i + blockSize : size); ++k) {
//                for (int l = k + 1; l < size; ++l) {
//                    (*this)(j, l) -= L(j, k) * U(k, l);
//                }
//            }
//        }
//    }
//}

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

std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    for (int i = 0; i < matrix.size; ++i) {
        for (int j = 0; j < matrix.size; ++j) {
            os << matrix.data[matrix.index(i, j)] << " ";
        }
        os << std::endl;
    }
    return os;
}
