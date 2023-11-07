#include "Matrix.h"

Matrix::Matrix(int sz) : size(sz), data(sz, vector<double>(sz, 0)) {}

void Matrix::input() {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            cin >> data[i][j];
        }
    }
}

Matrix Matrix::multiply(const Matrix& other) {
    if (size != other.size)
        throw invalid_argument("Матрицы разных размеров не могут быть перемножены.");
    Matrix result(size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                result.data[i][j] += data[i][k] * other.data[k][j];
            }
        }
    }
    return result;
}

vector<double>& Matrix::operator[](int index) {
    return data[index];
}
const vector<double>& Matrix::operator[](int index) const {
    return data[index];
}
int Matrix::get_size() const {
    return size;
}

//void Matrix::LU_Decomposition(Matrix& L, Matrix& U) {
//    if (L.size != size || U.size != size) {
//        throw std::invalid_argument("Матрицы L и U должны быть того же размера, что и исходная матрица.");
//    }
//
//    // Инициализация матриц L и U
//    for (size_t i = 0; i < size; ++i) {
//        for (size_t j = 0; j < size; ++j) {
//            L[i][j] = 0;
//            U[i][j] = 0;
//            if (i == j) {
//                L[i][j] = 1;
//            }
//        }
//    }
//    // Блочное LU-разложение
//    for (size_t i = 0; i < size; i += blockSize) {
//        for (size_t j = 0; j < size; j += blockSize) {
//            for (size_t k = 0; k < size; k += blockSize) {
//                // Обработка блока
//                for (size_t ii = i; ii < min(i + blockSize, size); ++ii) {
//                    for (size_t jj = j; jj < min(j + blockSize, size); ++jj) {
//                        if (ii <= jj) {
//                            // Вычисляем элементы матрицы U
//                            U[ii][jj] = data[ii][jj];
//                            for (size_t kk = k; kk < ii; ++kk) {
//                                U[ii][jj] -= L[ii][kk] * U[kk][jj];
//                            }
//                        }
//                        else {
//                            // Вычисляем элементы матрицы L
//                            L[ii][jj] = data[ii][jj];
//                            for (size_t kk = k; kk < jj; ++kk) {
//                                L[ii][jj] -= L[ii][kk] * U[kk][jj];
//                            }
//                            if (U[jj][jj] == 0) {
//                                throw std::runtime_error("Матрица вырожденная или не имеет LU-разложения.");
//                            }
//                            L[ii][jj] /= U[jj][jj];
//                        }
//                    }
//                }
//            }
//        }
//    }
//}

//Первая версия (без блочного разложения)
void Matrix::LU_Decomposition(Matrix& L, Matrix& U) {
    if (L.size != size || U.size != size) {
        throw std::invalid_argument("Матрицы L и U должны быть того же размера, что и исходная матрица.");
    }

    // Инициализация матриц L и U
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            L[i][j] = 0;
            U[i][j] = 0;
            if (i == j) {
                L[i][j] = 1; // Диагональ матрицы L заполняется единицами
            }
        }
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i <= j) {
                // Вычисляем элементы матрицы U
                U[i][j] = data[i][j];
                for (int k = 0; k < i; k++) {
                    U[i][j] -= L[i][k] * U[k][j];
                }
                if (i == j && U[i][j] == 0) {
                    throw std::runtime_error("Матрица вырожденная или не имеет LU-разложения.");
                }
            }
            else {
                // Вычисляем элементы матрицы L
                L[i][j] = data[i][j];
                for (int k = 0; k < j; k++) {
                    L[i][j] -= L[i][k] * U[k][j];
                }
                if (U[j][j] == 0) {
                    if (L[i][j] != 0) { // Если элемент не нулевой, то это ошибка
                        throw std::runtime_error("Матрица не имеет LU-разложения.");
                    }
                    // Если элемент нулевой, то это нулевая строка, и ошибки нет
                }
                else {
                    L[i][j] /= U[j][j];
                }
            }
        }
    }
}

ostream& operator<<(ostream& os, const Matrix& matrix) {
    for (int i = 0; i < matrix.size; ++i) {
        for (int j = 0; j < matrix.size; ++j) {
            os << matrix.data[i][j] << " ";
        }
        os << endl;
    }
    return os;
}




//старые версии, недоработки:
/*

//void Matrix::LU_Decomposition(Matrix& L, Matrix& U) {
//    if (L.size != size || U.size != size) {
//        throw std::invalid_argument("Матрицы L и U должны быть того же размера, что и исходная матрица.");
//    }
//    // Инициализация матриц L и U
//    for (int i = 0; i < size; ++i) {
//        for (int j = 0; j < size; ++j) {
//            L[i][j] = 0;
//            U[i][j] = 0;
//            if (i == j) {
//                L[i][j] = 1; // Диагональ матрицы L заполняется единицами
//            }
//        }
//    }
//    for (int i = 0; i < size; i++) {
//        for (int j = 0; j < size; j++) {
//            if (i <= j) {
//                // Вычисляем элементы матрицы U
//                U[i][j] = data[i][j];
//                for (int k = 0; k < i; k++) {
//                    U[i][j] -= L[i][k] * U[k][j];
//                }
//            }
//            else {
//                // Вычисляем элементы матрицы L
//                L[i][j] = data[i][j];
//                for (int k = 0; k < j; k++) {
//                    L[i][j] -= L[i][k] * U[k][j];
//                }
//                if (U[j][j] == 0) {
//                    throw std::runtime_error("Деление на ноль в процессе LU-разложения.");
//                }
//                L[i][j] /= U[j][j];
//            }
//        }
//    }
//}
*/