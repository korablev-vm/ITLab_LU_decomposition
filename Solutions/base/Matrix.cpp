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
        throw invalid_argument("������� ������ �������� �� ����� ���� �����������.");
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
//        throw std::invalid_argument("������� L � U ������ ���� ���� �� �������, ��� � �������� �������.");
//    }
//
//    // ������������� ������ L � U
//    for (size_t i = 0; i < size; ++i) {
//        for (size_t j = 0; j < size; ++j) {
//            L[i][j] = 0;
//            U[i][j] = 0;
//            if (i == j) {
//                L[i][j] = 1;
//            }
//        }
//    }
//    // ������� LU-����������
//    for (size_t i = 0; i < size; i += blockSize) {
//        for (size_t j = 0; j < size; j += blockSize) {
//            for (size_t k = 0; k < size; k += blockSize) {
//                // ��������� �����
//                for (size_t ii = i; ii < min(i + blockSize, size); ++ii) {
//                    for (size_t jj = j; jj < min(j + blockSize, size); ++jj) {
//                        if (ii <= jj) {
//                            // ��������� �������� ������� U
//                            U[ii][jj] = data[ii][jj];
//                            for (size_t kk = k; kk < ii; ++kk) {
//                                U[ii][jj] -= L[ii][kk] * U[kk][jj];
//                            }
//                        }
//                        else {
//                            // ��������� �������� ������� L
//                            L[ii][jj] = data[ii][jj];
//                            for (size_t kk = k; kk < jj; ++kk) {
//                                L[ii][jj] -= L[ii][kk] * U[kk][jj];
//                            }
//                            if (U[jj][jj] == 0) {
//                                throw std::runtime_error("������� ����������� ��� �� ����� LU-����������.");
//                            }
//                            L[ii][jj] /= U[jj][jj];
//                        }
//                    }
//                }
//            }
//        }
//    }
//}

//������ ������ (��� �������� ����������) 
void Matrix::LU_Decomposition(Matrix& L, Matrix& U) {
    if (L.size != size || U.size != size) {
        throw std::invalid_argument("������� L � U ������ ���� ���� �� �������, ��� � �������� �������.");
    }

    // ������������� ������ L � U
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            L[i][j] = 0;
            U[i][j] = 0;
            if (i == j) {
                L[i][j] = 1; // ��������� ������� L ����������� ���������
            }
        }
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i <= j) {
                // ��������� �������� ������� U
                U[i][j] = data[i][j];
                for (int k = 0; k < i; k++) {
                    U[i][j] -= L[i][k] * U[k][j];
                }
                if (i == j && U[i][j] == 0) {
                    throw std::runtime_error("������� ����������� ��� �� ����� LU-����������.");
                }
            }
            else {
                // ��������� �������� ������� L
                L[i][j] = data[i][j];
                for (int k = 0; k < j; k++) {
                    L[i][j] -= L[i][k] * U[k][j];
                }
                if (U[j][j] == 0) {
                    if (L[i][j] != 0) { // ���� ������� �� �������, �� ��� ������
                        throw std::runtime_error("������� �� ����� LU-����������.");
                    }
                    // ���� ������� �������, �� ��� ������� ������, � ������ ���
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




//������ ������, �����������: 
/*

//void Matrix::LU_Decomposition(Matrix& L, Matrix& U) {
//    if (L.size != size || U.size != size) {
//        throw std::invalid_argument("������� L � U ������ ���� ���� �� �������, ��� � �������� �������.");
//    }
//    // ������������� ������ L � U
//    for (int i = 0; i < size; ++i) {
//        for (int j = 0; j < size; ++j) {
//            L[i][j] = 0;
//            U[i][j] = 0;
//            if (i == j) {
//                L[i][j] = 1; // ��������� ������� L ����������� ���������
//            }
//        }
//    }
//    for (int i = 0; i < size; i++) {
//        for (int j = 0; j < size; j++) {
//            if (i <= j) {
//                // ��������� �������� ������� U
//                U[i][j] = data[i][j];
//                for (int k = 0; k < i; k++) {
//                    U[i][j] -= L[i][k] * U[k][j];
//                }
//            }
//            else {
//                // ��������� �������� ������� L
//                L[i][j] = data[i][j];
//                for (int k = 0; k < j; k++) {
//                    L[i][j] -= L[i][k] * U[k][j];
//                }
//                if (U[j][j] == 0) {
//                    throw std::runtime_error("������� �� ���� � �������� LU-����������.");
//                }
//                L[i][j] /= U[j][j];
//            }
//        }
//    }
//}
*/