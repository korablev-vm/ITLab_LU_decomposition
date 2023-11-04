#include "Matrix.h"

Matrix::Matrix(size_t sz) : size(sz), data(sz, vector<double>(sz, 0)) {}

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
size_t Matrix::get_size() const {
    return size;
}

void Matrix::LU_Decomposition() {
    vector<vector<double>> L(size, vector<double>(size, 0));
    vector<vector<double>> U(size, vector<double>(size, 0));

    for (int i = 0; i < size; i++) {
        L[i][i] = 1;
        for (int j = 0; j < size; j++) {
            if (i <= j) {
                U[i][j] = data[i][j];
                for (int k = 0; k < i; k++) {
                    U[i][j] -= L[i][k] * U[k][j];
                }
            }
            else {
                L[i][j] = data[i][j];
                for (int k = 0; k < j; k++) {
                    L[i][j] -= L[i][k] * U[k][j];
                }
                L[i][j] /= U[j][j];
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