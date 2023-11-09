#ifndef LU_DECOMPOSITION_MATRIX_H
#define LU_DECOMPOSITION_MATRIX_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <stdexcept>

using namespace std;

class Matrix {
private:
    size_t size;
    vector<vector<double>> data;
    static const size_t blockSize = 128;
public:
    vector<double>& operator[](int index);//доступ + изменение
    const vector<double>& operator[](int index) const;//доступ без изменения
    int get_size() const;
    Matrix(int sz);
    friend ostream& operator<<(ostream& os, const Matrix& matrix);
    void input();
    Matrix multiply(const Matrix& other);
    void LU_Decomposition(Matrix& L, Matrix& U);
};

#endif //LU_DECOMPOSITION_MATRIX_H
