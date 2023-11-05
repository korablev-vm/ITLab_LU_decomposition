#include <iostream>
#include <vector>
#include <algorithm>
#include <stdexcept>

using namespace std;

class Matrix {
private:
    size_t size;
    vector<vector<double>> data;
    static const size_t blockSize = 64;
public:
    vector<double>& operator[](int index);//������ + ���������
    const vector<double>& operator[](int index) const;//������ ��� ���������
    int get_size() const;
    Matrix(int sz);
    friend ostream& operator<<(ostream& os, const Matrix& matrix);
    void input();
    Matrix multiply(const Matrix& other);
    void LU_Decomposition(Matrix& L, Matrix& U);
};