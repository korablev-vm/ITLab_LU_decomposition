#include <iostream>
#include <vector>

using namespace std;

class Matrix {
private:
    size_t size;
    vector<vector<double>> data;
public:
    Matrix(size_t sz);
    friend ostream& operator<<(ostream& os, const Matrix& matrix);
    void input();
    Matrix multiply(const Matrix& other);
    void LU_Decomposition();
};