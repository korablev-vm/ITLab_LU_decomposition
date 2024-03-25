#ifndef LU_DECOMPOSITION_MATRIX_H
#define LU_DECOMPOSITION_MATRIX_H

#include <iostream>
#include <stdexcept>
#include <omp.h>
#include <cstring>
#include <algorithm>
#include <immintrin.h>
#include <thread>
#include <iomanip>


template<typename T>
class Matrix {
private:
    size_t size; // Размер матрицы
    T* data; // Данные матрицы
    const size_t blockSize = 32; // Размер блока для оптимизации

    // Индексация элементов матрицы
    inline size_t index(int row, int col) const {
        return row * size + col;
    }

public:
    // Конструкторы и деструкторы
    Matrix(int sz); 
    Matrix(const Matrix<T>& other); 
    Matrix(Matrix&& other) noexcept; 
    ~Matrix();

    // Перегруженные операторы присваивания
    Matrix& operator=(const Matrix<T>& other);
    Matrix& operator=(Matrix&& other) noexcept; 

    T& operator()(int row, int col);
    const T& operator()(int row, int col) const; 
    T& operator[](int i)
    {
        return data[i];
    }
    const T& operator[](int i) const
    {
        return data[i];
    }

    // Перегруженные операторы сравнения
    bool operator==(const Matrix& other) const; 
    bool operator!=(const Matrix& other) const; 

    // Перегруженные арифметические операторы
    Matrix& operator+=(const Matrix& m); 
    Matrix operator-(const Matrix& m); 
    Matrix operator+(const Matrix& m); 

    // Геттеры
    size_t get_size() const; 
    T* get_data() const; 

    // Статические методы
    static Matrix createDiagonallyDominantMatrix(size_t n); 

    // Методы класса
    void input(); 
    Matrix multiply(const Matrix& other); 
    void LU_Decomposition(Matrix& L, Matrix& U);
    void LU_Decomposition_base(Matrix& L, Matrix& U); 
    T norm() const noexcept; 
    inline friend void parallel_block_mult6(Matrix& F, Matrix& S, Matrix& RES)
    {
        if ((F.size != S.size) || (F.size != RES.size) || (S.size != RES.size)) throw std::invalid_argument("matrices sizes should match!");
        if ((&F == &RES) || (&S == &RES)) throw std::invalid_argument("RES cannot be used as argument F or S");

        const int block_size_row = 96, block_size_col = 192;

        const int sub_block_size = 6, sub_block_size2 = 48, sub_block_size3 = 16; //sub_sub_block_size3 == 2 * sizeof(__m512); (byte)
        // +- same time for sub_block_size2 = 48, 96
        const auto processor_count = std::thread::hardware_concurrency();// / 2;

        int t = F.size - (F.size % sub_block_size);// i
        int l = S.size - (S.size % sub_block_size3);// j
        int s = F.size - (F.size % sub_block_size2);// k

#pragma omp parallel for num_threads(processor_count)
        for (int i1 = 0; i1 < F.size; i1 += block_size_row)
            for (int k1 = 0; k1 < F.size; k1 += block_size_col)
                for (int j1 = 0; j1 < S.size; j1 += block_size_row)

                    for (int i3 = i1; i3 < i1 + block_size_row && i3 < t; i3 += sub_block_size)
                        for (int k3 = k1; k3 < k1 + block_size_col && k3 < s; k3 += sub_block_size2)
                            for (int j3 = j1; j3 < j1 + block_size_row && j3 < l; j3 += sub_block_size3)
                            {
                                __m512 c[(sub_block_size << 1)];
                                __m512 a, b1, b2;

                                for (int i4 = 0; i4 < sub_block_size; i4++)
                                    for (int j4 = 0; j4 < 2; j4++)
                                        c[(i4 << 1) + j4] = _mm512_loadu_pd(&RES[(i4 + i3) * RES.size + j3 + j4 * (sub_block_size3 >> 1)]);

                                for (int k5 = 0; k5 < sub_block_size2; k5++)
                                {
                                    b1 = _mm512_loadu_pd(&S[(k3 + k5) * S.size + j3]);
                                    b2 = _mm512_loadu_pd(&S[(k3 + k5) * S.size + j3 + (sub_block_size3 >> 1)]);

                                    for (int i5 = 0; i5 < sub_block_size; i5++)
                                    {
                                        int i = i5 << 1;

                                        a = _mm512_set1_pd(F[(i3 + i5) * F.size + (k3 + k5)]);

                                        c[i] = _mm512_fmadd_pd(a, b1, c[i]);
                                        c[i + 1] = _mm512_fmadd_pd(a, b2, c[i + 1]);
                                    }
                                }

                                for (int i6 = 0; i6 < sub_block_size; i6++)
                                    for (int j6 = 0; j6 < 2; j6++)
                                        _mm512_storeu_pd(&RES[(i6 + i3) * RES.size + j3 + j6 * (sub_block_size3 >> 1)], c[(i6 << 1) + j6]);
                            }

        if (S.size == l)
        {
            if ((F.size != t) && (F.size != s))
            {
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < F.size; i++)
                    for (int k = s; k < F.size; k++)
#pragma omp simd
                        for (int j = 0; j < S.size; j++)
                            RES[i * RES.size + j] += F[i * F.size + k] * S[k * S.size + j];

#pragma omp parallel for num_threads(processor_count)
                for (int i = t; i < F.size; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = 0; j < S.size; j++)
                            RES[i * RES.size + j] += F[i * F.size + k] * S[k * S.size + j];
            }
            else if ((F.size != t) && (F.size == s))
            {
#pragma omp parallel for num_threads(processor_count)
                for (int i = t; i < F.size; i++)
                    for (int k = 0; k < F.size; k++)
#pragma omp simd
                        for (int j = 0; j < S.size; j++)
                            RES[i * RES.size + j] += F[i * F.size + k] * S[k * S.size + j];

            }
            else if ((F.size == t) && (F.size != s))
            {
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < F.size; i++)
                    for (int k = s; k < F.size; k++)
#pragma omp simd
                        for (int j = 0; j < S.size; j++)
                            RES[i * RES.size + j] += F[i * F.size + k] * S[k * S.size + j];

            }
        }
        else if (S.size != l)
        {
            if ((F.size != t) && (F.size != s))
            {
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < F.size; i++)
                    for (int k = s; k < F.size; k++)
#pragma omp simd
                        for (int j = 0; j < S.size; j++)
                            RES[i * RES.size + j] += F[i * F.size + k] * S[k * S.size + j];

#pragma omp parallel for num_threads(processor_count)
                for (int i = t; i < F.size; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = 0; j < S.size; j++)
                            RES[i * RES.size + j] += F[i * F.size + k] * S[k * S.size + j];

#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < t; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = l; j < S.size; j++)
                            RES[i * RES.size + j] += F(i,k) * S(k,j);
            }
            else if ((F.size != t) && (F.size == s))
            {
                //int t = F.size - (F.size % block_size_row);// i
#pragma omp parallel for num_threads(processor_count)
                for (int i = t; i < F.size; i++)
                    for (int k = 0; k < F.size; k++)
#pragma omp simd
                        for (int j = 0; j < S.size; j++)
                            RES(i,j) += F(i,k) * S(k,j);

                //int l = S.size - (S.size % block_size_row);// j
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < t; i++)
                    for (int k = 0; k < F.size; k++)
#pragma omp simd
                        for (int j = l; j < S.size; j++)
                            RES(i,j) += F(i, k) * S(k, j);
            }
            else if ((F.size == t) && (F.size != s))
            {
                //int s = F.size - (F.size % block_size_col);// k
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < F.size; i++)
                    for (int k = s; k < F.size; k++)
#pragma omp simd
                        for (int j = 0; j < S.size; j++)
                            RES(i,j) += F(i,k) * S(k,j);

                //int l = S.size - (S.size % block_size_row);// j
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < F.size; i++)
                    for (int k = 0; k < s; k++)
#pragma omp simd
                        for (int j = l; j < S.size; j++)
                            RES(i,j) += F(i, k) * S(k, j);
            }
            else if ((F.size == t) && (F.size == s))
            {
                //int l = S.size - (S.size % block_size_row);// j
#pragma omp parallel for num_threads(processor_count)
                for (int i = 0; i < F.size; i++)
                    for (int k = 0; k < F.size; k++)
#pragma omp simd
                        for (int j = l; j < S.size; j++)
                            RES(i,j) += F(i, j) * S(i, j);
            }
        }
    }


    // Дружественные функции
    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix); // Вывод матрицы в поток
};


#endif //LU_DECOMPOSITION_MATRIX_H

template<typename T>
Matrix<T>::Matrix<T>(Matrix<T>&& other) noexcept : size(other.size), data(other.data) {
    other.size = 0;
    other.data = nullptr;
}

template <typename T>
bool Matrix<T>::operator==(const Matrix<T>& other) const {
    if (size != other.size) return false;
    for (size_t i = 0; i < size * size; ++i)
        if (data[i] != other.data[i]) return false;
    return true;
}

template<typename T>
bool Matrix<T>::operator!=(const Matrix<T>& other) const {
    return !(*this == other);
}

template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& m)
{
    if (size != m.size) throw std::invalid_argument("matrices sizes should match!");

    Matrix res(*this);
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
#pragma omp simd
        for (int j = 0; j < size; j++)
            res(i, j) += m(i, j);
    }
    return res;
}

template<typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& m)
{
    if (size != m.size) throw std::invalid_argument("matrices sizes should match!");

#pragma omp parallel for
    for (int i = 0; i < size; i++)
#pragma omp simd
        for (int j = 0; j < size; j++)
            data(i, j) += m(i, j);

    return *this;
}

template<typename T>
T Matrix<T>::norm() const noexcept
{
    T res{};

#pragma omp parallel for reduction(+:res)
    for (int i = 0; i < size * size; i++)
        res += data[i] * data[i];

    return res;
}

template<typename T>
Matrix<T>::Matrix(int sz) : size(sz) {
    data = new T[sz * sz];
    memset(data, 0, sz * sz * sizeof(T));
}

template<typename T>
Matrix<T>::Matrix(const Matrix& other) : size(other.size) {
    data = new T[size * size];
    memcpy(data, other.data, size * size * sizeof(T));
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other) {
    if (this != &other) {
        delete[] data;
        size = other.size;
        data = new T[size * size];
        memcpy(data, other.data, size * size * sizeof(T));
    }
    return *this;
}

template<typename T>
Matrix<T>::~Matrix<T>() {
    delete[] data;
}

template<typename T>
T& Matrix<T>::operator()(int row, int col) {
    return data[index(row, col)];
}

template<typename T>
const T& Matrix<T>::operator()(int row, int col) const {
    return data[index(row, col)];
}

template<typename T>
size_t Matrix<T>::get_size() const {
    return size;
}

template<typename T>
T* Matrix<T>::get_data() const {
    return data;
}

template<typename T>
void Matrix<T>::input() {
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            std::cin >> data[index(i, j)];
        }
    }
}

template<typename T>
Matrix<T> Matrix<T>::multiply(const Matrix<T>& other) {
    if (size != other.size)
        throw std::invalid_argument("Matrix sizes do not match for multiplication.");
    Matrix result(size);
    for (size_t i = 0; i < size; ++i)
        for (size_t j = 0; j < size; ++j)
            for (size_t k = 0; k < size; ++k)
                result(i, j) += (*this)(i, k) * other(k, j);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::createDiagonallyDominantMatrix(size_t n) {
    Matrix m(n);
    std::srand(std::time(nullptr));
    for (size_t i = 0; i < n; ++i) {
        T rowSum = 0;
        for (size_t j = 0; j < n; ++j) {
            if (i != j) {
                m(i, j) = static_cast<T>(std::rand() % 100) + 1;
                rowSum += std::abs(m(i, j));
            }
        }
        m(i, i) = rowSum + 1;
    }
    return m;
}

template<typename T>
void Matrix<T>::LU_Decomposition(Matrix& L, Matrix& U) {
    Matrix<T> tmp(size);

    // Инициализация L и U
#pragma omp parallel for simd
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            L(i, j) = 0;
            U(i, j) = 0;
            if (i == j) {
                L(i, i) = 1;
            }
        }
    }

    Matrix<T> temp(size);

    // Разложение
    for (size_t i = 0; i < size; i += blockSize) {
        size_t limit = std::min(i + blockSize, size);

        // Вычисление блоков L и U
        for (size_t j = i; j < limit; ++j) {
            // Обновление L
            for (size_t k = i; k < j; ++k) {
                parallel_block_mult6(L, U, temp); 
                L(j, k) = temp(j, k) / U(k, k);
            }

            // Обновление U
            for (size_t k = j; k < limit; ++k) {
                parallel_block_mult6(L, U, temp);
                U(j, k) = temp(j, k);
            }
        }

        for (size_t j = limit; j < size; ++j) {
            for (size_t k = i; k < limit; ++k) {
                parallel_block_mult6(L, U, temp);
                L(j, k) = temp(j, k) / U(k, k);
            }
        }

        for (size_t j = i; j < limit; ++j) {
            for (size_t k = limit; k < size; ++k) {
                parallel_block_mult6(L, U, temp);
                U(j, k) = temp(j, k);
            }
        }

        for (size_t j = limit; j < size; ++j) {
            for (size_t k = limit; k < size; ++k) {
                parallel_block_mult6(L, U, temp);
                (*this)(j, k) -= temp(j, k);
            }
        }
    }
}



//// рабочая версия, но пока медленная(
//template<typename T>
//void Matrix<T>::LU_Decomposition(Matrix& L, Matrix& U) {
//    Matrix<T> tmp;
//
//    // Инициализация L и U
//#pragma omp parallel for simd
//    for (size_t i = 0; i < size; ++i) {
//        for (size_t j = 0; j < size; ++j) {
//            L(i, j) = 0;
//            U(i, j) = 0;
//            if (i == j) {
//                L(i, i) = 1;
//            }
//        }
//    }
//    // Разложение
//    for (size_t i = 0; i < size; i += blockSize) {
//        size_t limit = std::min(i + blockSize, size);
//        // Вычисление блоков L и U
//        for (size_t j = i; j < limit; ++j) {
//#pragma omp parallel for simd
//            for (size_t k = i; k < j; ++k) {
//                T sum = 0;
//                for (size_t m = 0; m < k; ++m) {
//                    sum += L(j, m) * U(m, k);
//                }
//                L(j, k) = ((*this)(j, k) - sum) / U(k, k);
//            }
//#pragma omp parallel for simd
//            for (size_t k = j; k < limit; ++k) {
//                T sum = 0;
//                for (size_t m = 0; m < j; ++m) {
//                    sum += L(j, m) * U(m, k);
//                }
//                U(j, k) = (*this)(j, k) - sum;
//            }
//        }
//#pragma omp parallel for simd
//        for (size_t j = limit; j < size; ++j) {
//            for (size_t k = i; k < limit; ++k) {
//                T sum = 0.0;
//                for (size_t m = 0; m < k; ++m) {
//                    sum += L(j, m) * U(m, k);
//                }
//                L(j, k) = ((*this)(j, k) - sum) / U(k, k);
//            }
//        }
//#pragma omp parallel for simd
//        for (size_t j = i; j < limit; ++j) {
//            for (size_t k = limit; k < size; ++k) {
//                T sum = 0.0;
//                for (size_t m = 0; m < j; ++m) {
//                    sum += L(j, m) * U(m, k);
//                }
//                U(j, k) = (*this)(j, k) - sum;
//            }
//        }
//        // Применение вычисленных блоков L и U для обновления остальной части (A')
//#pragma omp parallel for simd
//        for (size_t j = limit; j < size; ++j) {
//            for (size_t k = limit; k < size; ++k) {
//                T sum = 0.0;
//                for (size_t m = i; m < limit; ++m) {
//                    sum += L(j, m) * U(m, k);
//                }
//                (*this)(j, k) -= sum;
//            }
//        }
//    }
//}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix) {
    for (int i = 0; i < matrix.size; ++i) {
        for (int j = 0; j < matrix.size; ++j) {
            os << matrix.data[matrix.index(i, j)] << " ";
        }
        os << std::endl;
    }
    return os;
}

template<typename T>
void Matrix<T>::LU_Decomposition_base(Matrix<T>& L, Matrix<T>& U) {
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
