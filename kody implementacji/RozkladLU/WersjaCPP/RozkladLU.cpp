#include <iostream>
#include <vector>

using namespace std;

constexpr size_t N = 1024;

static inline size_t idx(size_t r, size_t c) {
    return r * N + c;
}

void lu(vector<double>& A)
{
    vector<double> L(N * N, 0.0);

    // Inicjalizacja macierzy L (jedynki na przekątnej)
    for (size_t i = 0; i < N; ++i) {
        L[idx(i, i)] = 1.0;
    }

    // Główna pętla algorytmu LU
    for (size_t i = 0; i < N; ++i) {

        double aii = A[idx(i, i)];

        for (size_t j = i + 1; j < N; ++j) {

            double lji = 0.0;
            if (aii != 0.0)
                lji = A[idx(j, i)] / aii;

            L[idx(j, i)] = lji;

            // Aktualizacja macierzy A (U)
            for (size_t k = i + 1; k < N; ++k) {
                A[idx(j, k)] -= lji * A[idx(i, k)];
            }
        }
    }

    // (opcjonalnie) można wypisać fragment macierzy
    // cout << "L[0][0] = " << L[idx(0,0)] << endl;
}

int main()
{
    vector<double> A(N * N);

    // Generowanie macierzy diagonalnie dominującej
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            A[idx(i, j)] = (i == j) ? 2.0 * N : 1.0;

    // Wywołanie LU
    lu(A);

    cout << "Done\n";
    return 0;
}