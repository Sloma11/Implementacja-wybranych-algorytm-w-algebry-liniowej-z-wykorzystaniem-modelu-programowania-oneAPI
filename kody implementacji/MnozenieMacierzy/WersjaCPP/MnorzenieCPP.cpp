#include <iostream>
#include <vector>

using namespace std;

constexpr size_t N = 1024;

static inline size_t idx(size_t r, size_t c) {
    return r * N + c;
}

void matmul(const vector<double>& A,
    const vector<double>& B,
    vector<double>& C)
{
    C.assign(N * N, 0.0);
    for (size_t r = 0; r < N; ++r) {
        for (size_t c = 0; c < N; ++c) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += A[idx(r, k)] * B[idx(k, c)];
            }
            C[idx(r, c)] = sum;
        }
    }
}

int main() {

    vector<double> A(N * N), B(N * N), C;


    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            A[idx(i, j)] = (i == j) ? 2.0 : 0.001 * double((i + j) % 17);
            B[idx(i, j)] = (i == j) ? 1.5 : 0.002 * double((i * 3 + j) % 19);
        }
    }

    matmul(A, B, C);

    cout << "Done\n";
    return 0;
}