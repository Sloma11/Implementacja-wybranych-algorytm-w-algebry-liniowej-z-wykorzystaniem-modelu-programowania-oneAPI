#include <iostream>
#include <vector>

using namespace std;
constexpr size_t N = 1024;

static inline size_t idx(size_t r, size_t c) {
    return r * N + c;
}

void redukcja_wsteczna(const vector<double>& A_host,
    const vector<double>& b_host,
    vector<double>& x_host)
{
    vector<double> b = b_host;
    x_host.assign(N, 0.0);

    // Iterujemy od N-1 do 0
    for (size_t ii = N; ii-- > 0; ) {
        const double aii = A_host[idx(ii, ii)];
        const double xi = b[ii] / aii;
        x_host[ii] = xi;

        for (size_t j = 0; j < ii; ++j) {
            b[j] -= A_host[idx(j, ii)] * xi;

        }
    }
}

int main() {
    cout << "\n";

    vector<double> A(N * N, 0.0);
    vector<double> x_true(N, 0.0);
    vector<double> b(N, 0.0);

    for (size_t i = 0; i < N; ++i) {
        x_true[i] = 1.0 + (i % 7) * 0.1;
    }

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i; j < N; ++j) {
            A[idx(i, j)] = (i == j) ? (2.0 + (i % 5) * 0.01)
                : (0.001 * ((j - i) % 13));
        }
    }
    for (size_t i = 0; i < N; ++i) {
        double sum = 0.0;
        for (size_t j = i; j < N; ++j) {
            sum += A[idx(i, j)] * x_true[j];
        }
        b[i] = sum;
    }

    vector<double> x;
    redukcja_wsteczna(A, b, x);

    cout << "Done\n";
    return 0;
}