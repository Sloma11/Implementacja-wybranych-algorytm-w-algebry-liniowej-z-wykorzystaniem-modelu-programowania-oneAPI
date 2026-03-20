#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace sycl;

constexpr size_t N = 1024;
static inline size_t idx(size_t r, size_t c) {
  return r * N + c;
}

void matmul(queue& q,const vector<double>& A_host,
            const vector<double>& B_host)
{
  double* A = malloc_shared<double>(N * N, q);
  double* B = malloc_shared<double>(N * N, q);
  double* C = malloc_shared<double>(N * N, q);

  for (size_t i = 0; i < N * N; ++i) {
    A[i] = A_host[i];
    B[i] = B_host[i];
    C[i] = 0.0
    ;
  }

  q.submit([&](handler& h) {
    h.parallel_for(range<2>(N, N), [=](id<2> it) {
      const size_t r = it[0];
      const size_t c = it[1];

      double sum = 0.0;
      for (size_t k = 0; k < N; ++k) {
        sum += A[idx(r, k)] * B[idx(k, c)];
      }
      C[idx(r, c)] = sum;
    });
  });

  free(A, q);
  free(B, q);
  free(C, q);
}


int main() {
  queue q{default_selector_v};
  cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

  vector<double> A(N * N), B(N * N);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      A[idx(i, j)] = (i == j) ? 2.0 : 0.001 * double((i + j) % 17);
      B[idx(i, j)] = (i == j) ? 1.5 : 0.002 * double((i * 3 + j) % 19);
    }
  }

  matmul(q, A, B);

  cout << "Done\n";
  return 0;
}