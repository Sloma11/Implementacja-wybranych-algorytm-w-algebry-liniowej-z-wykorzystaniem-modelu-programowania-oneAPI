#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

using namespace sycl;

constexpr std::size_t N = 1024;
static inline std::size_t idx(std::size_t r, std::size_t c) {
  return r * N + c;}

void redukcja_wsteczna(queue& q, std::vector<double>& A_host,
                        std::vector<double>& b_host)
{
  double* A = malloc_shared<double>(N * N, q);
  double* b = malloc_shared<double>(N, q);
  double* x = malloc_shared<double>(N, q);

  for (std::size_t i = 0; i < N * N; ++i) A[i] = A_host[i];
  for (std::size_t i = 0; i < N; ++i) b[i] = b_host[i];

  for (std::size_t ii = N; ii-- > 0; ) {
    const double aii = A[idx(ii, ii)];
    const double xi = b[ii] / aii;
    x[ii] = xi;
    if (ii > 0) {
      q.submit([&](handler& h) {
        h.parallel_for(range<1>(ii), [=](id<1> tid) {
          std::size_t j = tid[0];
          b[j] -= A[idx(j, ii)] * xi;
        });
      });
      q.wait(); 
    }
  }
  free(A, q);
  free(b, q);
  free(x, q);
}

int main() {
  queue q{sycl::cpu_selector_v};
  std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

  std::vector<double> A(N * N, 0.0);
  std::vector<double> x_true(N, 0.0);
  std::vector<double> b(N, 0.0);

  for (std::size_t i = 0; i < N; ++i) x_true[i] = 1.0 + (i % 7) * 0.1;

  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = i; j < N; ++j) {
      A[idx(i, j)] = (i == j) ? (2.0 + (i % 5) * 0.01) : (0.001 * ((j - i) % 13));
    }
  }

  for (std::size_t i = 0; i < N; ++i) {
    double sum = 0.0;
    for (std::size_t j = i; j < N; ++j) {
      sum += A[idx(i, j)] * x_true[j];
    }
    b[i] = sum;
  }

  redukcja_wsteczna(q, A, b);
  
  std::cout << "Done\n";
  return 0;
}