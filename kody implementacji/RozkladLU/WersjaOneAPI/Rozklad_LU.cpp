#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

using namespace sycl;
constexpr std::size_t N = 1024;
static inline std::size_t idx(std::size_t r, std::size_t c) {
    return r * N + c;}

void lu(queue& q,std::vector<double>& AA)
{
    
  double* A = malloc_shared<double>(N * N, q);
  double* L = malloc_shared<double>(N * N, q);

  for (std::size_t i = 0; i < N * N; ++i) {
    A[i] = AA[i];
    L[i] = 0.0;}

  for (std::size_t i = 0; i < N; ++i) {
    L[idx(i, i)] = 1.0; double aii = A[idx(i, i)];
      
    if (i + 1 < N) {
      q.submit([&](handler& h) {
        h.parallel_for(range<1>(N - (i + 1)), [=](id<1> t) {
          std::size_t j = (i + 1) + t[0];
          double lji = 0.0;
          if (aii != 0.0)
            lji = A[idx(j, i)] / aii;
            L[idx(j, i)] = lji;});
          }).wait();

          q.submit([&](handler& h) {range<2> r{N - (i + 1), N - (i + 1)};
            h.parallel_for(r, [=](id<2> t) {
                std::size_t j = (i + 1) + t[0];
                std::size_t k = (i + 1) + t[1];
                double lji = L[idx(j, i)];
                A[idx(j, k)] -= lji * A[idx(i, k)];});
          }).wait();
      }
  }
  free(A, q);
  free(L, q);
}

int main()
{   

  queue q{sycl::cpu_selector_v};
  std::cout << "Device: "<< q.get_device().get_info<info::device::name>()<< "\n";

  std::vector<double> A(N * N);
  // Generowanie macierzy testowej (diagonalnie dominującej)
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = 0; j < N; ++j)
      A[idx(i, j)] = (i == j) ? 2.0 * N : 1.0;

    // Wywołanie funkcji LU
    lu(q, A);

    std::cout << "Done";
    return 0;
}