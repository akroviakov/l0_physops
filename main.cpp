#include <CL/sycl.hpp>

int printTest() {
  const size_t N = 8;
  sycl::queue q;
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  int *arr = sycl::malloc_shared<int>(N, q);
  for (size_t i = 0; i < N; i++)
    arr[i] = i;

  std::cout << "Before:" << std::endl;
  for (size_t i = 0; i < N; i++)
    std::cout << " - " << arr[i] << std::endl;

  q.parallel_for(sycl::range{static_cast<size_t>(N)}, [=](sycl::id<1> idx) {
     arr[idx] *= 2;
   }).wait();

  std::cout << "After:" << std::endl;
  for (size_t i = 0; i < N; i++)
    std::cout << " - " << arr[i] << std::endl;
  sycl::free(arr, q);
  return 0;
}

int main() { printTest(); }