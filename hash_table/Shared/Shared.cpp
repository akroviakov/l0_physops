#include "Shared.h"
#include <CL/sycl.hpp>

void set_valid_pos_flag(int32_t *pos_buff, const int32_t *count_buff,
                        const int64_t entry_count) {
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
     h.parallel_for(sycl::range{static_cast<size_t>(entry_count)},
                    [=](sycl::id<1> idx) {
                      if (count_buff[idx]) {
                        pos_buff[idx] = VALID_POS_FLAG;
                      }
                    });
   }).wait();
}

void set_valid_pos(int32_t *pos_buff, int32_t *count_buff,
                   const int64_t entry_count) {
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
     h.parallel_for(sycl::range{static_cast<size_t>(entry_count)},
                    [=](sycl::id<1> idx) {
                      if (VALID_POS_FLAG == pos_buff[idx]) {
                        pos_buff[idx] = !idx ? 0 : count_buff[idx - 1];
                      }
                    });
   }).wait();
}

void init_hash_join_buff_on_l0(int32_t *groups_buffer,
                               const int64_t hash_entry_count,
                               const int32_t invalid_slot_val) {
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
     h.parallel_for(
         sycl::range{static_cast<size_t>(hash_entry_count)},
         [=](sycl::id<1> idx) { groups_buffer[idx] = invalid_slot_val; });
   }).wait();
}
