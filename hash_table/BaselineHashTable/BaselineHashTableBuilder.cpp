#include <CL/sycl.hpp>
#include <cassert>
#include <limits>
#include <type_traits>

#include "../GenericKeyHandler.h"
#include "../MurMurHash.h"
#include "../Shared/Shared.h"
#include "BaselineHashTableBuilder.h"
#include "BaselineHashTableHelpers.h"

template <typename T>
inline bool keys_are_equal(const T *key1, const T *key2,
                           const size_t key_component_count) {
  for (size_t idx = 0; idx < key_component_count; idx++) {
    if (key1[idx] != key2[idx]) {
      return false;
    }
  }
  return true;
}

uint8_t get_rank(const uint64_t x, const uint32_t b) {
  return std::min(b, static_cast<uint32_t>(x ? sycl::clz(x) : 64)) + 1;
}

// This executes on the device (no need to create queues)
template <typename T>
T *get_matching_baseline_hash_slot_at(int8_t *hash_buff, const uint32_t h,
                                      const T *key,
                                      const size_t key_component_count,
                                      const int64_t hash_entry_size) {
  const uint32_t off =
      h * hash_entry_size; // Get row's offset in the hash table
  T *row_ptr = reinterpret_cast<T *>(hash_buff + off); // Get row itself
  const T empty_key_const = get_invalid_key<T>();      // empty_key constant
  T empty_key = empty_key_const;
  sycl::atomic_ref<T, sycl::memory_order::acq_rel, sycl::memory_scope::device>
      atomic_row_ptr(*row_ptr);
  const bool success = atomic_row_ptr.compare_exchange_strong(
      empty_key, *key,
      sycl::memory_order::acq_rel); // Expect empty key, put write pending flag
  if (success) {
    for (int64_t i = 1; i < key_component_count; ++i) {
      sycl::atomic_ref<T, sycl::memory_order::acq_rel,
                       sycl::memory_scope::device>
          atomic_row_ptr_i(*(row_ptr + i));
      atomic_row_ptr_i.store(key[i], sycl::memory_order::release);
    }
  }
  if (key_component_count > 1) {
    sycl::atomic_ref<T, sycl::memory_order::acq_rel, sycl::memory_scope::device>
        atomic_row_ptr_last(*(row_ptr + key_component_count - 1));
    while (atomic_row_ptr_last.load(sycl::memory_order::acquire) ==
           empty_key_const) {
      // spin until the winning thread has finished writing the entire key and
      // the init value
    }
  }
  bool match = true;
  for (uint32_t i = 0; i < key_component_count; ++i) {
    if (row_ptr[i] != key[i]) {
      return nullptr;
    }
  }
  return reinterpret_cast<T *>(row_ptr + key_component_count);
}

// This executes on the device (no need to create queues)
template <typename T>
int write_baseline_hash_slot_for_semi_join(
    const int32_t val, int8_t *hash_buff, const int64_t entry_count,
    const T *key, const size_t key_component_count, const bool with_val_slot,
    const int32_t invalid_slot_val, const size_t key_size_in_bytes,
    const size_t hash_entry_size) {
  const uint32_t h = MurmurHash1Impl(key, key_size_in_bytes, 0) % entry_count;
  T *matching_group = get_matching_baseline_hash_slot_at(
      hash_buff, h, key, key_component_count, hash_entry_size);
  if (!matching_group) {
    uint32_t h_probe = (h + 1) % entry_count;
    while (h_probe != h) {
      matching_group = get_matching_baseline_hash_slot_at(
          hash_buff, h_probe, key, key_component_count, hash_entry_size);
      if (matching_group) {
        break;
      }
      h_probe = (h_probe + 1) % entry_count;
    }
  }
  if (!matching_group) {
    return -2;
  }
  if (!with_val_slot) {
    return 0;
  }
  T invalid_slot_val_copy = static_cast<T>(invalid_slot_val);
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>
      atomic_matching_group(*matching_group);
  atomic_matching_group.compare_exchange_strong(invalid_slot_val_copy,
                                                static_cast<T>(val));
  return 0;
}

// This executes on the device (no need to create queues)
template <typename T>
int write_baseline_hash_slot(const int32_t val, int8_t *hash_buff,
                             const int64_t entry_count, const T *key,
                             const size_t key_component_count,
                             const bool with_val_slot,
                             const int32_t invalid_slot_val,
                             const size_t key_size_in_bytes,
                             const size_t hash_entry_size) {
  const uint32_t h = MurmurHash1Impl(key, key_size_in_bytes, 0) %
                     entry_count; // get row's position in the hash table
  T *matching_group = get_matching_baseline_hash_slot_at(
      hash_buff, h, key, key_component_count,
      hash_entry_size);  // Try to write the slot
  if (!matching_group) { // If couldn't write the slot for some reason (e.g.,
                         // someone else wrote to that slot another key)
    uint32_t h_probe = (h + 1) % entry_count; // we start linear probing
    while (h_probe != h) { // we go until we make the full circle
      matching_group = get_matching_baseline_hash_slot_at(
          hash_buff, h_probe, key, key_component_count, hash_entry_size);
      if (matching_group) {
        break;
      }
      h_probe = (h_probe + 1) % entry_count;
    }
  }
  if (!matching_group) { // If we went the full ht circle and couldn't find a
                         // slot
    return -2;
  }
  if (!with_val_slot) { // If the slot shouldn't have value (key only), just
                        // return
    return 0;
  }
  T invalid_slot_val_copy = static_cast<T>(invalid_slot_val);
  sycl::atomic_ref<T, sycl::memory_order::acq_rel, sycl::memory_scope::device>
      atomic_matching_group(*matching_group);
  if (!atomic_matching_group.compare_exchange_strong(
          invalid_slot_val_copy, static_cast<T>(val),
          sycl::memory_order::acq_rel)) { // We write the value
    return -1; // If couldn't write (slot was not invalid_slot_val_copy) ->
               // really one-to-one?
  }
  return 0;
}

template <typename T, typename KEY_HANDLER>
void fill_row_ids_baseline(int32_t *buff, const T *composite_key_dict,
                           const int64_t hash_entry_count,
                           const int32_t invalid_slot_val, const KEY_HANDLER *f,
                           const int64_t num_elems) {
  assert(composite_key_dict);
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
     h.parallel_for(
         sycl::range{static_cast<size_t>(hash_entry_count)},
         [=](sycl::id<1> tuple_idx) {
           int32_t *pos_buff = buff;
           int32_t *count_buff = buff + hash_entry_count;
           int32_t *id_buff = count_buff + hash_entry_count;
           const size_t key_size_in_bytes =
               f->get_key_component_count() * sizeof(T);
           auto key_buff_handler = [composite_key_dict, hash_entry_count,
                                    pos_buff, count_buff, id_buff,
                                    key_size_in_bytes](
                                       const int64_t row_index,
                                       const T *key_scratch_buff,
                                       const size_t key_component_count) {
             const T *matching_group = get_matching_baseline_hash_slot_readonly(
                 key_scratch_buff, key_component_count, composite_key_dict,
                 hash_entry_count, key_size_in_bytes);
             const auto entry_idx =
                 (matching_group - composite_key_dict) / key_component_count;
             int32_t *pos_ptr = pos_buff + entry_idx;
             const auto bin_idx = pos_ptr - pos_buff;
             sycl::atomic_ref<int32_t, sycl::memory_order::acq_rel,
                              sycl::memory_scope::device>
                 atomic_count_buff(*(count_buff + bin_idx));
             const auto id_buff_idx = atomic_count_buff.fetch_add(1) + *pos_ptr;
             id_buff[id_buff_idx] = static_cast<int32_t>(row_index);
             return 0;
           };

           JoinColumnTuple cols(f->get_number_of_columns(),
                                f->get_join_columns(),
                                f->get_join_column_type_infos());
           auto join_tuple_iter =
               JoinColumnTupleIterator(cols.num_cols, cols.join_column_per_key,
                                       cols.type_info_per_key, tuple_idx, 1);
           T key_scratch_buff[g_maximum_conditions_to_coalesce]; // The key
           if (join_tuple_iter != cols.end()) {
             (*f)(join_tuple_iter.join_column_iterators, key_scratch_buff,
                  key_buff_handler);
           }
         });
   }).wait();
}

template <typename T, typename KEY_HANDLER>
void count_matches_baseline(int32_t *count_buff, const T *composite_key_dict,
                            const int64_t entry_count,
                            const KEY_HANDLER *f, // On GPU
                            const int64_t num_elems) {
  sycl::queue q;
  assert(composite_key_dict);
  q.submit([&](sycl::handler &h) {
     // std::cout << q.get_device().get_info<sycl::info::device::name>() <<
     // "\n";
     h.parallel_for(
         sycl::range{static_cast<size_t>(entry_count)},
         [=](sycl::id<1> tuple_idx) {
           const size_t key_size_in_bytes =
               f->get_key_component_count() * sizeof(T);
           auto key_buff_handler =
               [composite_key_dict, entry_count, count_buff, key_size_in_bytes](
                   const int64_t row_entry_idx, const T *key_scratch_buff,
                   const size_t key_component_count) {
                 const auto matching_group =
                     get_matching_baseline_hash_slot_readonly(
                         key_scratch_buff, key_component_count,
                         composite_key_dict, entry_count, key_size_in_bytes);
                 const auto entry_idx = (matching_group - composite_key_dict) /
                                        key_component_count;
                 sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device>
                     atomic_count_buf_at_idx(count_buff[entry_idx]);
                 atomic_count_buf_at_idx.fetch_add(1);
                 return 0;
               };
           JoinColumnTuple cols(f->get_number_of_columns(),
                                f->get_join_columns(),
                                f->get_join_column_type_infos());
           T key_scratch_buff[g_maximum_conditions_to_coalesce]; // The key
           auto join_tuple_iter =
               JoinColumnTupleIterator(cols.num_cols, cols.join_column_per_key,
                                       cols.type_info_per_key, tuple_idx, 1);
           if (join_tuple_iter != cols.end()) {
             (*f)(join_tuple_iter.join_column_iterators, key_scratch_buff,
                  key_buff_handler);
           }
         });
   }).wait();
}

template <typename T>
const T *get_matching_baseline_hash_slot_readonly(
    const T *key, const size_t key_component_count, const T *composite_key_dict,
    const int64_t entry_count, const size_t key_size_in_bytes) {
  const uint32_t h = MurmurHash1Impl(key, key_size_in_bytes, 0) % entry_count;
  uint32_t off = h * key_component_count;
  if (keys_are_equal(&composite_key_dict[off], key, key_component_count)) {
    return &composite_key_dict[off];
  }
  uint32_t h_probe = (h + 1) % entry_count;
  while (h_probe != h) {
    off = h_probe * key_component_count;
    if (keys_are_equal(&composite_key_dict[off], key, key_component_count)) {
      return &composite_key_dict[off];
    }
    h_probe = (h_probe + 1) % entry_count;
  }
  assert(false);
  return nullptr;
}

template <typename T>
void init_baseline_hash_join_buff_on_l0(int8_t *hash_join_buff,
                                        const int64_t entry_count,
                                        const size_t key_component_count,
                                        const bool with_val_slot,
                                        const int32_t invalid_slot_val) {
  sycl::queue q;
  const size_t hash_entry_size =
      (key_component_count + (with_val_slot ? 1 : 0)) * sizeof(T);
  const T empty_key = get_invalid_key<T>();
  q.submit([&](sycl::handler &h) {
     h.parallel_for(
         sycl::range{static_cast<size_t>(entry_count)}, [=](sycl::id<1> idx) {
           const int64_t off = idx * hash_entry_size;
           auto row_ptr = reinterpret_cast<T *>(hash_join_buff + off);
           // std::fill(row_ptr, row_ptr+key_component_count, empty_key);
           // memset(row_ptr, empty_key, key_component_count*sizeof(int32_t));
           for (size_t k_component = 0; k_component < key_component_count;
                ++k_component) {
             row_ptr[k_component] = empty_key;
           }
           if (with_val_slot) {
             row_ptr[key_component_count] = invalid_slot_val;
           }
         });
   }).wait();
}

template <typename T>
void fill_baseline_hash_join_buff_on_l0(
    int8_t *hash_buff, const int64_t entry_count,
    const int32_t invalid_slot_val, const bool for_semi_join,
    const size_t key_component_count, const bool with_val_slot,
    int *dev_err_buff, const GenericKeyHandler *key_handler,
    const int64_t num_elems) {
  sycl::queue q(sycl::property::queue::enable_profiling{});
  const size_t key_size_in_bytes = key_component_count * sizeof(T);
  const size_t hash_entry_size =
      key_size_in_bytes + (with_val_slot * sizeof(T));
  auto key_buff_handler = [hash_buff, entry_count, with_val_slot,
                           invalid_slot_val, key_size_in_bytes, hash_entry_size,
                           for_semi_join](const int64_t entry_idx,
                                          const T *key_scratch_buffer,
                                          const size_t key_component_count) {
    if (for_semi_join) {
      return write_baseline_hash_slot_for_semi_join<T>(
          entry_idx, hash_buff, entry_count, key_scratch_buffer,
          key_component_count, with_val_slot, invalid_slot_val,
          key_size_in_bytes, hash_entry_size);
    } else {
      return write_baseline_hash_slot<T>(
          entry_idx, hash_buff, entry_count, key_scratch_buffer,
          key_component_count, with_val_slot, invalid_slot_val,
          key_size_in_bytes, hash_entry_size);
    }
  };

  sycl::event kernelEvent = q.submit([&](sycl::handler &h) {
    // std::cout << q.get_device().get_info<sycl::info::device::name>() << "\n";
    h.parallel_for(
        sycl::range{static_cast<size_t>(entry_count)},
        [=](sycl::id<1> tuple_idx) {
          JoinColumnTuple cols(key_handler->get_number_of_columns(),
                               key_handler->get_join_columns(),
                               key_handler->get_join_column_type_infos());
          T key_scratch_buff[g_maximum_conditions_to_coalesce]; // The key
          sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>
              atomic_dev_err_buff(*(dev_err_buff));
          auto join_tuple_iter =
              JoinColumnTupleIterator(cols.num_cols, cols.join_column_per_key,
                                      cols.type_info_per_key, tuple_idx, 1);
          if (join_tuple_iter != cols.end()) {
            const auto err =
                (*key_handler)(join_tuple_iter.join_column_iterators,
                               key_scratch_buff, key_buff_handler);
            if (err) {
              atomic_dev_err_buff.store(err);
            }
          }
        });
  });

  // sycl::event kernelEvent = q.single_task([=](){
  //   sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
  //   sycl::memory_scope::device> atomic_dev_err_buff(*(dev_err_buff));
  //   JoinColumnTuple cols(key_handler->get_number_of_columns(),
  //   key_handler->get_join_columns(),
  //   key_handler->get_join_column_type_infos()); T
  //   key_scratch_buff[g_maximum_conditions_to_coalesce]; // The key for (auto&
  //   it : cols.slice(0, 1)) {
  //     const auto err = (*key_handler)(it.join_column_iterators,
  //     key_scratch_buff, key_buff_handler); if (err) {
  //       atomic_dev_err_buff.store(err);
  //     }
  //   }
  // });

  kernelEvent.wait();
  //   auto start_time =
  //       kernelEvent
  //           .get_profiling_info<sycl::info::event_profiling::command_start>();
  //   auto end_time =
  //       kernelEvent
  //           .get_profiling_info<sycl::info::event_profiling::command_end>();
  //   std::cout << "fill_baseline_hash_join_buff_on_l0 time: "
  //             << (end_time - start_time) / 1e6 << " ms" << std::endl;
}

void approximate_distinct_tuples_on_l0(uint8_t *hll_buffer,
                                       int32_t *row_count_buffer,
                                       const uint32_t b,
                                       const int64_t num_elems,
                                       const GenericKeyHandler *f) {
  sycl::queue q;
  auto writer_to_hll_buff =
      [b, hll_buffer, row_count_buffer](const int64_t entry_idx,
                                        const int64_t *key_scratch_buff,
                                        const size_t key_component_count) {
        if (row_count_buffer) {
          row_count_buffer[entry_idx] += 1;
        }
        const uint64_t hash = MurmurHash64AImpl(
            key_scratch_buff, key_component_count * sizeof(int64_t), 0);
        const uint32_t index = hash >> (64 - b);
        const auto rank = get_rank(hash << b, 64 - b);
        hll_buffer[index] = std::max(hll_buffer[index], rank);
        return 0;
      };

  q.submit([&](sycl::handler &h) {
     h.parallel_for(
         sycl::range{static_cast<size_t>(num_elems)},
         [=](sycl::id<1> tuple_idx) {
           JoinColumnTuple cols(f->get_number_of_columns(),
                                f->get_join_columns(),
                                f->get_join_column_type_infos());
           int64_t
               key_scratch_buff[g_maximum_conditions_to_coalesce]; // The key
           auto join_cols_tuple_it =
               JoinColumnTupleIterator(cols.num_cols, cols.join_column_per_key,
                                       cols.type_info_per_key, tuple_idx, 1);
           if (join_cols_tuple_it != cols.end()) {
             (*f)(join_cols_tuple_it.join_column_iterators, key_scratch_buff,
                  writer_to_hll_buff);
           }
         });
   }).wait();

  // q.single_task([=](){
  //   JoinColumnTuple cols(
  //       f->get_number_of_columns(), f->get_join_columns(),
  //       f->get_join_column_type_infos());
  //   int64_t key_scratch_buff[g_maximum_conditions_to_coalesce]; // The key
  //   for (auto& it : cols.slice(0, 1)) {
  //     (*f)(it.join_column_iterators, key_scratch_buff, writer_to_hll_buff);
  //   }
  // }).wait();
}

template <typename T>
void fill_one_to_many_baseline_hash_table_on_l0(
    int32_t *buff, const T *composite_key_dict, const int64_t hash_entry_count,
    const int32_t invalid_slot_val, const GenericKeyHandler *key_handler,
    const size_t num_elems) {
  sycl::queue q;
  auto pos_buff = buff;
  auto count_buff = buff + hash_entry_count;
  q.memset(count_buff, 0, hash_entry_count * sizeof(int32_t)).wait();
  count_matches_baseline<T, GenericKeyHandler>(
      count_buff, composite_key_dict, hash_entry_count, key_handler, num_elems);
  set_valid_pos_flag(pos_buff, count_buff, hash_entry_count);
  q.single_task([=]() { // Inclusive scan
     for (size_t i = 1; i < hash_entry_count; i++) {
       count_buff[i] = count_buff[i - 1] + count_buff[i];
     }
   })
      .wait();
  set_valid_pos(pos_buff, count_buff, hash_entry_count);
  q.memset(count_buff, 0, hash_entry_count * sizeof(int32_t)).wait();
  fill_row_ids_baseline<T, GenericKeyHandler>(
      buff, composite_key_dict, hash_entry_count, invalid_slot_val, key_handler,
      num_elems);
}

template void init_baseline_hash_join_buff_on_l0<int32_t>(
    int8_t *, const int64_t, const size_t, const bool, const int32_t);
template void init_baseline_hash_join_buff_on_l0<int64_t>(
    int8_t *, const int64_t, const size_t, const bool, const int32_t);

template void fill_baseline_hash_join_buff_on_l0<int32_t>(
    int8_t *, const int64_t, const int32_t, const bool, const size_t,
    const bool, int *, const GenericKeyHandler *, const int64_t);
template void fill_baseline_hash_join_buff_on_l0<int64_t>(
    int8_t *, const int64_t, const int32_t, const bool, const size_t,
    const bool, int *, const GenericKeyHandler *, const int64_t);

template void fill_one_to_many_baseline_hash_table_on_l0<int32_t>(
    int32_t *, const int32_t *, const int64_t, const int32_t,
    const GenericKeyHandler *, const size_t);
template void fill_one_to_many_baseline_hash_table_on_l0<int64_t>(
    int32_t *, const int64_t *, const int64_t, const int32_t,
    const GenericKeyHandler *, const size_t);
