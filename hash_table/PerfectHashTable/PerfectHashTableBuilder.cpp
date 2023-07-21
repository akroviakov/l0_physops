#include <CL/sycl.hpp>
#include <cassert>
#include <limits>
#include <type_traits>

#include "../JoinColumnIterator.h"
#include "../Shared/Shared.h"
#include "PerfectHashTableBuilder.h"
#include "PerfectHashTableHelpers.h"

int32_t *get_hash_slot(int32_t *buff, const int64_t key,
                       const int64_t min_key) {
  return buff + (key - min_key);
}

int32_t *get_bucketized_hash_slot(int32_t *buff, const int64_t key,
                                  const int64_t min_key,
                                  const int64_t bucket_normalization) {
  return buff + (key - min_key) / bucket_normalization;
}

int fill_one_to_one_hashtable(const size_t idx, int32_t *entry_ptr,
                              const int32_t invalid_slot_val) {
  // the atomic takes the address of invalid_slot_val to write the value of
  // entry_ptr if not equal to invalid_slot_val. make a copy to avoid
  // dereferencing a const value.
  int32_t invalid_slot_val_copy = invalid_slot_val;
  sycl::atomic_ref<int32_t, sycl::memory_order::acq_rel,
                   sycl::memory_scope::device>
      atomic_entry_ptr(*entry_ptr);
  if (!atomic_entry_ptr.compare_exchange_strong(invalid_slot_val_copy,
                                                static_cast<int32_t>(idx),
                                                sycl::memory_order::acq_rel)) {
    // slot is full
    return -1;
  }
  return 0;
}

int fill_hashtable_for_semi_join(const size_t idx, int32_t *entry_ptr,
                                 const int32_t invalid_slot_val) {
  // just mark the existence of value to the corresponding hash slot
  // regardless of hashtable collision
  auto invalid_slot_val_copy = invalid_slot_val;
  sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                   sycl::memory_scope::device>
      atomic_entry_ptr(*entry_ptr);
  atomic_entry_ptr.compare_exchange_strong(invalid_slot_val_copy,
                                           static_cast<int32_t>(idx),
                                           sycl::memory_order::relaxed);
  return 0;
}

template <typename HASHTABLE_FILLING_FUNC>
void fill_hash_join_buff_impl(int32_t *buff, const int32_t invalid_slot_val,
                              const JoinColumn join_column,
                              const JoinColumnTypeInfo type_info,
                              const int32_t *sd_inner_to_outer_translation_map,
                              const int32_t min_inner_elem,
                              HASHTABLE_FILLING_FUNC filling_func,
                              int *dev_err_buff) {
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
     h.parallel_for(sycl::range{static_cast<size_t>(join_column.num_elems)},
                    [=](sycl::id<1> elem_idx) {
                      sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                       sycl::memory_scope::device>
                          atomic_dev_err(*dev_err_buff);
                      auto item = *(JoinColumnIterator(&join_column, &type_info,
                                                       elem_idx, 1));
                      const size_t index = item.index;
                      int64_t elem = item.element;
                      if (elem == type_info.null_val) {
                        if (type_info.uses_bw_eq) {
                          elem = type_info.translated_null_val;
                        } else {
                          return;
                        }
                      }
                      if (filling_func(elem, index)) {
                        atomic_dev_err.store(-1);
                      }
                    });
   }).wait();
};

template <typename SLOT_SELECTOR>
void count_matches_impl(int32_t *count_buff, const int32_t invalid_slot_val,
                        const JoinColumn join_column,
                        const JoinColumnTypeInfo type_info,
                        SLOT_SELECTOR slot_selector) {
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
     h.parallel_for(sycl::range{static_cast<size_t>(join_column.num_elems)},
                    [=](sycl::id<1> elem_idx) {
                      auto item = *(JoinColumnIterator(&join_column, &type_info,
                                                       elem_idx, 1));
                      int64_t elem = item.element;
                      if (elem == type_info.null_val) {
                        if (type_info.uses_bw_eq) {
                          elem = type_info.translated_null_val;
                        } else {
                          return;
                        }
                      }
                      int32_t *entry_ptr = slot_selector(count_buff, elem);
                      sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                       sycl::memory_scope::device>
                          atomic_slot_entry(*entry_ptr);
                      atomic_slot_entry.fetch_add(1);
                    });
   }).wait();
}

void count_matches(int32_t *count_buff, const int32_t invalid_slot_val,
                   const JoinColumn join_column,
                   const JoinColumnTypeInfo type_info) {
  auto slot_sel = [type_info](auto count_buff, auto elem) {
    return get_hash_slot(count_buff, elem, type_info.min_val);
  };
  count_matches_impl(count_buff, invalid_slot_val, join_column, type_info,
                     slot_sel);
}

template <typename SLOT_SELECTOR>
void fill_row_ids_impl(int32_t *buff, const int64_t hash_entry_count,
                       const int32_t invalid_slot_val,
                       const JoinColumn join_column,
                       const JoinColumnTypeInfo type_info,
                       SLOT_SELECTOR slot_selector) {
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
     int32_t *pos_buff = buff;
     int32_t *count_buff = buff + hash_entry_count;
     int32_t *id_buff = count_buff + hash_entry_count;
     h.parallel_for(
         sycl::range{static_cast<size_t>(join_column.num_elems)},
         [=](sycl::id<1> elem_idx) {
           auto item =
               *(JoinColumnIterator(&join_column, &type_info, elem_idx, 1));
           const size_t index = item.index;
           int64_t elem = item.element;
           if (elem == type_info.null_val) {
             if (type_info.uses_bw_eq) {
               elem = type_info.translated_null_val;
             } else {
               return;
             }
           }
           auto pos_ptr = slot_selector(pos_buff, elem);
           const auto bin_idx = pos_ptr - pos_buff;
           sycl::atomic_ref<int32_t, sycl::memory_order::acq_rel,
                            sycl::memory_scope::device>
               atomic_count_buff(*(count_buff + bin_idx));
           const auto id_buff_idx = atomic_count_buff.fetch_add(1) + *pos_ptr;
           id_buff[id_buff_idx] = static_cast<int32_t>(index);
         });
   }).wait();
}

void fill_row_ids(int32_t *buff, const int64_t hash_entry_count,
                  const int32_t invalid_slot_val, const JoinColumn join_column,
                  const JoinColumnTypeInfo type_info) {
  auto slot_sel = [type_info](auto pos_buff, auto elem) {
    return get_hash_slot(pos_buff, elem, type_info.min_val);
  };

  fill_row_ids_impl(buff, hash_entry_count, invalid_slot_val, join_column,
                    type_info, slot_sel);
}

template <typename COUNT_MATCHES_FUNCTOR, typename FILL_ROW_IDS_FUNCTOR>
void fill_one_to_many_hash_table_on_device_impl(
    int32_t *buff, const int64_t hash_entry_count,
    const int32_t invalid_slot_val, const JoinColumn &join_column,
    const JoinColumnTypeInfo &type_info,
    COUNT_MATCHES_FUNCTOR count_matches_func,
    FILL_ROW_IDS_FUNCTOR fill_row_ids_func) {
  sycl::queue q;
  int32_t *pos_buff = buff;
  int32_t *count_buff = buff + hash_entry_count;
  q.memset(count_buff, 0, hash_entry_count * sizeof(int32_t)).wait();
  count_matches_func();

  set_valid_pos_flag(pos_buff, count_buff, hash_entry_count);

  q.single_task([=]() { // Inclusive scan
     for (size_t i = 1; i < hash_entry_count; i++) {
       count_buff[i] = count_buff[i - 1] + count_buff[i];
     }
   })
      .wait();

  set_valid_pos(pos_buff, count_buff, hash_entry_count);
  q.memset(count_buff, 0, hash_entry_count * sizeof(int32_t)).wait();
  fill_row_ids_func();
}

void count_matches_bucketized(int32_t *count_buff,
                              const int32_t invalid_slot_val,
                              const JoinColumn join_column,
                              const JoinColumnTypeInfo type_info,
                              const int64_t bucket_normalization) {
  auto slot_sel = [bucket_normalization, type_info](auto count_buff,
                                                    auto elem) {
    return get_bucketized_hash_slot(count_buff, elem, type_info.min_val,
                                    bucket_normalization);
  };
  count_matches_impl(count_buff, invalid_slot_val, join_column, type_info,
                     slot_sel);
}

void fill_row_ids_bucketized(int32_t *buff, const int64_t hash_entry_count,
                             const int32_t invalid_slot_val,
                             const JoinColumn join_column,
                             const JoinColumnTypeInfo type_info,
                             const int64_t bucket_normalization) {
  auto slot_sel = [type_info, bucket_normalization](auto pos_buff, auto elem) {
    return get_bucketized_hash_slot(pos_buff, elem, type_info.min_val,
                                    bucket_normalization);
  };
  fill_row_ids_impl(buff, hash_entry_count, invalid_slot_val, join_column,
                    type_info, slot_sel);
}

void fill_hash_join_buff_bucketized_on_l0(
    int32_t *buff, const int32_t invalid_slot_val, const bool for_semi_join,
    const JoinColumn join_column, const JoinColumnTypeInfo type_info,
    const int32_t *sd_inner_to_outer_translation_map,
    const int32_t min_inner_elem, const int64_t bucket_normalization,
    int *dev_err_buff) {
  auto filling_func =
      for_semi_join ? fill_hashtable_for_semi_join : fill_one_to_one_hashtable;
  auto hashtable_filling_func = [=](auto elem, size_t index) {
    auto entry_ptr = get_bucketized_hash_slot(buff, elem, type_info.min_val,
                                              bucket_normalization);
    return for_semi_join
               ? fill_hashtable_for_semi_join(index, entry_ptr,
                                              invalid_slot_val)
               : fill_one_to_one_hashtable(index, entry_ptr, invalid_slot_val);
  };

  fill_hash_join_buff_impl(buff, invalid_slot_val, join_column, type_info,
                           sd_inner_to_outer_translation_map, min_inner_elem,
                           hashtable_filling_func, dev_err_buff);
}

void fill_one_to_many_hash_table_on_l0(int32_t *buff,
                                       const HashEntryInfo hash_entry_info,
                                       const int32_t invalid_slot_val,
                                       const JoinColumn &join_column,
                                       const JoinColumnTypeInfo &type_info) {
  auto hash_entry_count = hash_entry_info.hash_entry_count;
  auto count_matches_func = [hash_entry_count,
                             count_buff = buff + hash_entry_count,
                             invalid_slot_val, join_column, type_info] {
    count_matches(count_buff, invalid_slot_val, join_column, type_info);
  };

  auto fill_row_ids_func = [buff, hash_entry_count, invalid_slot_val,
                            join_column, type_info] {
    fill_row_ids(buff, hash_entry_count, invalid_slot_val, join_column,
                 type_info);
  };

  fill_one_to_many_hash_table_on_device_impl(
      buff, hash_entry_count, invalid_slot_val, join_column, type_info,
      count_matches_func, fill_row_ids_func);
}

void fill_one_to_many_hash_table_on_l0_bucketized(
    int32_t *buff, const HashEntryInfo hash_entry_info,
    const int32_t invalid_slot_val, const JoinColumn &join_column,
    const JoinColumnTypeInfo &type_info) {
  auto hash_entry_count = hash_entry_info.getNormalizedHashEntryCount();
  auto count_matches_func =
      [count_buff = buff + hash_entry_count, invalid_slot_val, join_column,
       type_info, bucket_normalization = hash_entry_info.bucket_normalization] {
        count_matches_bucketized(count_buff, invalid_slot_val, join_column,
                                 type_info, bucket_normalization);
      };

  auto fill_row_ids_func =
      [buff, hash_entry_count = hash_entry_info.getNormalizedHashEntryCount(),
       invalid_slot_val, join_column, type_info,
       bucket_normalization = hash_entry_info.bucket_normalization] {
        fill_row_ids_bucketized(buff, hash_entry_count, invalid_slot_val,
                                join_column, type_info, bucket_normalization);
      };

  fill_one_to_many_hash_table_on_device_impl(
      buff, hash_entry_count, invalid_slot_val, join_column, type_info,
      count_matches_func, fill_row_ids_func);
}
