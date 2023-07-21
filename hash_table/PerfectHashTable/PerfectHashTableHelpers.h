#ifndef PERFECT_HT_HELPER_H__
#define PERFECT_HT_HELPER_H__
#include "../CommonDecls.h"

template <typename HASHTABLE_FILLING_FUNC>
void fill_hash_join_buff_impl(int32_t *buff, const int32_t invalid_slot_val,
                              const JoinColumn join_column,
                              const JoinColumnTypeInfo type_info,
                              const int32_t *sd_inner_to_outer_translation_map,
                              const int32_t min_inner_elem,
                              HASHTABLE_FILLING_FUNC filling_func,
                              int *dev_err_buff);

template <typename SLOT_SELECTOR>
void count_matches_impl(int32_t *count_buff, const int32_t invalid_slot_val,
                        const JoinColumn join_column,
                        const JoinColumnTypeInfo type_info,
                        SLOT_SELECTOR slot_selector);

int fill_one_to_one_hashtable(size_t idx, int32_t *entry_ptr,
                              const int32_t invalid_slot_val);

int fill_hashtable_for_semi_join(size_t idx, int32_t *entry_ptr,
                                 const int32_t invalid_slot_val);

void count_matches(int32_t *count_buff, const int32_t invalid_slot_val,
                   const JoinColumn join_column,
                   const JoinColumnTypeInfo type_info);

template <typename SLOT_SELECTOR>
void fill_row_ids_impl(int32_t *buff, const int64_t hash_entry_count,
                       const int32_t invalid_slot_val,
                       const JoinColumn join_column,
                       const JoinColumnTypeInfo type_info,
                       SLOT_SELECTOR slot_selector);

void fill_row_ids(int32_t *buff, const int64_t hash_entry_count,
                  const int32_t invalid_slot_val, const JoinColumn join_column,
                  const JoinColumnTypeInfo type_info);

template <typename COUNT_MATCHES_FUNCTOR, typename FILL_ROW_IDS_FUNCTOR>
void fill_one_to_many_hash_table_on_device_impl(
    int32_t *buff, const int64_t hash_entry_count,
    const int32_t invalid_slot_val, const JoinColumn &join_column,
    const JoinColumnTypeInfo &type_info,
    COUNT_MATCHES_FUNCTOR count_matches_func,
    FILL_ROW_IDS_FUNCTOR fill_row_ids_func);
void count_matches_bucketized(int32_t *count_buff,
                              const int32_t invalid_slot_val,
                              const JoinColumn join_column,
                              const JoinColumnTypeInfo type_info,
                              const int64_t bucket_normalization);

void fill_row_ids_bucketized(int32_t *buff, const int64_t hash_entry_count,
                             const int32_t invalid_slot_val,
                             const JoinColumn join_column,
                             const JoinColumnTypeInfo type_info,
                             const int64_t bucket_normalization);
#endif // PERFECT_HT_HELPER_H__
