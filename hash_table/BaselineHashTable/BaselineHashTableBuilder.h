#ifndef BASELINE_HT_BUILDER_H__
#define BASELINE_HT_BUILDER_H__

#include "../CommonDecls.h"

// Called from HDK
template <typename T>
void init_baseline_hash_join_buff_on_l0(int8_t *hash_join_buff,
                                        const int64_t entry_count,
                                        const size_t key_component_count,
                                        const bool with_val_slot,
                                        const int32_t invalid_slot_val);

// Called from HDK
void init_hash_join_buff_on_l0(int32_t *groups_buffer,
                               const int64_t hash_entry_count,
                               const int32_t invalid_slot_val);

// Called from HDK
template <typename T>
void fill_baseline_hash_join_buff_on_l0(
    int8_t *hash_buff, const int64_t entry_count,
    const int32_t invalid_slot_val, const bool for_semi_join,
    const size_t key_component_count, const bool with_val_slot,
    int *dev_err_buff, const GenericKeyHandler *key_handler,
    const int64_t num_elems);
// Called from HDK
void approximate_distinct_tuples_on_l0(uint8_t *hll_buffer,
                                       int32_t *row_count_buffer,
                                       const uint32_t b,
                                       const int64_t num_elems,
                                       const GenericKeyHandler *f);

// Called from HDK
template <typename T>
void fill_one_to_many_baseline_hash_table_on_l0(
    int32_t *buff, const T *composite_key_dict, const int64_t hash_entry_count,
    const int32_t invalid_slot_val, const GenericKeyHandler *key_handler,
    const size_t num_elems);

#endif // BASELINE_HT_BUILDER_H__