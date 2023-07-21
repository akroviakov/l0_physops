#ifndef BASELINE_HT_HELPER_H__
#define BASELINE_HT_HELPER_H__
#include "../CommonDecls.h"

template <typename T, typename KEY_HANDLER>
void count_matches_baseline(int32_t *count_buff, const T *composite_key_dict,
                            const int64_t entry_count, const KEY_HANDLER *f,
                            const int64_t num_elems);

template <typename T, typename KEY_HANDLER>
void fill_row_ids_baseline(int32_t *buff, const T *composite_key_dict,
                           const int64_t hash_entry_count,
                           const int32_t invalid_slot_val, const KEY_HANDLER *f,
                           const int64_t num_elems);

template <typename T>
const T *get_matching_baseline_hash_slot_readonly(
    const T *key, const size_t key_component_count, const T *composite_key_dict,
    const int64_t entry_count, const size_t key_size_in_bytes);

#endif // BASELINE_HT_HELPER_H__