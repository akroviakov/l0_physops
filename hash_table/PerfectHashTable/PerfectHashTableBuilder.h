#ifndef PERFECT_HT_BUILDER_H__
#define PERFECT_HT_BUILDER_H__

#include "../CommonDecls.h"

void init_hash_join_buff_on_l0(int32_t *groups_buffer,
                               const int64_t hash_entry_count,
                               const int32_t invalid_slot_val);

void fill_hash_join_buff_bucketized_on_l0(
    int32_t *buff, const int32_t invalid_slot_val, const bool for_semi_join,
    const JoinColumn join_column, const JoinColumnTypeInfo type_info,
    const int32_t *sd_inner_to_outer_translation_map,
    const int32_t min_inner_elem, const int64_t bucket_normalization,
    int *dev_err_buff);

void fill_one_to_many_hash_table_on_l0_bucketized(
    int32_t *buff, const HashEntryInfo hash_entry_info,
    const int32_t invalid_slot_val, const JoinColumn &join_column,
    const JoinColumnTypeInfo &type_info);

void fill_one_to_many_hash_table_on_l0(int32_t *buff,
                                       const HashEntryInfo hash_entry_info,
                                       const int32_t invalid_slot_val,
                                       const JoinColumn &join_column,
                                       const JoinColumnTypeInfo &type_info);
#endif // BASELINE_HT_BUILDER_H__