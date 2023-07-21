#ifndef GENERIC_KEY_HANDLER_H__
#define GENERIC_KEY_HANDLER_H__

#include <cstdint>
#include <cstddef>
#include <cassert>
#include "JoinColumnIterator.h"
#include "Types.h"

struct GenericKeyHandler {
  GenericKeyHandler(const size_t key_component_count,
                    const bool should_skip_entries,
                    const JoinColumn *join_column_per_key,
                    const JoinColumnTypeInfo *type_info_per_key,
                    const int32_t *const *sd_inner_to_outer_translation_maps,
                    const int32_t *sd_min_inner_elems)
      : key_component_count_(key_component_count),
        should_skip_entries_(should_skip_entries),
        join_column_per_key_(join_column_per_key),
        type_info_per_key_(type_info_per_key) {
    if (sd_inner_to_outer_translation_maps) {
      sd_inner_to_outer_translation_maps_ = sd_inner_to_outer_translation_maps;
      sd_min_inner_elems_ = sd_min_inner_elems;
    } else
    {
      sd_inner_to_outer_translation_maps_ = nullptr;
      sd_min_inner_elems_ = nullptr;
    }
  }

  template <typename T, typename KEY_BUFF_HANDLER>
  int SYCL_EXTERNAL operator()(JoinColumnIterator* join_column_iterators, T* key_scratch_buff, KEY_BUFF_HANDLER f) const {
    bool skip_entry = false;
    for (size_t key_component_index = 0; key_component_index < key_component_count_; ++key_component_index) { // All key components (cols, e.g., Group By 3 cols -> 3 components)
      const auto& join_column_iterator = join_column_iterators[key_component_index]; // Get iterator on the current column of the "key"
      int64_t elem = (*join_column_iterator).element; // Get its element
      if (should_skip_entries_ && elem == join_column_iterator.type_info->null_val && !join_column_iterator.type_info->uses_bw_eq) {
        skip_entry = true;
        break;
      }
      // Translation map pts will already be set to nullptr if invalid
      if (sd_inner_to_outer_translation_maps_) {
        const auto sd_inner_to_outer_translation_map =
            sd_inner_to_outer_translation_maps_[key_component_index];
        const auto sd_min_inner_elem = sd_min_inner_elems_[key_component_index];
        if (sd_inner_to_outer_translation_map &&
            elem != join_column_iterator.type_info->null_val) {
          const auto outer_id =
              sd_inner_to_outer_translation_map[elem - sd_min_inner_elem];
          if (outer_id == StringDictionary_INVALID_STR_ID) {
            skip_entry = true;
            break;
          }
          elem = outer_id;
        }
      }
      key_scratch_buff[key_component_index] = elem; // set key's cols
    }

    if (!skip_entry) { // If entry shouldn't be skipped (all components are non null), we call a callback that writes key to the hash slot
      return f(join_column_iterators[0].index, key_scratch_buff, key_component_count_);
    }

    return 0;
  }

  size_t get_number_of_columns() const {
    return key_component_count_;
  }

  size_t get_key_component_count() const {
    return key_component_count_;
  }

  const JoinColumn* get_join_columns() const {
    return join_column_per_key_;
  }

  const JoinColumnTypeInfo* get_join_column_type_infos() const {
    return type_info_per_key_;
  }

  const size_t key_component_count_;
  const bool should_skip_entries_;
  const JoinColumn* join_column_per_key_;
  const JoinColumnTypeInfo* type_info_per_key_;
  const int32_t* const* sd_inner_to_outer_translation_maps_;
  const int32_t* sd_min_inner_elems_;
};
#endif // GENERIC_KEY_HANDLER_H__