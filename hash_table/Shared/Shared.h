#ifndef SAHRED_HT_H__
#define SAHRED_HT_H__

#include "../CommonDecls.h"
#include "../Types.h"

#define VALID_POS_FLAG 0

template <typename T = int64_t> inline T get_invalid_key() {
  return EMPTY_KEY_64;
}

template <> inline int32_t get_invalid_key() { return EMPTY_KEY_32; }

void set_valid_pos_flag(int32_t *pos_buff, const int32_t *count_buff,
                        const int64_t entry_count);

void set_valid_pos(int32_t *pos_buff, int32_t *count_buff,
                   const int64_t entry_count);

// Interface call
void init_hash_join_buff_on_l0(int32_t *groups_buffer,
                               const int64_t hash_entry_count,
                               const int32_t invalid_slot_val);

#endif // SAHRED_HT_H__