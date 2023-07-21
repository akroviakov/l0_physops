#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

constexpr size_t g_maximum_conditions_to_coalesce{8};

enum class ColumnType { SmallDate = 0, Signed = 1, Unsigned = 2, Double = 3 };

struct JoinChunk {
  const int8_t* col_buff;  // actually from AbstractBuffer::getMemoryPtr() via Chunk_NS::Chunk
  size_t num_elems;
};

struct JoinColumn {
  const int8_t* col_chunks_buff;  // actually a JoinChunk* from ColumnFetcher::makeJoinColumn(), malloced in CPU
  size_t col_chunks_buff_sz;
  size_t num_chunks;
  size_t num_elems;
  size_t elem_sz;
};

struct JoinColumnTypeInfo {
  const size_t elem_sz;
  const int64_t min_val;
  const int64_t max_val;
  const int64_t null_val;
  const bool uses_bw_eq;
  const int64_t translated_null_val;
  const ColumnType column_type;
};

struct HashEntryInfo {
  alignas(sizeof(int64_t)) size_t hash_entry_count;
  alignas(sizeof(int64_t)) int64_t bucket_normalization;

  inline size_t getNormalizedHashEntryCount() const {
    assert(bucket_normalization > 0);
    auto modulo_res =
        hash_entry_count % static_cast<size_t>(bucket_normalization);
    auto entry_count =
        hash_entry_count / static_cast<size_t>(bucket_normalization);
    if (modulo_res) {
      return entry_count + 1;
    }
    return entry_count;
  }

  bool operator!() const { return !(this->getNormalizedHashEntryCount()); }
};

enum class ExecutorDeviceType { CPU = 0, GPU };
enum class HashType : int { OneToOne, OneToMany, ManyToMany };

constexpr int StringDictionary_INVALID_STR_ID{-1};

#define EMPTY_KEY_64 std::numeric_limits<int64_t>::max()
#define EMPTY_KEY_32 std::numeric_limits<int32_t>::max()
#define EMPTY_KEY_16 std::numeric_limits<int16_t>::max()
#define EMPTY_KEY_8 std::numeric_limits<int8_t>::max()

#define NULL_BOOLEAN INT8_MIN
#define NULL_TINYINT INT8_MIN
#define NULL_SMALLINT INT16_MIN
#define NULL_INT INT32_MIN
#define NULL_BIGINT INT64_MIN
#define NULL_FLOAT FLT_MIN
#define NULL_DOUBLE DBL_MIN

#define NULL_ARRAY_BOOLEAN (INT8_MIN + 1)
#define NULL_ARRAY_TINYINT (INT8_MIN + 1)
#define NULL_ARRAY_SMALLINT (INT16_MIN + 1)
#define NULL_ARRAY_INT (INT32_MIN + 1)
#define NULL_ARRAY_BIGINT (INT64_MIN + 1)
#define NULL_ARRAY_FLOAT (FLT_MIN * 2.0)
#define NULL_ARRAY_DOUBLE (DBL_MIN * 2.0)

#define NULL_ARRAY_COMPRESSED_32 0x80000000U