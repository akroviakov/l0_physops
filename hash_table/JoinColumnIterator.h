/*
 * Copyright 2019 OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef JOIN_COL_ITER_H__
#define JOIN_COL_ITER_H__

#include "Types.h"

inline int64_t fixed_width_int_decode(const int8_t *byte_stream,
                                      const int32_t byte_width,
                                      const int64_t pos) {
  switch (byte_width) {
    case 1:
      return static_cast<int64_t>(byte_stream[pos * byte_width]);
    case 2:
      return *(reinterpret_cast<const int16_t*>(
          &byte_stream[pos * byte_width]));
    case 4:
      return *(reinterpret_cast<const int32_t*>(
          &byte_stream[pos * byte_width]));
    case 8:
      return *(reinterpret_cast<const int64_t*>(
          &byte_stream[pos * byte_width]));
    default:
      return std::numeric_limits<int64_t>::min() + 1;
  }
}

inline int64_t fixed_width_small_date_decode(const int8_t *byte_stream,
                                             const int32_t byte_width,
                                             const int32_t null_val,
                                             const int64_t ret_null_val,
                                             const int64_t pos) {
  auto val = fixed_width_int_decode(byte_stream, byte_width, pos);
  return val == null_val ? ret_null_val : val * 86400;
}

inline int64_t fixed_width_unsigned_decode(const int8_t *byte_stream,
                                           const int32_t byte_width,
                                           const int64_t pos) {
  switch (byte_width) {
    case 1:
      return reinterpret_cast<const uint8_t*>(
          byte_stream)[pos * byte_width];
    case 2:
      return *(reinterpret_cast<const uint16_t*>(
          &byte_stream[pos * byte_width]));
    case 4:
      return *(reinterpret_cast<const uint32_t*>(
          &byte_stream[pos * byte_width]));
    case 8:
      return *(reinterpret_cast<const uint64_t*>(
          &byte_stream[pos * byte_width]));
    default:
      return std::numeric_limits<int64_t>::min() + 1;
  }
}

inline double fixed_width_double_decode(const int8_t *byte_stream,
                                        const int64_t pos) {
  return *(reinterpret_cast<const double*>(&byte_stream[pos * sizeof(double)]));
}

//! Iterates over the rows of a JoinColumn across multiple fragments/chunks.
struct JoinColumnIterator {
  const JoinColumn* join_column;        // WARNING: pointer might be on GPU
  const JoinColumnTypeInfo* type_info;  // WARNING: pointer might be on GPU
  const struct JoinChunk* join_chunk_array;
  const int8_t* chunk_data;  // bool(chunk_data) tells if this iterator is valid
  size_t index_of_chunk;
  size_t index_inside_chunk;
  size_t index;
  size_t start;
  size_t step;

  operator bool() const { return chunk_data; }

  const int8_t* ptr() const {
    return &chunk_data[index_inside_chunk * join_column->elem_sz];
  }

  int64_t getElementSwitch() const {
    switch (type_info->column_type) {
    case ColumnType::SmallDate:
      return fixed_width_small_date_decode(
          chunk_data, type_info->elem_sz,
          type_info->elem_sz == 4 ? NULL_INT : NULL_SMALLINT,
          type_info->elem_sz == 4 ? NULL_INT : NULL_SMALLINT,
          index_inside_chunk);
    case ColumnType::Signed:
      return fixed_width_int_decode(chunk_data, type_info->elem_sz,
                                    index_inside_chunk);
    case ColumnType::Unsigned:
      return fixed_width_unsigned_decode(chunk_data, type_info->elem_sz,
                                         index_inside_chunk);
    case ColumnType::Double:
      return fixed_width_double_decode(chunk_data, index_inside_chunk);
    default:
      assert(0);
      return 0;
    }
  }

  struct IndexedElement {
    size_t index;
    int64_t element;
  };  // struct IndexedElement

  IndexedElement operator*() const {
    return {index, getElementSwitch()};
  }

  JoinColumnIterator& operator++() {
    index += step;              // make step in the global index
    index_inside_chunk += step; // make step in the local index
    // So if we make too big step, we just go from current chunk, to the last
    // one
    while (chunk_data &&
           index_inside_chunk >=
               join_chunk_array[index_of_chunk]
                   .num_elems) { // while valid and local index is overflown
      index_inside_chunk -=
          join_chunk_array[index_of_chunk]
              .num_elems; // reduce local index by current chunk's num_elems
      ++index_of_chunk;   // go to next chunk
      if (index_of_chunk <
          join_column
              ->num_chunks) { // if we are still in the range of col's chunks
        chunk_data =
            join_chunk_array[index_of_chunk].col_buff; // update chunk ptr
      } else {
        chunk_data = nullptr; // set to invalid
      }
    }
    return *this;
  }

  JoinColumnIterator() : chunk_data(nullptr) {}

  JoinColumnIterator(
      const JoinColumn* join_column,        // WARNING: pointer might be on GPU
      const JoinColumnTypeInfo* type_info,  // WARNING: pointer might be on GPU
      size_t start,
      size_t step)
      : join_column(join_column)
      , type_info(type_info)
      , join_chunk_array(reinterpret_cast<const struct JoinChunk*>(join_column->col_chunks_buff))
      , chunk_data(join_column->num_elems > 0 ? join_chunk_array->col_buff : nullptr)
      , index_of_chunk(0)
      , index_inside_chunk(0)
      , index(0)
      , start(start)
      , step(step) {
    // Stagger the index differently for each thread iterating over the column.
    auto temp = this->step;
    this->step = this->start;
    operator++();
    this->step = temp;
  }
};  // struct JoinColumnIterator

//! Helper class for viewing a JoinColumn and it's matching JoinColumnTypeInfo as a single
//! object.
struct JoinColumnTyped {
  // NOTE(sy): Someday we might want to merge JoinColumnTypeInfo into JoinColumn but
  // this class is a good enough solution for now until we have time to do more cleanup.
  const struct JoinColumn* join_column;
  const struct JoinColumnTypeInfo* type_info;

  JoinColumnIterator begin() {
    return JoinColumnIterator(join_column, type_info, 0, 1);
  }

  JoinColumnIterator end() {
    return JoinColumnIterator(); /* chunk_data = nullptr;  //set to invalid*/
  }

  struct Slice {
    JoinColumnTyped* join_column_typed;
    size_t start;
    size_t step;

    JoinColumnIterator begin() {
      return JoinColumnIterator(join_column_typed->join_column, join_column_typed->type_info, start, step);
    }

    JoinColumnIterator end() {
      return JoinColumnIterator(); /* chunk_data = nullptr;  //set to invalid*/
    }
  };  // struct Slice

  Slice slice(size_t start, size_t step) { return Slice{this, start, step}; }

};  // struct JoinColumnTyped

//! Iterates over the rows of a JoinColumnTuple across multiple fragments/chunks.
struct JoinColumnTupleIterator {
  // NOTE(sy): Someday we'd prefer to JIT compile this iterator, producing faster,
  // custom, code for each combination of column types encountered at runtime.

  size_t num_cols;
  JoinColumnIterator join_column_iterators[g_maximum_conditions_to_coalesce];

  // NOTE(sy): Are these multiple iterator instances (one per column) required when
  // we are always pointing to the same row in all N columns? Yes they are required,
  // if the chunk sizes can be different from column to column. I don't know if they
  // can or can't, so this code plays it safe for now.

  JoinColumnTupleIterator() : num_cols(0) {}

  JoinColumnTupleIterator(size_t num_cols,
                                 const JoinColumn* join_column_per_key,
                                 const JoinColumnTypeInfo* type_info_per_key,
                                 size_t start,
                                 size_t step)
      : num_cols(num_cols) {
    assert(num_cols <= g_maximum_conditions_to_coalesce);
    for (size_t i = 0; i < num_cols; ++i) {
      join_column_iterators[i] =
          JoinColumnIterator(&join_column_per_key[i],
                             type_info_per_key ? &type_info_per_key[i] : nullptr,
                             start,
                             step);
    }
  }

  operator bool() const {
    for (size_t i = 0; i < num_cols; ++i) {
      if (join_column_iterators[i]) {
        return true;
        // If any column iterator is still valid, then the tuple is still valid.
      }
    }
    return false;
  }

  JoinColumnTupleIterator& operator++() {
    for (size_t i = 0; i < num_cols; ++i) {
      ++join_column_iterators[i];
    }
    return *this;
  }

  JoinColumnTupleIterator& operator*() {
    return *this;
  }
};  // struct JoinColumnTupleIterator

//! Helper class for viewing multiple JoinColumns and their matching JoinColumnTypeInfos
//! as a single object.
struct JoinColumnTuple {
  size_t num_cols;
  const JoinColumn* join_column_per_key;
  const JoinColumnTypeInfo* type_info_per_key;

  JoinColumnTuple()
      : num_cols(0), join_column_per_key(nullptr), type_info_per_key(nullptr) {}

  JoinColumnTuple(size_t num_cols,
                         const JoinColumn* join_column_per_key,
                         const JoinColumnTypeInfo* type_info_per_key)
      : num_cols(num_cols)
      , join_column_per_key(join_column_per_key)
      , type_info_per_key(type_info_per_key) {}

  JoinColumnTupleIterator begin() {
    return JoinColumnTupleIterator(
        num_cols, join_column_per_key, type_info_per_key, 0, 1);
  }

  JoinColumnTupleIterator end() { return JoinColumnTupleIterator(); }

  struct Slice {
    const JoinColumnTuple* join_column_tuple;
    size_t start;
    size_t step;

    JoinColumnTupleIterator begin() {
      return JoinColumnTupleIterator(join_column_tuple->num_cols,
                                     join_column_tuple->join_column_per_key,
                                     join_column_tuple->type_info_per_key,
                                     start,
                                     step);
    }

    JoinColumnTupleIterator end() { return JoinColumnTupleIterator(); }

  };  // struct Slice

  std::pair<size_t, size_t> get_shape() const {
    size_t num_rows{0};
    for (size_t i = 0; i < num_cols; ++i) {
      num_rows = std::max(num_rows, join_column_per_key[i].num_elems);
    }
    return {num_rows, num_cols};
  }
  Slice slice(size_t start, size_t step) const { return Slice{this, start, step}; }

};  // struct JoinColumnTuple

#endif // JOIN_COL_ITER_H__
