set(hash_table_source_files
    PerfectHashTable/PerfectHashTableBuilder.cpp
    BaselineHashTable/BaselineHashTableBuilder.cpp
    Shared/Shared.cpp
)

add_dpcpp_lib(hash_table ${hash_table_source_files})
