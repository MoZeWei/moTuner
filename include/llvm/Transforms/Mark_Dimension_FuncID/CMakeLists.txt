add_library(MDFID MODULE 
    MDFID.cpp
)
target_compile_features(MDFID PRIVATE cxx_range_for cxx_auto_type)
