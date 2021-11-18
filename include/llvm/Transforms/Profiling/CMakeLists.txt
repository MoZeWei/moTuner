add_library(Profiling MODULE
    Profiling.cpp
)
target_compile_features(TUNING PRIVATE cxx_range_for cxx_auto_type)
