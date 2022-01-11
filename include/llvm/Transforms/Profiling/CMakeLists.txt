add_library(Profiling MODULE
    Profiling.cpp
    Graph.cpp
)
target_compile_features(Profiling PRIVATE cxx_range_for cxx_auto_type)
