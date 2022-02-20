add_library(flow MODULE 
    flow.cpp
)
target_compile_features(flow PRIVATE cxx_range_for cxx_auto_type)
