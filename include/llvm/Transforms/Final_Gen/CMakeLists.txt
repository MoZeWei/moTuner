add_library(GEN MODULE
    Gen.cpp
)
target_compile_features(TUNING PRIVATE cxx_range_for cxx_auto_type)

