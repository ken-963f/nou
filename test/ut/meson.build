# Network test
test(
  'ut_network',
  executable(
    'ut_network', 'network.cc', include_directories: inc_dir, dependencies: [test_dep, tbb_dep]
  )
)

# Layer test
test(
  'ut_input_layer',
  executable(
    'ut_input_layer', 'layer/input_layer.cc', include_directories: inc_dir, dependencies: test_dep
  )
)

# Utility test
test(
  'ut_algorithm',
  executable(
    'ut_algorithm', 'utility/algorithm.cc', include_directories: inc_dir, dependencies: [test_dep, tbb_dep]
  )
)

test(
  'ut_connect_layers',
  executable(
    'ut_connect_layers', 'utility/connect_layers.cc', include_directories: inc_dir, dependencies: test_dep
  )
)

