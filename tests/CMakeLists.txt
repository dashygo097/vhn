print_info("―――――――――――――――――TEST SCOPE――――――――――――――――――" "0")

add_subdirectory(sanity)
add_subdirectory(nn_parser)

# stastics
file(GLOB_RECURSE TEST_FILES *.c *.cc *.cpp)
make_paths_relative(REL_TEST_FILES TEST_FILES)
make_preview_string(REL_TEST_FILES 3)
print_info("[DEBUG] Test files: ${PREVIOUS_SCOPE_VAR}" "93")
