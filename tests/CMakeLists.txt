print_info("―――――――――――――――――TEST SCOPE――――――――――――――――――" "0")
set(EXECUTABLE_OUTPUT_PATH ../outputs/tests)

file(GLOB_RECURSE SANITY_TEST_FILES *.c *.cc *.cpp)
list(REMOVE_ITEM SANITY_TEST_FILES)

if (SANITY_TEST_FILES)
    add_executable(test_sanity ${SANITY_TEST_FILES})
    target_link_libraries(test_sanity
        PRIVATE
            gtest_main
            gtest
            hls
    )
    gtest_discover_tests(test_sanity
        EXTRA_ARGS --gtest_color=yes
        DISCOVERY_TIMEOUT 30
    )
endif()

# stastics

make_paths_relative(REL_TEST_FILES TEST_FILES)
make_preview_string(REL_TEST_FILES 3)
print_info("[DEBUG] Test files: ${PREVIOUS_SCOPE_VAR}" "93")
