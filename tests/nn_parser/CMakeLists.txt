file(GLOB_RECURSE TEST_FILES *.c *.cc *.cpp)
list(REMOVE_ITEM TEST_FILES)

if (TEST_FILES)
    add_executable(test_nn_parser ${TEST_FILES})
    target_link_libraries(test_nn_parser
        PRIVATE
            gtest_main
            gtest
            hls
    )
    gtest_discover_tests(test_nn_parser
        EXTRA_ARGS --gtest_color=yes
        DISCOVERY_TIMEOUT 30
    )
endif()
