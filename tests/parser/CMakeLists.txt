set(EXECUTABLE_OUTPUT_PATH ../../bin)
file(GLOB_RECURSE TEST_FILES *.c *.cc *.cpp)
list(REMOVE_ITEM TEST_FILES)

if (TEST_FILES)
    add_executable(test_parser ${TEST_FILES})
    target_link_libraries(test_parser
        PRIVATE
            gtest_main
            gtest
            vhn 
    )
    gtest_discover_tests(test_parser
        EXTRA_ARGS --gtest_color=yes
        DISCOVERY_TIMEOUT 30
    )
endif()
