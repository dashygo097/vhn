set(EXECUTABLE_OUTPUT_PATH ../bin)
add_executable(parser parser.cc)

target_link_libraries(parser PRIVATE
    vhn
    nlohmann_json::nlohmann_json
)

# Install the tool
install(TARGETS parser 
    RUNTIME DESTINATION bin
)

# Helper function 
function(parser_header CONFIG_JSON OUTPUT_HEADER)
    add_custom_command(
        OUTPUT ${OUTPUT_HEADER}
        COMMAND parser ${CONFIG_JSON} ${OUTPUT_HEADER}
        DEPENDS parser ${CONFIG_JSON}
        COMMENT "Generating ${OUTPUT_HEADER} from ${CONFIG_JSON}"
        VERBATIM
    )
endfunction()
