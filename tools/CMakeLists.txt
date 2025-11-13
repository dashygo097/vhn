set(EXECUTABLE_OUTPUT_PATH ../bin)
add_executable(gen_config gen_config.cc)

target_link_libraries(gen_config PRIVATE 
    vhn
    nlohmann_json::nlohmann_json
)

target_compile_definitions(gen_config PRIVATE
    # Make sure we're NOT in HLS mode
    # (though model_builder.hh is already guarded)
)

# Install the tool
install(TARGETS gen_config
    RUNTIME DESTINATION bin
)

# Helper function to generate configs
function(generate_config_header CONFIG_JSON OUTPUT_HEADER)
    add_custom_command(
        OUTPUT ${OUTPUT_HEADER}
        COMMAND gen_config ${CONFIG_JSON} ${OUTPUT_HEADER}
        DEPENDS gen_config ${CONFIG_JSON}
        COMMENT "Generating ${OUTPUT_HEADER} from ${CONFIG_JSON}"
        VERBATIM
    )
endfunction()
