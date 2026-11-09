add_executable(avatar_core_test cpp/core_test.cpp)
target_include_directories(avatar_core_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(avatar_core_test PRIVATE avatar)
add_test(NAME core_contract COMMAND avatar_core_test)
