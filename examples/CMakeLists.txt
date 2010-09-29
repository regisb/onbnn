include_directories ("${PROJECT_SOURCE_DIR}/src")

add_executable (example1 example1-binary-classification.cpp)
target_link_libraries (example1 onbnn)
target_link_libraries (example1 boost_filesystem)
