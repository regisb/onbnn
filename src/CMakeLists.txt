add_library (onbnn onbnn.cpp aux_functions.cpp)
target_link_libraries (onbnn glpk)
install (TARGETS onbnn DESTINATION lib)
install (FILES onbnn.h DESTINATION include)
