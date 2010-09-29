add_library (onbnn SHARED onbnn.cpp aux-functions.cpp 
  lshkit-onbnn/char_bit_cnt.cpp lshkit-onbnn/debug.cpp 
  lshkit-onbnn/mplsh.cpp lshkit-onbnn/mplsh-fit-tune.cpp 
  lshkit-onbnn/mplsh-model.cpp lshkit-onbnn/mplsh-custom.cpp)

include_directories (lshkit-onbnn lshkit-onbnn/lshkit)
target_link_libraries (onbnn glpk)
target_link_libraries (onbnn gsl)
target_link_libraries (onbnn gslcblas)

install (TARGETS onbnn DESTINATION lib)
install (FILES *.h lshkit/*.h DESTINATION include)
