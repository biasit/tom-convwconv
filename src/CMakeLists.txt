add_library(inverter SHARED inverter.cpp)
install(TARGETS inverter DESTINATION lib)

add_library(lut_amm SHARED lut_amm.hpp)
install(TARGETS lut_amm DESTINATION lib)