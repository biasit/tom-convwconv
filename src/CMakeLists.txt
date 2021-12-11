add_library(inverter SHARED inverter.cpp)
install(TARGETS inverter DESTINATION lib)

add_library(lut_amm SHARED mithral/lut_amm.cpp)
install(TARGETS lut_amm DESTINATION lib)