# Since templatedInverse is a templated function in c++, we can't just import
# it directly into python (its code isn't fully defined until we
# decide what template arguments it takes). However, in inverter_wrapper.i
# we called:
#     %template(templatedInverse_d) templatedInverse<double>;
# which creates an explicit instantiation of templatedInverse for double-
# precision values. So we can just import that instantiation into python
# and use it instead.
# from .inverter_wrapper import templatedInverse_d as templatedInverse

# inverter_wrapper.Inverter is the class generated by SWIG to wrap our c++ Inverter class
from .inverter_wrapper import Inverter