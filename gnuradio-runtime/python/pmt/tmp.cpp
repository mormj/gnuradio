#include <boost/format.hpp>
#include <gnuradio/misc.h>
#include <vector>

int foo(int x)
{
    std::vector<float> v(100);
    gr_zero_vector(v);

    boost::format("hello");
   return x+100;
}