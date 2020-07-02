#define BOOST_TEST_MODULE FFTW3Tests

#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_parameters.hpp>

using namespace boost::unit_test;

struct Init {
    Init()
    {
        boost::unit_test::unit_test_log.set_threshold_level( boost::unit_test::log_messages);
    }
    ~Init()  = default;
};

BOOST_GLOBAL_FIXTURE( Init );
