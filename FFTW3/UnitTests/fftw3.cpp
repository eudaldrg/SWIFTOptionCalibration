//#define BOOST_TEST_MODULE Suite_example
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <cstdlib>
#include "FFTW3/include_fftw3.h"
#include "SWIFT/my_math.h"

BOOST_AUTO_TEST_SUITE(FFTW3UT)

BOOST_AUTO_TEST_CASE(MainTest)
{
    std::cout << "Begin TESTFFTW3" << std::endl;
    fftw_complex* in, * out;
    fftw_plan p;
    std::size_t N = 4;
    in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    std::vector<std::complex<double>> x = { {1,0}, {2, -1}, {0, -1}, {-1, 2} };
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        in[i][0] = x[i].real();
        in[i][1] = x[i].imag();
    }
    fftw_execute(p); /* repeat as needed */
    std::vector<std::complex<double>> X = { {2,0}, {-2, -2}, {0, -2}, {4, 4} };
    std::cout << "Expected result";
    for (std::complex<double> val : X)
        std::cout << " " << val;
    std::cout << std::endl;
    std::cout << "Obtained result";
    for (std::size_t i = 0; i < N; ++i)
        std::cout << " (" << out[i][0] << "," << out[i][1] << ")";
    std::cout << std::endl;
    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);
}

BOOST_AUTO_TEST_CASE(CompareExactTest)
{
    std::cout << "Begin TESTFFTW3" << std::endl;
    fftw_complex* in, * out;
    fftw_plan p;
    std::size_t N = 4;
    in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    std::vector<std::complex<double>> x = { {1,0}, {2, -1}, {0, -1}, {-1, 2} };
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        in[i][0] = x[i].real();
        in[i][1] = x[i].imag();
    }
    fftw_execute(p); /* repeat as needed */
    std::vector<std::complex<double>> X = DFT(x);

    for (std::size_t i = 0; i < N; ++i)
    {
        std::complex<double> out_complex(out[i][0], out[i][1]);
        BOOST_CHECK_CLOSE(std::abs(X[i]), std::abs(out_complex), 1e-10);
        BOOST_CHECK_CLOSE(std::arg(X[i]), std::arg(out_complex), 1e-10);
    }
    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);
}

BOOST_AUTO_TEST_SUITE_END()