#include "density_coefficients_calculators.h"

#include "SWIFT/distributions.h"
#include "FFTW3/include_fftw3.h"
#include "SWIFT/my_math.h"
#include "SWIFT/swift.h"

namespace Swift {

CMCalculator::CMCalculator(Distribution const& distribution, SwiftParameters const& params) : m_distribution(distribution), m_params(params)
{ }

ParsevalCalculator::ParsevalCalculator(Distribution const& distribution, SwiftParameters const& params, std::size_t integral_buckets)
    : CMCalculator(distribution, params), m_integral_buckets(integral_buckets)
{ }



std::vector<std::vector<double>> ParsevalCalculator::GetGradientCMCoefs(double /*x*/) const
{
    throw std::runtime_error("Not implemented.");
}

std::vector<double> ParsevalCalculator::GetCMCoefs(double x) const
{
    std::vector<double> c_m;
    c_m.reserve(m_params.m_k2 - m_params.m_k1);
    for (int k = m_params.m_k1; k <= m_params.m_k2; ++k)
    {
        auto f = [this, x, k](double t)
        {
            double u = std::pow(2.0, m_params.m_m + 1) * MY_PI * t;
            return (m_distribution.GetChar(u, x) * std::exp(1i * 2.0 * MY_PI * static_cast<double>(k) * t)).real();
        };
        c_m.push_back(TrapezoidInt(-0.5, 0.5, f, m_integral_buckets) * std::pow(2.0, m_params.m_m / 2.0));
    }
    return c_m;
}

std::vector<std::vector<double>> ExplicitVietaCalculator::GetGradientCMCoefs(double x) const
{
    const std::size_t N = 2 * m_params.m_J_density;
    const std::size_t two_to_the_m = 1UL << m_params.m_m;

    std::vector<std::vector<double>> c_m;
    c_m.reserve(m_params.m_k2 - m_params.m_k1);
    for (int k = m_params.m_k1; k <= m_params.m_k2; ++k)
    {
        std::vector<double> c_m_k(m_distribution.GetNumberOfParameters(), 0);
        for (int i = 1; i <= m_params.m_J_density; ++i)
        {
            double u = (2 * i - 1) * MY_PI * two_to_the_m / N;

            std::vector<std::complex<double>> f_hat = m_distribution.GetCharGradient(u, x);
            for (std::size_t param = 0; param < m_distribution.GetNumberOfParameters(); ++param)
            {
                f_hat[param] *= std::exp(1i * static_cast<double>(k) * MY_PI * (2.0 * i - 1.0) / static_cast<double>(N));
                c_m_k[param] += f_hat[param].real();
            }
        }
        for (std::size_t param = 0; param < m_distribution.GetNumberOfParameters(); ++param)
            c_m_k[param] *= std::sqrt(two_to_the_m) / m_params.m_J_density;
        c_m.push_back(c_m_k);
    }
    return c_m;
}

std::vector<double> ExplicitVietaCalculator::GetCMCoefs(double x) const
{
    std::vector<double> c_m;
    c_m.reserve(m_params.m_k2 - m_params.m_k1 + 1);
    for (int k = m_params.m_k1; k <= m_params.m_k2; ++k)
    {
        double c_m_k = 0;
        for (int j = 1; j <= m_params.m_J_density; ++j)
        {
            double w_j = (2 * j - 1) * MY_PI / m_params.m_N_density;
            double u_j = w_j * m_params.m_two_to_the_m;
            std::complex<double> f_hat = m_distribution.GetChar(u_j, x);
            std::complex<double> cv = f_hat * std::exp(1i * static_cast<double>(k) * w_j);
            c_m_k += cv.real();
        }
        c_m.push_back(c_m_k * m_params.m_sqrt_two_to_the_m / m_params.m_J_density);
    }
    return c_m;
}

ExplicitVietaCalculator::ExplicitVietaCalculator(Distribution const& distribution, SwiftParameters const& params) : CMCalculator(distribution, params)
{ }

std::vector<std::vector<double>> FastVietaCalculator::GetGradientCMCoefs(double x) const
{
    std::vector<std::vector<double>> c_m;
    std::vector<std::vector<std::complex<double>>> frequencies(m_distribution.GetNumberOfParameters(), std::vector<std::complex<double>>{});

    for (int j = 0; j < m_params.m_N_density; ++j)
    {
        std::vector<std::complex<double>> values_to_push = j < m_params.m_J_density ?
                                                           m_distribution.GetCharGradient((2.0 * j + 1.0) * MY_PI * m_params.m_two_to_the_m / static_cast<double>(m_params.m_N_density), x)
                                                                                    :
                                                           std::vector<std::complex<double>>(m_distribution.GetNumberOfParameters(), {0.0, 0.0});

        for (std::size_t param = 0; param < m_distribution.GetNumberOfParameters(); ++param)
            frequencies[param].push_back(values_to_push[param]);
    }
    std::vector<std::vector<std::complex<double>>> times;
    for (std::size_t param = 0; param < m_distribution.GetNumberOfParameters(); ++param)
        times.push_back(MY_IDFT(frequencies[param], false));

    for (int k = m_params.m_k1; k <= m_params.m_k2; ++k)
    {
        std::size_t k_mod_N = Mod(k, static_cast<int>(m_params.m_N_density));
        std::vector<double> c_m_k(m_distribution.GetNumberOfParameters(), 0);
        c_m.emplace_back(std::vector<double>{});
        for (std::size_t param = 0; param < m_distribution.GetNumberOfParameters(); ++param)
        {
            std::complex<double> c_m_k_complex_part_dft = std::exp(1i * static_cast<double>(k) * MY_PI / static_cast<double>(m_params.m_N_density)) * times[param][k_mod_N];
            c_m.back().push_back(c_m_k_complex_part_dft.real() * m_params.m_sqrt_two_to_the_m / m_params.m_J_density);
        }
    }
    return c_m;
}

std::vector<double> FastVietaCalculator::GetCMCoefs(double x) const
{
    std::vector<double> c_m;
    c_m.reserve(m_params.m_k2 - m_params.m_k1 + 1);
    std::vector<std::complex<double>> frequencies;

    for (int i = 0; i < m_params.m_N_density; ++i)
    {
        if (i < m_params.m_J_density)
        {
            double u = (2.0 * i + 1.0) * MY_PI * static_cast<double>(m_params.m_two_to_the_m) / static_cast<double>(m_params.m_N_density);
            frequencies.push_back(m_distribution.GetChar(u, x));
        }
        else
            frequencies.emplace_back(0, 0);
    }
    std::vector<std::complex<double>> times = MY_IDFT(frequencies, true);

    for (int k = m_params.m_k1; k <= m_params.m_k2; ++k)
    {
        std::size_t k_mod_N = Mod(k, m_params.m_N_density);
        std::complex<double> idf_part = times[k_mod_N];
        std::complex<double> c_m_k_complex_part_dft = std::exp(1i * static_cast<double>(k) * MY_PI / static_cast<double>(m_params.m_N_density)) * idf_part;
        c_m.push_back(c_m_k_complex_part_dft.real() * m_params.m_sqrt_two_to_the_m / m_params.m_J_density);
    }
    return c_m;
}

FastVietaCalculator::FastVietaCalculator(Distribution const& distribution, SwiftParameters const& params) : CMCalculator(distribution, params)
{ }

NewPaperExplicitCalculator::NewPaperExplicitCalculator(Distribution const& distribution, SwiftParameters const& params) : CMCalculator(distribution, params)
{ }

std::vector<double> NewPaperExplicitCalculator::GetCMCoefs(double x) const
{
    std::vector<double> c_m;
    c_m.reserve(m_params.m_k2 - m_params.m_k1);
    for (int k = m_params.m_k1; k <= m_params.m_k2; ++k)
    {
        double c_m_k = 0;
        for (int j = 1; j <= m_params.m_J_density; ++j)
        {
            double w_j = (2.0 * j - 1.0) * MY_PI / m_params.m_N_density;
            double u = w_j * m_params.m_two_to_the_m;
            std::complex<double> factor = std::exp(1i * static_cast<double>(k) * w_j);

            std::complex<double> f_hat = m_distribution.GetChar(u, x);
            std::complex<double> cv = f_hat * factor;
            c_m_k += cv.real();
        }
        c_m.push_back(c_m_k * m_params.m_sqrt_two_to_the_m / m_params.m_J_density);
    }
    return c_m;
}

std::vector<std::vector<double>> NewPaperExplicitCalculator::GetGradientCMCoefs(double /*x*/) const
{
    throw std::runtime_error("Not Implemented");
}

}