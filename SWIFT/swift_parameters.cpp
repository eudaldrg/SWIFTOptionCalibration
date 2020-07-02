#include "swift_parameters.h"

#include "distributions.h"

namespace Swift {

std::ostream& operator<<(std::ostream& out, SwiftParameters const& params)
{
    return out << "Swift params" << std::endl << "m: "<< params.m_m << ", " << "two_to_the_m: "<< params.m_two_to_the_m << ", " << "sqrt_two_to_the_m: "<< params.m_sqrt_two_to_the_m << ", "
               <<"k1: "<< params.m_k1 << ", " << "payoff_from: "<< params.m_payoff_from << ", " << "k2: "<< params.m_k2 << ", " << "payoff_to: "<< params.m_payoff_to << ", "
               << "J_density: "<< params.m_J_density << ", " << "N_density: "<< params.m_N_density << ", " << "J_payoff: "<< params.m_J_payoff << ", " << "N_payoff: "<< params.m_N_payoff << std::endl;
}
SwiftParameters::SwiftParameters(size_t m, int k_1, int k_2, size_t iota_density, size_t iota_payoff) : m_m(m), m_k1(k_1), m_k2(k_2), m_iota_density(iota_density), m_iota_payoff(iota_payoff)
{
    // Iota bar is used for generating 2^(iota_payoff) frequency samples for the char_function Fourier transform.
    double log2_k_range = std::log2(m_k2 - m_k1);
    int ceil_log_2_k_range = std::lround(std::ceil(log2_k_range));
    if (m_iota_density < ceil_log_2_k_range)
    {
        std::cout << __FUNCTION__ << ": Wrong params. iota_density " << m_iota_density << " not big enough to apply FFT. Increasing to " << ceil_log_2_k_range << std::endl;
        m_iota_density = ceil_log_2_k_range;
    }

    m_two_to_the_m = std::pow(2, m_m);
    m_payoff_from = static_cast<double>(m_k1) / m_two_to_the_m;
    m_payoff_to = static_cast<double>(m_k2) / m_two_to_the_m;
    m_sqrt_two_to_the_m = std::sqrt(static_cast<double>(m_two_to_the_m));
    m_J_density = std::pow(2, m_iota_density);
    m_N_density = 2 * m_J_density;
    m_J_payoff = std::pow(2, m_iota_payoff);
    m_N_payoff = 2 * m_J_payoff;
//    std::cout << *this;
}

SwiftParameters::SwiftParameters(int n, Distribution const& distribution, double min_x0, double max_x0)
{
    double L = 10.;
    m_m = n;
    m_two_to_the_m = std::pow(2, m_m);
    m_sqrt_two_to_the_m = std::sqrt(m_two_to_the_m);

    double c = std::abs(distribution.GetFirstCumulant()) + L * std::sqrt(std::abs(distribution.GetSecondCumulant()) + std::sqrt(std::abs(distribution.GetFourthCumulant())));
    m_payoff_from = min_x0 - c;
    m_payoff_to = max_x0 + c;

    m_k1=ceil(m_two_to_the_m * m_payoff_from);
    m_k2=floor(m_two_to_the_m * m_payoff_to);

    // If the sinc function is queried in the interval [a, b], then J * pi / 2 >=

//    double min_t_value = 2^m min_x0 - max_k
//    double a = std::max(std::fabs(m_payoff_from),std::fabs(m_payoff_to));
//    double min_t_value_for_density = std::max(std::fabs(m_two_to_the_m * a - m_k1), std::fabs(m_two_to_the_m * a + m_k2));

    // J >= pi / 2 (2^m from - k1, 2^

//    m_iota_density = ceil(log2(MY_PI * min_t_value_for_density));
    m_iota_density = ceil(log2(MY_PI * std::abs(m_k2 - m_k1))) - 1;
    m_J_density = pow(2, m_iota_density-1);
    m_N_density = 2 * m_J_density;

    //Payoff
    m_iota_payoff = ceil(log2(MY_PI * std::abs(m_k2 - m_k1))) - 1;
    m_J_payoff = pow(2, m_iota_payoff - 1);
    m_N_payoff = 2 * m_J_payoff;
//    std::cout << *this;
}

}