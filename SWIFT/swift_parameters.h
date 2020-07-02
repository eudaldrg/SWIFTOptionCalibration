#pragma once

#include <cstdlib>
#include <iostream>

class Distribution;

namespace Swift {

class SwiftParameters
{
public:
    SwiftParameters(size_t m, int k_1, int k_2, size_t iota_density, size_t iota_payoff);
    SwiftParameters(int n, Distribution const& distribution, double min_x0, double max_x0);

    int m_m;
    int m_two_to_the_m;
    double m_sqrt_two_to_the_m;
    int m_k1;
    double m_payoff_from;
    int m_k2;
    double m_payoff_to;
    int m_J_density;
    int m_N_density;
    int m_J_payoff;
    int m_N_payoff;
private:
    int m_iota_density; // Iota for the C_m_k calculation. Determines the number of intervals in which we subdivide the sinc integral.
    int m_iota_payoff; // Iota for the V_m_k calculation. (determines the truncation of the integration space).
};

std::ostream& operator<<(std::ostream& out, SwiftParameters const& params);

}