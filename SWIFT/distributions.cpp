#include "distributions.h"

std::ostream& operator<<(std::ostream& out, HestonParameters const& heston_parameters)
{
    return out << "k: " << heston_parameters.m_kappa << " v_bar: " << heston_parameters.m_v_bar << " sigma: " << heston_parameters.m_sigma << " rho: "
               << heston_parameters.m_rho << " v0: " << heston_parameters.m_v0;
}