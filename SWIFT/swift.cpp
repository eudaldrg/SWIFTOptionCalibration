#include "swift.h"

#include <iostream>

#include "distributions.h"
#include "option_contracts.h"
#include "density_coefficients_calculators.h"
#include "payoff_coefficients_calculators.h"

namespace Swift {

SwiftEvaluator::SwiftEvaluator(SwiftParameters const& params, Distribution const& distribution, OptionContract const& option_contract)
    : m_params(params), m_distribution(distribution), m_option_contract(option_contract) { }

double SwiftEvaluator::GetPrice(double S, double K, double r, double q, bool is_call) const
{
    double inverse_exp_interest_times_years = std::exp(-r * m_distribution.m_T);
//    if (is_call)
//    {
////////        double prev_k1 = m_params.m_k1;
////////        double prev_k2 = m_params.m_k2;
////////        m_params.m_k1 = -prev_k2;
////////        m_params.m_k2 = -prev_k1;
//        double put_price = GetPrice(S, K, r, q, !is_call);
////////        m_params.m_k1 = prev_k1;
////////        m_params.m_k2 = prev_k2;
//        return put_price + S - K * inverse_exp_interest_times_years;
//    }

    double x = Distribution::GetXCompression(S, K, r, q, m_distribution.m_T);

    FastVietaCalculator calculator(m_distribution, m_params);
//    ExplicitVietaCalculator calculator(m_distribution, m_params);
    std::vector<double> c_m = calculator.GetCMCoefs(x);

    FFTPayoffCalculator payoff_calculator(m_option_contract, m_params);
//    ExplicitPayoffCalculator payoff_calculator(m_option_contract, m_params);
    std::vector<double> non_k_payoff_coefficients = payoff_calculator.GetNonKPayoffCoefs(is_call);

    double option_price = 0.0;
    for (int k = m_params.m_k1; k <= m_params.m_k2; ++k)
        option_price += c_m[k - m_params.m_k1] * non_k_payoff_coefficients[k - m_params.m_k1];
    return option_price * inverse_exp_interest_times_years * m_option_contract.GetPayoffKComponent(K);
}

std::vector<double> SwiftEvaluator::GetGradient(double F, double K, double r, double q, bool is_call) const
{
    double x = Distribution::GetXCompression(F, K, r, q, m_distribution.m_T);

    FastVietaCalculator fast_vieta_calculator(m_distribution, m_params);
    std::vector<std::vector<double>> c_m = fast_vieta_calculator.GetGradientCMCoefs(x);


    FFTPayoffCalculator payoff_calculator(m_option_contract, m_params);
//    ExplicitPayoffCalculator payoff_calculator(m_option_contract, m_params);
    std::vector<double> non_k_payoff_coefficients = payoff_calculator.GetNonKPayoffCoefs(is_call);

    std::vector<double> option_gradient(m_distribution.GetNumberOfParameters(), 0.0);
    for (int k = m_params.m_k1; k <= m_params.m_k2; ++k)
    {
        for (std::size_t param = 0; param < m_distribution.GetNumberOfParameters(); ++param)
            option_gradient[param] += c_m[k - m_params.m_k1][param] * non_k_payoff_coefficients[k - m_params.m_k1];
    }
    for (std::size_t param = 0; param < m_distribution.GetNumberOfParameters(); ++param)
        option_gradient[param] *= std::exp(-r * m_distribution.m_T) * m_option_contract.GetPayoffKComponent(K); // TODO: Add q or remove altogether.
    return option_gradient;
}

}