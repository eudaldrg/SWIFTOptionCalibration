#include <vector>
#include <cmath>
#include "quick_callibration_swift.h"

#include "distributions.h"
#include "option_contracts.h"
#include "density_coefficients_calculators.h"
#include "payoff_coefficients_calculators.h"

namespace Swift {

SwiftInvariantData::SwiftInvariantData(SwiftParameters const& params, double T, OptionContract const& option_contract, bool is_call, double S, std::vector<double> Ks, double r, double q)
{
    FFTPayoffCalculator payoff_calculator(option_contract, params);
    std::vector<double> non_k_payoff_coefficients = payoff_calculator.GetNonKPayoffCoefs(is_call);
    m_U_tilde_j.reserve(params.m_J_density);
    m_char_position_factor_per_strike_per_j_density = std::vector<std::vector<std::complex<double>>>(Ks.size(), std::vector<std::complex<double>>{});
    for (auto& vec : m_char_position_factor_per_strike_per_j_density)
        vec.reserve(params.m_J_density);
    for (int j = 1; j <= params.m_J_density; ++j)
    {
        double w_j = (2.0 * j - 1.0) * MY_PI / params.m_N_density;
        double u_j = w_j * params.m_two_to_the_m;
        std::complex<double> sum = 0.0;
        for (int k = params.m_k1; k <= params.m_k2; ++k)
            sum += non_k_payoff_coefficients[k - params.m_k1] * std::exp(1i * w_j * static_cast<double>(k));
        m_U_tilde_j.push_back(sum * static_cast<double>(params.m_sqrt_two_to_the_m) / static_cast<double>(params.m_J_density));
        for (std::size_t K_index = 0; K_index < Ks.size(); ++K_index)
        {
            double K = Ks[K_index];
            double x = Distribution::GetXCompression(S, K, r, q, T);
            m_char_position_factor_per_strike_per_j_density[K_index].push_back(Distribution::GetCharPositionFactor(u_j, x));
        }
    }
}

//QuickCallibrationSwiftEvaluator::QuickCallibrationSwiftEvaluator(Swift::SwiftParameters const& params, Distribution const& distribution, OptionContract const& option_contract, bool is_call)
//: m_params(params), m_distribution(distribution), m_option_contract(option_contract)
//{
//    FFTPayoffCalculator payoff_calculator(m_option_contract, m_params);
//    std::vector<double> non_k_payoff_coefficients = payoff_calculator.GetNonKPayoffCoefs(is_call);
//    m_U_j.reserve(m_params.m_J_density);
//    for (int j = 1; j <= m_params.m_J_density; ++j)
//    {
//        double w_j = (2.0 * j - 1.0) * MY_PI / m_params.m_N_density;
//        std::complex<double> sum = 0.0;
//        for (int k = m_params.m_k1; k <= m_params.m_k2; ++k)
//            sum += non_k_payoff_coefficients[k - m_params.m_k1] * std::exp(1i * w_j * static_cast<double>(k));
//        m_U_j.push_back(sum * static_cast<double>(m_params.m_sqrt_two_to_the_m) / static_cast<double>(m_params.m_J_density));
//    }
//}

QuickCallibrationSwiftEvaluator::QuickCallibrationSwiftEvaluator(SwiftInvariantData const& swift_invariant_data, SwiftParameters const& params, Distribution const& distribution, OptionContract const& option_contract)
: m_params(params), m_distribution(distribution), m_option_contract(option_contract), m_swift_invariant_data(swift_invariant_data) { }

std::vector<double> QuickCallibrationSwiftEvaluator::GetPrice(double /*S*/, std::vector<double> const& Ks, double r, double /*q*/ /*, double* result*/) const
{
    double inverse_exp_interest_times_years = std::exp(-r * m_distribution.m_T);
    if (m_char_values.empty())
    {
        m_char_values.reserve(m_params.m_J_density);
        for (int j = 1; j <= m_params.m_J_density; ++j)
        {
            double w_j = (2.0 * j - 1.0) * MY_PI / m_params.m_N_density;
            double u_j = w_j * m_params.m_two_to_the_m;
            m_char_values.push_back(m_distribution.GetNonXChar(u_j));
        }
    }

    std::vector<double> result;
    result.reserve(Ks.size());
    for (std::size_t K_index = 0; K_index < Ks.size(); ++K_index)
    {
        double value = 0.0;
        double K = Ks[K_index];
        for (int j = 1; j <= m_params.m_J_density; ++j)
            value += std::real(m_char_values[j - 1] * m_swift_invariant_data.m_U_tilde_j[j - 1] * m_swift_invariant_data.m_char_position_factor_per_strike_per_j_density[K_index][j - 1]);
        result.push_back(value * inverse_exp_interest_times_years * m_option_contract.GetPayoffKComponent(K));
    }
    return result;
}

std::vector<std::vector<double>> QuickCallibrationSwiftEvaluator::GetGradient(double /*S*/, std::vector<double> const& Ks, double r, double /*q*//*, double* result*/) const
{
    double inverse_exp_interest_times_years = std::exp(-r * m_distribution.m_T);
    if (m_gradient_values.empty())
    {
        m_gradient_values.reserve(m_params.m_J_density);
        for (int j = 1; j <= m_params.m_J_density; ++j)
        {
            double w_j = (2.0 * j - 1.0) * MY_PI / m_params.m_N_density;
            double u_j = w_j * m_params.m_two_to_the_m;
            m_gradient_values.push_back(m_distribution.GetNonXCharGradient(u_j));
//            std::cout << "j " << j << " gradient " <<  m_gradient_values.back()[0] << ", "<<  m_gradient_values.back()[1] << ", "<<  m_gradient_values.back()[2] << ", "<<  m_gradient_values.back()[3] << ", "<<  m_gradient_values.back()[4] << ", " << std::endl;
//            std::cout << "j " << j << " u_j " <<  m_swift_invariant_data.m_U_tilde_j[j - 1] << ", " << std::endl;
//            std::cout << "j " << j << " K_factor ";
//            for (std::size_t K_index = 0; K_index < Ks.size(); ++K_index)
//                std::cout << ", K " << Ks[K_index] << ", factor " << m_swift_invariant_data.m_char_position_factor_per_strike_per_j_density[K_index][j - 1];
//            std::cout << std::endl;
        }
    }

    std::vector<std::vector<double>> results;
    results.reserve(Ks.size());
    for (std::size_t K_index = 0; K_index < Ks.size(); ++K_index)
    {
        std::vector<double> values(m_distribution.GetNumberOfParameters(), 0.0);
        double K = Ks[K_index];
        for (int j = 1; j <= m_params.m_J_density; ++j)
        {
            for (std::size_t param_index = 0; param_index < m_distribution.GetNumberOfParameters(); ++param_index)
                values[param_index] += std::real(m_gradient_values[j - 1][param_index] * m_swift_invariant_data.m_U_tilde_j[j - 1] * m_swift_invariant_data.m_char_position_factor_per_strike_per_j_density[K_index][j - 1]);
        }
        for (std::size_t param_index = 0; param_index < m_distribution.GetNumberOfParameters(); ++param_index)
        {
            values[param_index] *= inverse_exp_interest_times_years * m_option_contract.GetPayoffKComponent(K);
        }
        results.push_back(values);
    }
    return results;
}

}