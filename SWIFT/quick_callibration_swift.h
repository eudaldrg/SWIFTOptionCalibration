#pragma once

#include <complex>
#include "swift_parameters.h"

class Distribution;
class OptionContract;

namespace Swift {


class SwiftInvariantData
{
public:
    SwiftInvariantData(SwiftParameters const& params, double T, OptionContract const& option_contract, bool is_call, double S, std::vector<double> Ks, double r, double q);
    std::vector<std::complex<double>> m_U_tilde_j;
    std::vector<std::vector<std::complex<double>>> m_char_position_factor_per_strike_per_j_density;
};

class QuickCallibrationSwiftEvaluator
{
public:
//    QuickCallibrationSwiftEvaluator(SwiftInvariantData const& m_swift_invariant_data, SwiftParameters const& params, Distribution const& distribution, OptionContract const& option_contract, bool is_call);
    QuickCallibrationSwiftEvaluator(SwiftInvariantData const& swift_invariant_data, SwiftParameters const& params, Distribution const& distribution, OptionContract const& option_contract);

    [[nodiscard]] std::vector<double> GetPrice(double S, std::vector<double> const& K, double r, double q) const;
    [[nodiscard]] std::vector<std::vector<double>> GetGradient(double S, std::vector<double> const& K, double r, double q) const;

    mutable SwiftParameters m_params;
    Distribution const& m_distribution;
    OptionContract const& m_option_contract;
    SwiftInvariantData const& m_swift_invariant_data;
    mutable std::vector<std::complex<double>> m_char_values;
    mutable std::vector<std::vector<std::complex<double>>> m_gradient_values;
//    std::vector<std::complex<double>> m_U_j;
};

}
