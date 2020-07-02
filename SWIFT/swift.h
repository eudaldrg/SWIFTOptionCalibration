#pragma once

#include <functional>
#include <numeric>

#include "my_math.h"
#include "swift_parameters.h"

class Distribution;
class OptionContract;

namespace Swift {

class SwiftEvaluator
{
public:
    SwiftEvaluator(SwiftParameters const& params, Distribution const& distribution, OptionContract const& option_contract);

    [[nodiscard]] double GetPrice(double S, double K, double r, double q, bool is_call) const;
    [[nodiscard]] std::vector<double> GetGradient(double S, double K, double r, double q, bool is_call) const;

    SwiftParameters m_params;
    Distribution const& m_distribution;
    OptionContract const& m_option_contract;

};

inline double Rect(double x)
{
    double abs_x = std::abs(x);
    if (abs_x < 0.5)
        return 1;
    if (abs_x == 0.5) // TODO: A lot of double precision issues, but who cares, a single point doesn't affect the integral.
        return 0.5;
    //        if (abs_x > 0.5)
    return 0;
}

/// Expression 14 in the paper. It computes the area under f through the trapezoid's rule
inline double AreaUnderCurve(std::size_t m, std::vector<double> c_m_k)
{
    return (std::accumulate(c_m_k.begin(), c_m_k.end(), 0.0) - c_m_k.front() * 0.5 - c_m_k.back() * 0.5)
           / std::pow(2, m * 0.5);
}

inline std::function<double(double)> GetTheta(std::size_t m, int k)
{
    return [m, k](double x)
    {
        return std::pow(2, m / 2) * Sinc(std::pow(2, m) * x - k);
    };
}

inline std::function<double(double)> GetPsi(std::size_t m, int k)
{
    return [m, k](double x)
    {
        return std::pow(2, m / 2) * (std::sin(MY_PI * (std::pow(2, m) * x - k - 0.5)
                                              - std::sin(2 * MY_PI * (std::pow(2, m) * x - k - 0.5))) / (MY_PI * (std::pow(2, m) * x - k - 0.5)));
    };
}

}