#include "option_contracts.h"
#include "swift.h"

double EuropeanOptionContract::GetPayoffNonKComponent(int k, Swift::SwiftParameters const& params, bool is_call) const
{
    return GetPayoffNonKComponentNewPaper(k, params, is_call);
}

//
double EuropeanOptionContract::GetPayoffNonKComponentOldPaper(int k, Swift::SwiftParameters const& params, bool is_call) const
{
    double from = GetBoundedFrom(params.m_payoff_from, is_call);
    double to = GetBoundedTo(params.m_payoff_to, is_call);
    if (to <= from)
        return 0.0;

    auto Integral = [is_call, k](double y, double u, double w_j)
    {
        double u2 = u * u;
        double exp_y = std::exp(y);
        double multiplier_if_call = is_call ? 1.0 : -1.0;
        double B = w_j * k;
        double B_minus_u_times_y = B - u * y;
        double cosine = std::cos(B_minus_u_times_y);
        double sine = std::sin(B_minus_u_times_y);
        // (e^y u cos(b - u y) - (-1 + (-1 + e^y) u^2) sin(b - u y))/(u + u^3)
        double value = (exp_y * u * cosine - (-1 + (-1 + exp_y) * u2) * sine) / (u * (1 + u2));
        return multiplier_if_call * value;
    };

    double return_value = 0.0;
    for (int j = 1; j <= static_cast<int>(params.m_J_payoff); ++j)
    {
        double w_j = (2.0 * j - 1.0) * MY_PI / params.m_N_payoff;
        double u = w_j * params.m_two_to_the_m;
        return_value += Integral(to, u, w_j) - Integral(from, u, w_j);
    }

    double result_times_two_to_the_m = return_value * params.m_sqrt_two_to_the_m;
    return result_times_two_to_the_m / params.m_J_payoff;
}

double EuropeanOptionContract::GetPayoffNonKComponentNewPaper(int k, Swift::SwiftParameters const& params, bool is_call) const
{
    double from = GetBoundedFrom(params.m_payoff_from, is_call);
    double to = GetBoundedTo(params.m_payoff_to, is_call);
    if (to <= from)
        return 0.0;

    auto Integral = [is_call](double y, double u)
    {
        double multiplier_if_call = is_call ? -1.0 : 1.0;
        // -i e^(i A y) (1/A - e^y/(-i + A))
        return -1i * std::exp(1i * u * y) * (1.0 / u - std::exp(y) / (u - 1i)) * multiplier_if_call;
    };

    std::complex<double> complex_val{0.0, 0.0};
    for (int j = 1; j <= params.m_J_payoff; ++j)
    {
        double w_j = (2.0 * j - 1.0) * MY_PI / params.m_N_payoff;
        double u = w_j * params.m_two_to_the_m;
        std::complex<double> factor = std::exp(-1i * w_j * static_cast<double>(k));
        // integral (1 - e^y) * e^(i * w_j * 2^m * y)
        complex_val += factor * (Integral(to, u) - Integral(from, u));
    }

    return params.m_sqrt_two_to_the_m / params.m_J_payoff * complex_val.real();
}

double EuropeanOptionContract::GetPayoffKComponent(double K) const
{
    return K;
}

double EuropeanOptionContract::V(int k, Swift::SwiftParameters const& params, bool is_call) const
{
    double from = GetBoundedFrom(params.m_payoff_from, is_call);
    double to = GetBoundedTo(params.m_payoff_to, is_call);
    if (to <= from)
        return 0.0;
    double sum = 0;
    for (int j = 1; j <= params.m_J_payoff; j++)
    {
        double integral1 = I1(from, to, k, j, params);
        double integral2 = I2(from, to, k, j, params);
        sum += integral1 - integral2;
    }
    return sum / params.m_J_payoff * params.m_sqrt_two_to_the_m;
}

double EuropeanOptionContract::I1(double from, double to, int k, int j, Swift::SwiftParameters const& params) const
{
    double C= ((2*j-1.)/(params.m_N_payoff)) * MY_PI;
    double C2= (C * params.m_two_to_the_m) / (1. + pow(C * params.m_two_to_the_m, 2));
    double C3= exp(to) * sin(C * (params.m_two_to_the_m * to - k)) - exp(from) * sin(C * (params.m_two_to_the_m * from - k));
    double C4= (1./(C * params.m_two_to_the_m)) * (exp(to) * cos(C * (params.m_two_to_the_m * to - k)) - exp(from) * cos(C * (params.m_two_to_the_m * from - k)));
    return(C2*(C3+C4));
}

double EuropeanOptionContract::I2(double from, double to, int k, int j, Swift::SwiftParameters const& params) const
{
    double C= ((2*j-1.)/(params.m_N_payoff)) * MY_PI;
    double C2= (1./(C * params.m_two_to_the_m)) * (sin(C * (params.m_two_to_the_m * to - k)) - sin(C * (params.m_two_to_the_m * from - k)));
    return(C2);
}

double CashOrNothingContract::GetPayoffNonKComponent(int k, Swift::SwiftParameters const& params, bool is_call) const
{
//        const int sign = (k == 0) ? 0 : ((k > 0) ? 1 : -1);
//        return std::pow(2.0, m / -2.0) * (sign * SIApprox(std::abs(k), J) + 0.5);
    double multiplier_if_call = is_call ? 1.0 : -1.0;
    double SI_of_k = Sign(k) * SIApprox(std::abs(k), params.m_J_payoff);
    return (multiplier_if_call * SI_of_k + 0.5) / params.m_sqrt_two_to_the_m;
}

double CashOrNothingContract::GetPayoffKComponent(double /*K*/) const
{
    return 1.0;
}