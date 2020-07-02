#pragma once

#include <algorithm>
#include <cmath>
#include <cassert>
#include "my_math.h"

namespace Swift {
class SwiftParameters;
}

class OptionContract
{
public:

    [[nodiscard]] virtual double GetPayoffNonKComponent(int k, Swift::SwiftParameters const& params, bool is_call) const = 0;
    [[nodiscard]] virtual double GetPayoffKComponent(double K) const = 0;
    [[nodiscard]] double GetPayoff(double K, int k, Swift::SwiftParameters const& params, bool is_call) const
    {
        return GetPayoffKComponent(K) * GetPayoffNonKComponent(k, params, is_call);
    }
    [[nodiscard]] virtual std::complex<double> GetDefiniteIntegral(double from, double to, double u, bool is_call) const
    {
        (void)from;
        (void)to;
        (void)u;
        (void)is_call;
        throw std::runtime_error("Not implemented");
    }
};

class EuropeanOptionContract : public OptionContract
{
public:
//    [[nodiscard]] double GetExplicitPayoff
    [[nodiscard]] double GetPayoffNonKComponent(int k, Swift::SwiftParameters const& params, bool is_call) const final;
    [[nodiscard]] double GetPayoffKComponent(double K) const final;
    [[nodiscard]] std::complex<double> GetDefiniteIntegral(double from, double to, double u, bool is_call) const final
    {
        if (to <= from)
            return {0.0, 0.0};
        double multiplier_if_call = is_call ? -1.0 : 1.0;
        auto Integral = [u](double y)
        {
            // -i e^(i A y) (1/A - e^y/(-i + A))
            // i e^(-i a x) (1/a - e^x/(i + a))
//            return -1i * std::exp(1i * u * y) * (1.0 / u - std::exp(y) / (u - 1i));
            return 1i * std::exp(-1i * u * y) * (1.0 / u - std::exp(y) / (u + 1i));
        };
        return multiplier_if_call * (Integral(to) - Integral(from));
    }

private:
    // TODO: Document why
    [[nodiscard]] double GetBoundedFrom(double from, bool is_call) const
    {
        return is_call ? std::max(from, 0.0) : from;
    }
    [[nodiscard]] double GetBoundedTo(double to, bool is_call) const
    {
        return is_call ? to : std::min(to, 0.0);
    }

    [[nodiscard]] double GetPayoffNonKComponentNewPaper(int k, Swift::SwiftParameters const& params, bool is_call) const;
    [[nodiscard]] double GetPayoffNonKComponentOldPaper(int k, Swift::SwiftParameters const& params, bool is_call) const;

    [[nodiscard]] double V(int k, Swift::SwiftParameters const& params, bool is_call) const;
    [[nodiscard]] double I1(double from, double to, int k, int j, Swift::SwiftParameters const& params) const;
    [[nodiscard]] double I2(double from, double to, int k, int j, Swift::SwiftParameters const& params) const;

};

class CashOrNothingContract : public OptionContract
{
public:
    [[nodiscard]] double GetPayoffNonKComponent(int k, Swift::SwiftParameters const& params, bool is_call) const final;
    [[nodiscard]] double GetPayoffKComponent(double K) const final;
};



