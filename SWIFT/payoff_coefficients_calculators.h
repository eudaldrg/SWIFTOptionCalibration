#pragma once

#include <vector>
#include <complex>
#include "option_contracts.h"

class OptionContract;
namespace Swift {

class SwiftParameters;

class PayoffCalculator
{
public:
    PayoffCalculator(OptionContract const& contract, SwiftParameters const& params);

    [[nodiscard]] virtual std::vector<double> GetNonKPayoffCoefs(bool is_call) const = 0;

    [[nodiscard]] double GetBoundedFrom(double from, bool is_call) const
    {
        return is_call ? std::max(from, 0.0) : from;
    }
    [[nodiscard]] double GetBoundedTo(double to, bool is_call) const
    {
        return is_call ? to : std::min(to, 0.0);
    }

protected:
    OptionContract const& m_contract;
    SwiftParameters const& m_params;
};

class ExplicitPayoffCalculator : PayoffCalculator
{
public:
    ExplicitPayoffCalculator(OptionContract const& contract, SwiftParameters const& params);

    [[nodiscard]] std::vector<double> GetNonKPayoffCoefs(bool is_call) const final;
};

class ExplicitVietaPayoffCalculator : PayoffCalculator
{
public:
    ExplicitVietaPayoffCalculator(OptionContract const& contract, SwiftParameters const& params);

    [[nodiscard]] std::vector<double> GetNonKPayoffCoefs(bool is_call) const final;
};

class FFTPayoffCalculator : PayoffCalculator
{
public:
    FFTPayoffCalculator(OptionContract const& contract, SwiftParameters const& params);

    [[nodiscard]] std::vector<double> GetNonKPayoffCoefs(bool is_call) const final;
};

class DCTAndDSTPayoffCalculator : PayoffCalculator
{
public:
    DCTAndDSTPayoffCalculator(OptionContract const& contract, SwiftParameters const& params);

    [[nodiscard]] std::vector<double> GetNonKPayoffCoefs(bool is_call) const final;
};

}