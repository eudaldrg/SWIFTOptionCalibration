//#pragma once
//
//#include <vector>
//#include <complex>
//#include "option_contracts.h"
//
//namespace Swift {
//
//class SwiftParameters;
//
//class PayoffCalculator
//{
//public:
//    PayoffCalculator(SwiftParameters const& params);
//
//    [[nodiscard]] virtual std::vector<double> GetSincIntegralApproximation() const = 0;
//
//protected:
//    SwiftParameters const& m_params;
//};
//
//class ExplicitPayoffCalculator : PayoffCalculator
//{
//public:
//    ExplicitPayoffCalculator(OptionContract const& contract, SwiftParameters const& params);
//
//    [[nodiscard]] std::vector<double> GetNonKPayoffCoefs(bool is_call) const final;
//};
//
//class ExplicitVietaPayoffCalculator : PayoffCalculator
//{
//public:
//    ExplicitVietaPayoffCalculator(OptionContract const& contract, SwiftParameters const& params);
//
//    [[nodiscard]] std::vector<double> GetNonKPayoffCoefs(bool is_call) const final;
//};
//
//class FFTPayoffCalculator : PayoffCalculator
//{
//public:
//    FFTPayoffCalculator(OptionContract const& contract, SwiftParameters const& params);
//
//    [[nodiscard]] std::vector<double> GetNonKPayoffCoefs(bool is_call) const final;
//};
//
//class DCTAndDSTPayoffCalculator : PayoffCalculator
//{
//public:
//    DCTAndDSTPayoffCalculator(OptionContract const& contract, SwiftParameters const& params);
//
//    [[nodiscard]] std::vector<double> GetNonKPayoffCoefs(bool is_call) const final;
//};
//
//}