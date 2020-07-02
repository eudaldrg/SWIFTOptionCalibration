#pragma once

#include <vector>
#include <complex>

class Distribution;

namespace Swift {

struct SwiftParameters;

class CMCalculator
{
public:
    CMCalculator(Distribution const& distribution, SwiftParameters const& params);

    [[nodiscard]] virtual std::vector<double> GetCMCoefs(double x) const = 0;
    [[nodiscard]] virtual std::vector<std::vector<double>> GetGradientCMCoefs(double x) const = 0;

    /// c_{m,k} \approx c^*_{m,k} = \dfrac{2^{m/2}}{2^{J-1} \sum_{j=1}^{2^{J-1}}} \R [\hat{f}(\dfrac{(2j - 1)
/// \pi 2^m}{2^J}) e^{\dfrac{ik\pi(2j-1)}{2^J}}])]}

//    double GetC_m_k_Error(std::size_t m, int k, double a, double e)

protected:
    Distribution const& m_distribution;
    SwiftParameters const& m_params;
};

class ParsevalCalculator : public CMCalculator
{
public:
    ParsevalCalculator(Distribution const& distribution, SwiftParameters const& params, std::size_t integral_buckets = 10000);

    [[nodiscard]] std::vector<double> GetCMCoefs(double x) const final;
    [[nodiscard]] std::vector<std::vector<double>> GetGradientCMCoefs(double x) const final;

private:
    std::size_t m_integral_buckets;
};

class ExplicitVietaCalculator : public CMCalculator
{
public:
    ExplicitVietaCalculator(Distribution const& distribution, SwiftParameters const& params);

    [[nodiscard]] std::vector<double> GetCMCoefs(double x) const final;
    [[nodiscard]] std::vector<std::vector<double>> GetGradientCMCoefs(double x) const final;

};

class FastVietaCalculator : public CMCalculator
{
public:
    FastVietaCalculator(Distribution const& distribution, SwiftParameters const& params);

    [[nodiscard]] std::vector<double> GetCMCoefs(double x) const final;
    [[nodiscard]] std::vector<std::vector<double>> GetGradientCMCoefs(double x) const final;
};

class NewPaperExplicitCalculator : public CMCalculator
{
public:
    NewPaperExplicitCalculator(Distribution const& distribution, SwiftParameters const& params);

    [[nodiscard]] std::vector<double> GetCMCoefs(double x) const final;
    [[nodiscard]] std::vector<std::vector<double>> GetGradientCMCoefs(double x) const final;
};

}