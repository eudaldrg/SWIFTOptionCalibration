#pragma once

#include <boost/math/constants/constants.hpp>
#include <complex>
#include <cmath>
#include <iterator>
#include <iostream>
#include <fftw3.h>

using namespace std::complex_literals;
const double MY_PI = boost::math::constants::pi<double>();
const double MATH_EPSILON = 1e-5;

/// \param [in] x: positive integer to reverse.
/// \param [in] log2n: size of x in bits.
/// \return bitwise reverse of x. i.e. x = 0b1101 and log2n = 4 returns n = 0b1011
inline unsigned int bitReverse(std::size_t x, std::size_t log2n) {
    std::size_t n = 0;
    while (log2n > 0)
    {
        n <<= 1U;
        n |= (x & 1U);
        x >>= 1U;
        --log2n;
    }
    return n;
}

template <typename Func>
double TrapezoidInt(double a, double b, Func f, std::size_t N = 50)
{
    double step = 1.0 / N;
    double result = 0.5 * (f(a) + f(b));
    for (std::size_t i = 1; i < N; ++i)
        result += f(a + i * step);
    result *= step;
    return result;
}

class GaussLegendreIntegrator
{
public:
    enum class GaussLegendreMode{
        Fast,   // number of nodes = 64 costs around 0.25 sec to opt = 1e-12 <Recommended.>. can ensure 1e-8 accuracy for the option price. that is, pvtrapz - pvgauss <=1e-8.
        Medium, // number of nodes = 96 costs around 1.1 sec to opt = 1e-12
        Precise // number of nodes = 128 costs around 1.5 sec to opt = 1e-12
    };

    GaussLegendreIntegrator(GaussLegendreMode mode = GaussLegendreMode::Fast);

    template <typename Function, typename Values>
    Values GetIntegral(Function const& F, double from, double to, Values initial_values) const
    {
        // Gauss-Legendre defines the quadrature points in the interval [-1, 1], so any integration over [from, to] needs a change of variables
        // Int_from^to (F(x) dx) becomes Q * Int_-1^1 (F(P + Q * t) dt)
        double Q = 0.5 * (to - from);
        double P = 0.5 * (to + from);
        // numerical integral settings
        std::size_t NumGrids = m_num_nodes;
        NumGrids = (NumGrids + 1UL) >> 1UL;

        // Computable variables.
        for (std::size_t current_grid = 0; current_grid < NumGrids; current_grid++) {
            // Each u has two associated points with weight w: +-u.
            double quadrature_point_1 = P + Q * m_u[current_grid];
            double quadrature_point_2 = P - Q * m_u[current_grid];
            initial_values += m_w[current_grid] * (F(quadrature_point_2) + F(quadrature_point_1));
        }
        initial_values *= Q;
        return initial_values;
    }

    std::vector<double> const& m_u; // nodes
    std::vector<double> const& m_w; // weights
    std::size_t m_num_nodes; // # of nodes
};

template <typename T> int Sign(T val) {
    return (T(0) < val) - (val < T(0));
}

inline bool IsZero(double val, double tol = MATH_EPSILON)
{
    return std::abs(val) <= tol;
}

template <typename T>
T Mod(T a, T b)
{
    T r = a % b;
    return r >= 0 ? r : r + std::abs(b);
}

inline bool IsSame(double lhs, double rhs, double abs_tol = MATH_EPSILON, double rel_tol = MATH_EPSILON)
{
    if (std::abs(lhs - rhs) > abs_tol)
        return false;
    if (!IsZero(lhs, abs_tol))
        return std::abs((lhs - rhs) / lhs) <= rel_tol;
    return std::abs((lhs - rhs) / rhs) <= rel_tol;
}

/// X = DFT(x) <=> X_k = Sum_{n = 1}^{N}(x_n * e^(-2 * i * PI * k * n / N))
inline std::vector<std::complex<double>> DFT(std::vector<std::complex<double>> const& x)
{
    std::size_t N = x.size();
    std::vector<std::complex<double>> X;
    for (std::size_t k = 0; k < N; ++k)
    {
        std::complex<double> current_transform{0.0,0.0};
        for (std::size_t n = 0; n < N; ++n)
            current_transform += x[n] * std::exp(-1i * 2.0 * MY_PI / static_cast<double>(N) * static_cast<double>(k) * static_cast<double>(n));
        X.push_back(current_transform);
    }
    return X;
}

/// x = IDFT(X) <=> x_n = 1/N * Sum_{k = 1}^{N}(x_n * e^(2 * i * PI * k * n / N))
inline std::vector<std::complex<double>> IDFT(std::vector<std::complex<double>> const& X, bool normalized = true)
{
    std::size_t N = X.size();
    std::vector<std::complex<double>> x;
    for (std::size_t n = 0; n < N; ++n)
    {
        std::complex<double> current_transform{0.0,0.0};
        for (std::size_t k = 0; k < N; ++k)
            current_transform += X[k] * std::exp(1i * 2.0 * MY_PI / static_cast<double>(N) * static_cast<double>(k) * static_cast<double>(n));
        if (normalized)
            current_transform /= N;
        x.push_back(current_transform);
    }
    return x;
}

inline std::vector<std::complex<double>> MY_IDFT(std::vector<std::complex<double>> const& X, bool fast)
{
    if (!fast)
        return IDFT(X, false);

    fftw_complex* frequency_values, * time_values;
    fftw_plan p;
    frequency_values = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * X.size());
    time_values = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * X.size());
    p = fftw_plan_dft_1d(X.size(), frequency_values, time_values, FFTW_BACKWARD, FFTW_ESTIMATE);
    for (std::size_t i = 0; i < X.size(); ++i)
    {
        frequency_values[i][0] = X[i].real();
        frequency_values[i][1] = X[i].imag();
    }

    fftw_execute(p);
    std::vector<std::complex<double>> time_values_std;
    for (std::size_t i = 0; i < X.size(); ++i)
        time_values_std.emplace_back(time_values[i][0], time_values[i][1]);
    return time_values_std;
}

inline void fft(std::vector<std::complex<double>>& a, bool invert) {
    int n = a.size();
    if (n == 1)
        return;

    std::vector<std::complex<double>> a0(n / 2);
    std::vector<std::complex<double>> a1(n / 2);
    for (int i = 0; 2 * i < n; i++) {
        a0[i] = a[2 * i];
        a1[i] = a[2 * i + 1];
    }
    fft(a0, invert);
    fft(a1, invert);

    double ang = 2 * MY_PI / n * (invert ? -1 : 1);
    std::complex<double> w(1), wn(cos(ang), sin(ang));
    for (int i = 0; 2 * i < n; i++) {
        a[i] = a0[i] + w * a1[i];
        a[i + n / 2] = a0[i] - w * a1[i];
        if (invert) {
            a[i] /= 2;
            a[i + n / 2] /= 2;
        }
        w *= wn;
    }
}

inline std::vector<std::complex<double>> FFT(std::vector<std::complex<double>> a, std::size_t log2n)
{
    std::size_t n = 1UL << log2n;
    std::vector<std::complex<double>> b(n);
    for (std::size_t i = 0; i < n; ++i)
        b[bitReverse(i, log2n)] = a[i];

    for (std::size_t s = 1; s <= log2n; ++s) {
        std::size_t m = 1U << s;
        std::size_t m2 = m >> 1U;
        std::complex<double> w(1, 0);
        std::complex<double> wm = exp(-MY_PI * 1i / static_cast<double>(m2));
        for (std::size_t j = 0; j < m2; ++j) {
            for (std::size_t k = j; k < n; k += m) {
                std::complex<double> t = w * b[k + m2];
                std::complex<double> u = b[k];
                b[k] = u + t;
                b[k + m2] = u - t;
            }
            w *= wm;
        }
    }
    return b;
}

inline double Sinc(double x)
{
    return std::sin(MY_PI * x) / (MY_PI * x);
}

/// 2 / pi * (sum_{j=1}^{2^{J-1}} sin(pi * x * (2j - 1) / 2^J) / (2j - 1)
inline double SIApprox(double x, int J)
{
    const int N = J * 2;
    double result = 0.0;
    for (int j = 1; j <= J; ++j)
    {
        std::size_t two_j_minus_1 = 2 * j - 1;
        result += std::sin(two_j_minus_1 * MY_PI * x / N) / two_j_minus_1;
    }
    return result * 2 / MY_PI;
}