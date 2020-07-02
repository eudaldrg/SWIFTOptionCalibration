#pragma once

#include <boost/math/distributions/normal.hpp>
#include "distributions.h"

inline double GetStandardCDF(double x)
{
    return boost::math::cdf(boost::math::normal(0, 1), x);
}

inline double GetStandardPDF(double x)
{
    return boost::math::pdf(boost::math::normal(0, 1), x);
}
//inline double GetStandardPDF(double x)
//{
//	return boost::math::pdf(boost::math::normal(0, 1), x);
//}

/// https://quant.stackexchange.com/questions/15489/derivation-of-the-formulas-for-the-values-of-european-asset-or-nothing-and-cash
inline double GetBSCallCashOrNothingPrice(double K, double F, double r, double years, double vol, double q = 0)
{
    double vol_times_sqrt_years = vol * sqrt(years);
    double d1 = (std::log(F / K) + (r - q + 0.5 * vol * vol) * years) / vol_times_sqrt_years;
    double d2 = d1 - vol_times_sqrt_years;
    return std::exp(-r * years) * GetStandardCDF(d2);
}

inline double GetBSEuropeanPrice(double K, double F, double r, double years, double vol, bool is_call, double q = 0)
{
    double inverse_exp_t_r = std::exp(-r * years);
    double inverse_exp_t_q = std::exp(-q * years);
    double vol_times_sqrt_years = vol * sqrt(years);
    double d1 = (std::log(F / K) + (r - q + 0.5 * vol * vol) * years) / vol_times_sqrt_years;
    double d2 = d1 - vol_times_sqrt_years;
    return is_call ? F * inverse_exp_t_q * GetStandardCDF(d1) - inverse_exp_t_r * K * GetStandardCDF(d2)
                   : K * inverse_exp_t_r * GetStandardCDF(-d2) - F * inverse_exp_t_q * GetStandardCDF(-d1);
}

inline double GetBSEuropeanVega(double K, double F, double r, double years, double vol, double q)
{
    double inverse_exp_t_r = std::exp(-r * years);
    double vol_times_sqrt_years = vol * sqrt(years);
    double d1 = (std::log(F / K) + (r - q + 0.5 * vol * vol) * years) / vol_times_sqrt_years;
    double d2 = d1 - vol_times_sqrt_years;
    return K * inverse_exp_t_r * GetStandardPDF(d2) * sqrt(years);
}

inline double GetHestonEuropeanPriceCuiMyChar(HestonParameters const& parameters, double S, double K, double r, double q, double T, double max_u = 200)
{
    // x = log(S_T / K) = log(S_t * e^(rT) / K) = log(S_t / K) + rT
    double x = Distribution::GetXCompression(S, K, r, q, T);

    // Heston Pricing Formula: C = S * e^(-qT) * P1 - K * e^(-rT) * P2
    // P1 = 0.5 + 1/pi * Int_0^inf(Re(e^(-i * u * log (K / S_0)) / iu * Char(u - i) / Char(i)) =    0.5 + Qv1 / pi
    // P2 = 0.5 + 1/pi * Int_0^inf(Re(e^(-i * u * log (K / S_0)) / iu * Char(u)) =                  0.5 + Qv2 / pi

    // If we compute K * C / K, we can avoid some computations in each of the quadratures by pre-computing S / K
    // Heston Pricing Final Formula: C = K * 0.5 * (S / K * e^(-qT) - 1 * e^(-rT)) + e^(-rT) / PI *
    //  Int_0^inf
    //  [
    //    Re( e^(iu * log S / K) * S / K / iu (Char(u - i) - Char(u))) du)
    //  ]
    HestonDistribution distribution(parameters, T);
    auto F = [x, &distribution](double u)
    {
        std::complex<double> lhs = distribution.GetChar(-(u - 1i), x);
        std::complex<double> rhs = distribution.GetChar(-u, x);
        return ((lhs - rhs) / (1i * u)).real();
    };

    GaussLegendreIntegrator gauss_legendre_integrator(GaussLegendreIntegrator::GaussLegendreMode::Fast);
    double quadrature_value = gauss_legendre_integrator.GetIntegral(F, 0, max_u, 0.0);
    double inverse_exponential_r_times_T = std::exp(-r * T);
    double inverse_exponential_q_times_T = std::exp(-q * T);
    return K * (0.5 * (inverse_exponential_q_times_T * S / K - inverse_exponential_r_times_T) + inverse_exponential_r_times_T / MY_PI * quadrature_value);
}

inline std::vector<double> GetHestonEuropeanGradientCuiMyChar(HestonParameters const& parameters, double S, double K, double r, double q, double T, double max_u = 200)
{
    // x = log(S_T / K) = log(S_t * e^(rT) / K) = log(S_t / K) + rT
    double x = Distribution::GetXCompression(S, K, r, q, T);

    HestonDistribution distribution(parameters, T);
    auto F0 = [x, &distribution](double u)
    {
        std::complex<double> lhs = distribution.GetCharGradient(-(u - 1i), x)[0];
        std::complex<double> rhs = distribution.GetCharGradient(-u, x)[0];
        return ((lhs - rhs) / (1i * u)).real();
    };
    auto F1 = [x, &distribution](double u)
    {
        std::complex<double> lhs = distribution.GetCharGradient(-(u - 1i), x)[1];
        std::complex<double> rhs = distribution.GetCharGradient(-u, x)[1];
        return ((lhs - rhs) / (1i * u)).real();
    };
    auto F2 = [x, &distribution](double u)
    {
        std::complex<double> lhs = distribution.GetCharGradient(-(u - 1i), x)[2];
        std::complex<double> rhs = distribution.GetCharGradient(-u, x)[2];
        return ((lhs - rhs) / (1i * u)).real();
    };
    auto F3 = [x, &distribution](double u)
    {
        std::complex<double> lhs = distribution.GetCharGradient(-(u - 1i), x)[3];
        std::complex<double> rhs = distribution.GetCharGradient(-u, x)[3];
        return ((lhs - rhs) / (1i * u)).real();
    };
    auto F4 = [x, &distribution](double u)
    {
        std::complex<double> lhs = distribution.GetCharGradient(-(u - 1i), x)[4];
        std::complex<double> rhs = distribution.GetCharGradient(-u, x)[4];
        return ((lhs - rhs) / (1i * u)).real();
    };

    GaussLegendreIntegrator gauss_legendre_integrator(GaussLegendreIntegrator::GaussLegendreMode::Fast);
    double quadrature_value0 = gauss_legendre_integrator.GetIntegral(F0, 0, max_u, 0.0);
    double quadrature_value1 = gauss_legendre_integrator.GetIntegral(F1, 0, max_u, 0.0);
    double quadrature_value2 = gauss_legendre_integrator.GetIntegral(F2, 0, max_u, 0.0);
    double quadrature_value3 = gauss_legendre_integrator.GetIntegral(F3, 0, max_u, 0.0);
    double quadrature_value4 = gauss_legendre_integrator.GetIntegral(F4, 0, max_u, 0.0);
    double inverse_exponential_r_times_T = std::exp(-r * T);
    return {
        K * (inverse_exponential_r_times_T / MY_PI * quadrature_value0),
        K * (inverse_exponential_r_times_T / MY_PI * quadrature_value1),
        K * (inverse_exponential_r_times_T / MY_PI * quadrature_value2),
        K * (inverse_exponential_r_times_T / MY_PI * quadrature_value3),
        K * (inverse_exponential_r_times_T / MY_PI * quadrature_value4)
    };
}

inline double GetBSEuropeanPriceCuiMyChar(GBMParameters const& parameters, double S, double K, double r, double q, double T)
{
    // x = log(S_T / K) = log(S_t * e^(rT) / K) = log(S_t / K) + rT
    double x = Distribution::GetXCompression(S, K, r, q, T);

    // Heston Pricing Formula: C = S * e^(-qT) * P1 - K * e^(-rT) * P2
    // P1 = 0.5 + 1/pi * Int_0^inf(Re(e^(-i * u * log (K / S_0)) / iu * Char(u - i) / Char(i)) =    0.5 + Qv1 / pi
    // P2 = 0.5 + 1/pi * Int_0^inf(Re(e^(-i * u * log (K / S_0)) / iu * Char(u)) =                  0.5 + Qv2 / pi

    // If we compute K * C / K, we can avoid some computations in each of the quadratures by pre-computing S / K
    // Heston Pricing Final Formula: C = K * 0.5 * (S / K * e^(-qT) - 1 * e^(-rT)) + e^(-rT) / PI *
    //  Int_0^inf
    //  [
    //    Re((e^(i(u - i) * x) * Char(u - i) - e^(iux) * Char(u)) / iu) du)
    //  ]
    GBM distribution(parameters.m_vol, T);
    auto F = [x, &distribution](double u)
    {
        std::complex<double> lhs = distribution.GetChar(-(u - 1i), x);
        std::complex<double> rhs = distribution.GetChar(-u, x);
        return ((lhs - rhs) / (1i * u)).real();
    };

    GaussLegendreIntegrator gauss_legendre_integrator(GaussLegendreIntegrator::GaussLegendreMode::Fast);
    double quadrature_value = gauss_legendre_integrator.GetIntegral(F, 0, 200, 0.0);
    double inverse_exponential_r_times_T = std::exp(-r * T);
//    double inverse_exponential_q_times_T = std::exp(-q * T); // we assume 1
//    return K * (0.5 * (inverse_exponential_q_times_T * S / K - inverse_exponential_r_times_T) + inverse_exponential_r_times_T / MY_PI * quadrature_value);
    return K * (0.5 * (/*inverse_exponential_q_times_T * */ S / K - inverse_exponential_r_times_T) + inverse_exponential_r_times_T / MY_PI * quadrature_value);
//    return K * (0.5 * (std::exp(x) - 1) + 1.0 / MY_PI * quadrature_value);
}

inline double GetHestonEuropeanPrice(HestonParameters const& parameters, double S, double K, double r, double T, double q)
{
    double K_over_S = K / S;
    // retrieve model parameters
    double sigma_squared = parameters.m_sigma * parameters.m_sigma;

    // Heston Pricing Formula: C = S * e^(-qT) * P1 - K * e^(-rT) * P2
    // P1 = 0.5 + 1/pi * Int_0^inf(Re(e^(-i * u * log (K / S_0)) / iu * Char(u - i) / Char(i)) = 0.5 + Qv1 / (S_t * pi)
    // P2 = 0.5 + 1/pi * Int_0^inf(Re(e^(-i * u * log (K / S_0)) / iu * Char(u)) = 0.5 + Qv2 / pi

    // If we compute S * C / S, we can avoid some computations in each of the quadratures by pre-computing K / S
    // Heston Pricing Final Formula: C = S * 0.5 * (e^(-qT) - K / S * e^(-rT)) + e^(-rT) / PI *
    //  Int_0^inf
    //  [
    //    Re( e^(-iu * log K / S) / iu (Char(u - i) -  K / S * Char(u - i))) du)
    //  ]
    auto F = [K_over_S, r, T, q, &parameters, sigma_squared](double u)
    {
        auto const&[kappa, v_bar, sigma, rho, v0] = parameters;
        // We need to evaluate the Char function at (u - i) and at u and do
        std::complex<double> ui = 1i * u;
        std::complex<double> u_minus_i_times_i = 1i * (u - 1i);
        // xi = kappa-1i*sigma*rho*u1;
        double sigma_rho = sigma * rho;
        std::complex<double> xi_u1 = kappa - sigma_rho * u_minus_i_times_i;
        std::complex<double> xi_u2 = xi_u1 + sigma_rho;

        // m = 1i*u1 + pow(u1,2);
        std::complex<double> m_u1 = ui + 1.0 + pow(u - 1i, 2); // m_u1 = (PQ_M-1i)*1i + pow(PQ_M-1i, 2);
        std::complex<double> m_u2 = ui + pow(u - 0.0 * 1i, 2);

        // d = sqrt(pow(kes,2) + m*pow(sigma,2));
        std::complex<double> d_u1 = sqrt(pow(xi_u1, 2) + m_u1 * sigma_squared);
        std::complex<double> d_u2 = sqrt(pow(xi_u2, 2) + m_u2 * sigma_squared);

        // g = exp(-kappa*v_bar*rho*T*u1*1i/sigma);
        double tmp1 = -kappa * v_bar * rho * T / sigma;
        double g = exp(tmp1);
        std::complex<double> g_u2 = exp(tmp1 * ui);
        std::complex<double> g_u1 = g_u2 * g;

        // alp, calp, salp
        double half_years_to_expiry = 0.5 * T;
        std::complex<double> alpha = d_u1 * half_years_to_expiry;
        std::complex<double> calp_u1 = cosh(alpha);
        std::complex<double> salp_u1 = sinh(alpha);

        alpha = d_u2 * half_years_to_expiry;
        std::complex<double> calp_u2 = cosh(alpha);
        std::complex<double> salp_u2 = sinh(alpha);

        // A2 = d*calp + kes*salp;
        std::complex<double> A2_u1 = d_u1 * calp_u1 + xi_u1 * salp_u1;
        std::complex<double> A2_u2 = d_u2 * calp_u2 + xi_u2 * salp_u2;

        // A1 = m*salp;
        std::complex<double> A1_u1 = m_u1 * salp_u1;
        std::complex<double> A1_u2 = m_u2 * salp_u2;

        // A = A1/A2;
        std::complex<double> A_u1 = A1_u1 / A2_u1;
        std::complex<double> A_u2 = A1_u2 / A2_u2;

        double twice_kappa_v_bar_over_sigma_squared = 2 * kappa * v_bar / sigma_squared;
        std::complex<double> D_u1 = log(d_u1) + (kappa - d_u1) * half_years_to_expiry - log((d_u1 + xi_u1) * 0.5 + (d_u1 - xi_u1) * 0.5 * exp(-d_u1 * T));
        std::complex<double> D_u2 = log(d_u2) + (kappa - d_u2) * half_years_to_expiry - log((d_u2 + xi_u2) * 0.5 + (d_u2 - xi_u2) * 0.5 * exp(-d_u2 * T));
//        std::complex<double> D_u2 = log(d_u2) + (kappa - d_u2) * half_years_to_expiry - log((d_u2 + xi_u2) * 0.5 + (d_u1 - xi_u2) * 0.5 * exp(-d_u2 * T));

        // F = S * e^((r - q) * T);
        double log_F_over_S = (r - q) * T;
        // characteristic function: Char(u) = exp(ui * log(F / S)) * exp(-v0*A) * g * exp(2*kappa*v_bar/pow(sigma,2)*D) = exp(ui * log(F / S) - v0 * A + 2 * kappa * v_bar / sigma^2 * D) * g
        std::complex<double> char_u_minus_i = exp(u_minus_i_times_i * log_F_over_S - v0 * A_u1 + twice_kappa_v_bar_over_sigma_squared * D_u1) * g_u1;
        std::complex<double> char_u = exp(ui * log_F_over_S - v0 * A_u2 + twice_kappa_v_bar_over_sigma_squared * D_u2) * g_u2;

        // h = e^(-iu log(K / S)) / ui = (K / S) ^ (-ui) / ui
        std::complex<double> h = std::pow(K_over_S, -ui) / ui;
        return std::real(h * (char_u_minus_i - K_over_S * char_u));
    };

    GaussLegendreIntegrator gauss_legendre_integrator(GaussLegendreIntegrator::GaussLegendreMode::Fast);
    double quadrature_value = gauss_legendre_integrator.GetIntegral(F, 0, 200, 0.0);
    double inverse_exponential_r_times_T = std::exp(-r * T);
    double inverse_exponential_q_times_T = std::exp(-q * T);
    return S * (0.5 * (inverse_exponential_q_times_T - K_over_S * inverse_exponential_r_times_T) + inverse_exponential_r_times_T / MY_PI * quadrature_value);
}

struct HestonPriceGradient
{
    double m_k;         // mean reversion rate
    double m_v_bar;     // long term variance
    double m_sigma;     // variance of volatility
    double m_rho;       // correlation between spot and volatility
    double m_v0;        // initial variance

    HestonPriceGradient operator+(HestonPriceGradient const& other)
    {
        HestonPriceGradient return_val = *this;
        return_val += other;
        return return_val;
    }

    HestonPriceGradient& operator+=(HestonPriceGradient const& other)
    {
        m_k += other.m_k;
        m_v_bar += other.m_v_bar;
        m_sigma += other.m_sigma;
        m_rho += other.m_rho;
        m_v0 += other.m_v0;
        return *this;
    }

    HestonPriceGradient operator*(double factor) const
    {
        HestonPriceGradient return_val = *this;
        return_val *= factor;
        return return_val;
    }

    HestonPriceGradient& operator*=(double factor)
    {
        m_k *= factor;
        m_v_bar *= factor;
        m_sigma *= factor;
        m_rho *= factor;
        m_v0 *= factor;
        return *this;
    }
};
inline HestonPriceGradient operator*(double factor, HestonPriceGradient const& heston_partial_derivatives)
{
    return heston_partial_derivatives * factor;
}

inline HestonPriceGradient GetHestonEuropeanJacobian(HestonParameters const& parameters, double S, double K, double r, double T, double q)
{
    double K_over_S = K / S;
    // retrieve model parameters
    double sigma_squared = parameters.m_sigma * parameters.m_sigma;

    auto F = [&parameters, K_over_S, r, T, q, sigma_squared](double u) -> HestonPriceGradient
    {
        auto const&[kappa, v_bar, sigma, rho, v0] = parameters;

        std::complex<double> ui = 1i * u;
        std::complex<double> u_minus_i_times_i = 1i * (u - 1i);

        // We need to evaluate everything at u1 = u - i and u2 = u.
        double sigma_times_rho = sigma * rho;
        std::complex<double> kes_u1 = kappa - sigma_times_rho * u_minus_i_times_i;
        std::complex<double> kes_u2 = kes_u1 + sigma_times_rho;

        // m = 1i*u1 + pow(u1,2);
        std::complex<double> _msqr = pow(u - 1i, 2);
        std::complex<double> msqr = pow(u - 0.0 * 1i, 2);

        std::complex<double> m_u1 = ui + 1.0 + _msqr; //    m_u1 = (PQ_M - 1i)*1i + pow(PQ_M-1i, 2);
        std::complex<double> m_u2 = ui + msqr;

        // d = sqrt(pow(kes,2) + m*pow(sigma,2));
        std::complex<double> d_u1 = sqrt(pow(kes_u1, 2) + m_u1 * sigma_squared);
        std::complex<double> d_u2 = sqrt(pow(kes_u2, 2) + m_u2 * sigma_squared);

        // g = exp(-kappa * b * rho * T * u1 * i / sigma);
        double kappa_v_bar_rho_T = kappa * v_bar * rho * T;
        double minus_kappa_v_bar_rho_T_over_sigma = -kappa_v_bar_rho_T / sigma;
        double inverse_exp_kappa_v_bar_rho_T_over_sigma = exp(minus_kappa_v_bar_rho_T_over_sigma);

        std::complex<double> g_u2 = exp(minus_kappa_v_bar_rho_T_over_sigma * ui);
        std::complex<double> g_u1 = g_u2 * inverse_exp_kappa_v_bar_rho_T_over_sigma;

        // alp, calp, salp
        double halft = 0.5 * T;
        std::complex<double> alpha_u1 = d_u1 * halft;
        std::complex<double> cosh_alpha_u1 = cosh(alpha_u1);
        std::complex<double> sinh_alpha_u1 = sinh(alpha_u1);

        std::complex<double> alpha_u2 = d_u2 * halft;
        std::complex<double> cosh_alpha_u2 = cosh(alpha_u2);
        std::complex<double> sinh_alpha_u2 = sinh(alpha_u2);

        // A2 = d * calp + kes * salp;
        std::complex<double> A2_u1 = d_u1 * cosh_alpha_u1 + kes_u1 * sinh_alpha_u1;
        std::complex<double> A2_u2 = d_u2 * cosh_alpha_u2 + kes_u2 * sinh_alpha_u2;

        // A1 = m * salp;
        std::complex<double> A1_u1 = m_u1 * sinh_alpha_u1;
        std::complex<double> A1_u2 = m_u2 * sinh_alpha_u2;

        // A = A1 / A2;
        std::complex<double> A_u1 = A1_u1 / A2_u1;
        std::complex<double> A_u2 = A1_u2 / A2_u2;

        // B = d * exp(kappa * T / 2) / A2;
        double exp_kappa_times_half_T = exp(kappa * halft); // exp(kappa * T / 2)
        std::complex<double> B_u1 = d_u1 * exp_kappa_times_half_T / A2_u1;
        std::complex<double> B_u2 = d_u2 * exp_kappa_times_half_T / A2_u2;

        double two_kappa_v_bar_over_sigma_squared = 2 * kappa * v_bar / sigma_squared;
        std::complex<double> D_u1 = log(d_u1) + (kappa - d_u1) * halft - log((d_u1 + kes_u1) * 0.5 + (d_u1 - kes_u1) * 0.5 * exp(-d_u1 * T));
//        std::complex<double> D_u2 = log(d_u2) + (kappa - d_u2) * halft - log((d_u2 + kes_u2) * 0.5 + (d_u1 - kes_u2) * 0.5 * exp(-d_u2 * T)); // TODO: It was like this, but I think it's a mistake
        std::complex<double> D_u2 = log(d_u2) + (kappa - d_u2) * halft - log((d_u2 + kes_u2) * 0.5 + (d_u2 - kes_u2) * 0.5 * exp(-d_u2 * T));

        // H = kes*calp + d*salp;
        std::complex<double> H_u1 = kes_u1 * cosh_alpha_u1 + d_u1 * sinh_alpha_u1;
        std::complex<double> H_u2 = kes_u2 * cosh_alpha_u2 + d_u2 * sinh_alpha_u2;

        // lnB = log(B);
        std::complex<double> lnB_u1 = D_u1;
        std::complex<double> lnB_u2 = D_u2;

        // partial b: y3 = y1*(2*kappa*lnB/pow(sigma,2)-kappa*rho*T*u1*i/sigma);
        double two_kappa_over_sigma_squared = two_kappa_v_bar_over_sigma_squared / v_bar;
        double minus_kappa_rho_T_over_sigma = minus_kappa_v_bar_rho_T_over_sigma / v_bar;

        std::complex<double> y3M1 = two_kappa_over_sigma_squared * lnB_u1 + minus_kappa_rho_T_over_sigma * u_minus_i_times_i;
        std::complex<double> y3M2 = two_kappa_over_sigma_squared * lnB_u2 + minus_kappa_rho_T_over_sigma * ui;

        // partial rho:
        double minus_kappa_v_bar_t_over_sigma = minus_kappa_v_bar_rho_T_over_sigma / rho; //-kappa * v_bar * T/sigma;

        // for M1
        std::complex<double> u_minus_i_over_d_u1 = sigma * u_minus_i_times_i / d_u1;
        std::complex<double> pd_prho_u1 = -kes_u1 * u_minus_i_over_d_u1;
        std::complex<double> pA1_prho_u1 = m_u1 * cosh_alpha_u1 * halft * pd_prho_u1;
        std::complex<double> pA2_prho_u1 = -u_minus_i_over_d_u1 * H_u1 * (1.0 + kes_u1 * halft);
        std::complex<double> pA_prho_u1 = (pA1_prho_u1 - A_u1 * pA2_prho_u1) / A2_u1;
        std::complex<double> pd_prho_u1_minus_pA2_prho_u1_times_d_u1_over_A2_u1 = pd_prho_u1 - pA2_prho_u1 * d_u1 / A2_u1;
        std::complex<double> pB_prho_u1 = exp_kappa_times_half_T / A2_u1 * pd_prho_u1_minus_pA2_prho_u1_times_d_u1_over_A2_u1;
        std::complex<double> y4M1 = -v0 * pA_prho_u1 + two_kappa_v_bar_over_sigma_squared * pd_prho_u1_minus_pA2_prho_u1_times_d_u1_over_A2_u1 / d_u1 + minus_kappa_v_bar_t_over_sigma * u_minus_i_times_i;

        // for M2
        std::complex<double> sigma_ui_over_d_u2 = sigma * ui / d_u2;
        std::complex<double> pd_prho_u2 = -kes_u2 * sigma_ui_over_d_u2;
        std::complex<double> pA1_prho_u2 = m_u2 * cosh_alpha_u2 * halft * pd_prho_u2;
        std::complex<double> pA2_prho_u2 = -sigma_ui_over_d_u2 * H_u2 * (1.0 + kes_u2 * halft);
        std::complex<double> pA_prho_u2 = (pA1_prho_u2 - A_u2 * pA2_prho_u2) / A2_u2;
        std::complex<double> pd_phrho_u2_minus_pA2_prho_u2_times_d_u2_over_A2_u2 = pd_prho_u2 - pA2_prho_u2 * d_u2 / A2_u2;
        std::complex<double> pB_prho_u2 = exp_kappa_times_half_T / A2_u2 * pd_phrho_u2_minus_pA2_prho_u2_times_d_u2_over_A2_u2;
        std::complex<double> y4M2 = -v0 * pA_prho_u2 + two_kappa_v_bar_over_sigma_squared * pd_phrho_u2_minus_pA2_prho_u2_times_d_u2_over_A2_u2 / d_u2 + minus_kappa_v_bar_t_over_sigma * ui;

        // partial kappa:
        double v_bar_rho_T_over_sigma = v_bar * rho * T / sigma;
        double two_v_bar_over_sigma_squared = two_kappa_v_bar_over_sigma_squared / kappa; // 2 * v_bar / sigma_squared;

        std::complex<double> minus_one_over_sigma_u_minus_i_times_i = -1.0 / (sigma * u_minus_i_times_i);
        std::complex<double> pB_pa_u1 = minus_one_over_sigma_u_minus_i_times_i * pB_prho_u1 + B_u1 * halft;
        std::complex<double> y5M1 = -v0 * pA_prho_u1 * minus_one_over_sigma_u_minus_i_times_i + two_v_bar_over_sigma_squared * lnB_u1 + kappa * two_v_bar_over_sigma_squared * pB_pa_u1 / B_u1 -
                                    v_bar_rho_T_over_sigma * u_minus_i_times_i;

        std::complex<double> minus_one_over_sigma_ui = -1.0 / (sigma * ui);
        std::complex<double> pB_pa_u2 = minus_one_over_sigma_ui * pB_prho_u2 + B_u2 * halft;
        std::complex<double> y5M2 =
            -v0 * pA_prho_u2 * minus_one_over_sigma_ui + two_v_bar_over_sigma_squared * lnB_u2 + kappa * two_v_bar_over_sigma_squared * pB_pa_u2 / B_u2 - v_bar_rho_T_over_sigma * ui;

        // partial sigma:
        double rho_over_sigma = rho / sigma;
        double four_kappa_v_bar_over_sigma_cubed = 4 * kappa * v_bar / pow(sigma, 3);
        double kappa_v_bar_rho_T_over_sigma_squared = kappa_v_bar_rho_T / sigma_squared;

        // M1
        std::complex<double> pd_pc_u1 = (rho_over_sigma - 1.0 / kes_u1) * pd_prho_u1 + sigma * _msqr / d_u1;
        std::complex<double> pA1_pc_u1 = m_u1 * cosh_alpha_u1 * halft * pd_pc_u1;
        std::complex<double> pA2_pc_u1 = rho_over_sigma * pA2_prho_u1 - 1.0 / u_minus_i_times_i * (2.0 / (T * kes_u1) + 1.0) * pA1_prho_u1 + sigma * halft * A1_u1;
        std::complex<double> pA_pc_u1 = pA1_pc_u1 / A2_u1 - A_u1 / A2_u1 * pA2_pc_u1;
        std::complex<double> y6M1 = -v0 * pA_pc_u1 - four_kappa_v_bar_over_sigma_cubed * lnB_u1 + two_kappa_v_bar_over_sigma_squared / d_u1 * (pd_pc_u1 - d_u1 / A2_u1 * pA2_pc_u1) +
                                    kappa_v_bar_rho_T_over_sigma_squared * u_minus_i_times_i;

        // M2
        std::complex<double> pd_pc_u2 = (rho_over_sigma - 1.0 / kes_u2) * pd_prho_u2 + sigma * msqr / d_u2;
        std::complex<double> pA1_pc_u2 = m_u2 * cosh_alpha_u2 * halft * pd_pc_u2;
        std::complex<double> pA2_pc_u2 = rho_over_sigma * pA2_prho_u2 - 1.0 / ui * (2.0 / (T * kes_u2) + 1.0) * pA1_prho_u2 + sigma * halft * A1_u2;
        std::complex<double> pA_pc_u2 = pA1_pc_u2 / A2_u2 - A_u2 / A2_u2 * pA2_pc_u2;
        std::complex<double> y6M2 = -v0 * pA_pc_u2 - four_kappa_v_bar_over_sigma_cubed * lnB_u2 + two_kappa_v_bar_over_sigma_squared / d_u2 * (pd_pc_u2 - d_u2 / A2_u2 * pA2_pc_u2) +
                                    kappa_v_bar_rho_T_over_sigma_squared * ui;


        double log_F_over_S = (r - q) * T;
        // F = S * e^((r - q) * T)
        // characteristic function: y1 = exp(i * log(F / S) * u1 - v0 * A * 2 * kappa * b / pow(sigma, 2) * D) * g
        std::complex<double> char_u_minus_i_times_i = exp(log_F_over_S * u_minus_i_times_i - v0 * A_u1 + two_kappa_v_bar_over_sigma_squared * D_u1) * g_u1;
        std::complex<double> char_u = exp(log_F_over_S * ui - v0 * A_u2 + two_kappa_v_bar_over_sigma_squared * D_u2) * g_u2;

        std::complex<double> h = std::pow(K_over_S, -ui) / ui;
        double partial_kappa = std::real(h * (char_u_minus_i_times_i * y5M1 - K_over_S * char_u * y5M2));
        double partial_v_bar = std::real(h * (char_u_minus_i_times_i * y3M1 - K_over_S * char_u * y3M2));
        double partial_sigma = std::real(h * (char_u_minus_i_times_i * y6M1 - K_over_S * char_u * y6M2));
        double partial_rho = std::real(h * (char_u_minus_i_times_i * y4M1 - K_over_S * char_u * y4M2));
        double partial_v0 = std::real(-h * (char_u_minus_i_times_i * A_u1 - K_over_S * char_u * A_u2)); // partial v0: y2 = -A*y1; // TODO: Should be -A*y1/v_0

        return {partial_kappa, partial_v_bar, partial_sigma, partial_rho, partial_v0};
    };

    GaussLegendreIntegrator gauss_legendre_integrator(GaussLegendreIntegrator::GaussLegendreMode::Fast);
    HestonPriceGradient quadrature_value = gauss_legendre_integrator.GetIntegral(F, 0, 200, HestonPriceGradient{0.0, 0.0, 0.0, 0.0, 0.0});
    double inverse_exponential_r_times_T_over_pi = std::exp(-r * T) / MY_PI;
    quadrature_value *= inverse_exponential_r_times_T_over_pi * S; // not sure about the S.
    return quadrature_value;
}