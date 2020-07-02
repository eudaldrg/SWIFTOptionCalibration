#pragma once

#include "my_math.h"

class Distribution
{
public:
    explicit Distribution(double T) : m_T(T)
    { }
    virtual ~Distribution() = default;
    [[nodiscard]] virtual std::complex<double> GetNonXChar(std::complex<double> u) const = 0;
    [[nodiscard]] std::complex<double> GetChar(std::complex<double> u, double x) const
    {
        std::complex<double> non_x = GetNonXChar(u);
        std::complex<double> x_factor = GetCharPositionFactor(x, u);
        return non_x * x_factor;
    }
    [[nodiscard]] virtual std::vector<std::complex<double>> GetNonXCharGradient(std::complex<double> u) const = 0;

    [[nodiscard]] std::vector<std::complex<double>> GetCharGradient(std::complex<double> u, double x) const
    {
        std::vector<std::complex<double>> result_components = GetNonXCharGradient(u);
        std::complex<double> x_factor = GetCharPositionFactor(x, u);
        for (auto& val : result_components)
            val *= x_factor;
        return result_components;
    }

    [[nodiscard]] virtual double GetFirstCumulant() const = 0;
    [[nodiscard]] virtual double GetSecondCumulant() const = 0;
    [[nodiscard]] virtual double GetFourthCumulant() const = 0;
    [[nodiscard]] double GetDomainTruncationInitialValue() const
    {
        double c1 = GetFirstCumulant();
        double c2 = GetSecondCumulant();
        double c4 = GetFourthCumulant();
        return c1 + 6 * std::sqrt(c2 + std::sqrt(c4));
    }

    [[nodiscard]] virtual double GetErrorInTermsOfM(std::size_t m, double x) const
    {
        double two_to_the_m = 1U << m;
        const int convergence_coeff = 2;
        return std::pow(two_to_the_m * MY_PI, 1 - convergence_coeff) / (2 * MY_PI * convergence_coeff * m_T)
            * (std::norm(GetChar(-two_to_the_m * MY_PI, x)) + std::norm(GetChar(two_to_the_m * MY_PI, x)));
    }

    // The overall char is regular char * this.
    // x = log(F_{t, T}/K) = log(S_t * exp((r - q) * tau) / K) = (r - q) * tau * log (S_t / K)

    ///// CAREFUL. If you take S / K it's positive, but K / S negative
    static std::complex<double> GetCharPositionFactor(double x, std::complex<double> u)
    {
        return std::exp(-1i * x * u);
    }

    virtual std::size_t GetNumberOfParameters() const = 0;

    static double GetXCompression(double S, double K, double r, double q, double tau)
    {
//    return std::log(S * std::exp((r - q) * tau) / K);
        return (r - q) * tau + std::log(S / K);
    }

    double m_T;
};

struct GBMParameters
{
    double m_vol;
    static std::size_t GetNumberOfParameters()
    {
        return 1;
    }
};


// AKA B&S
class GBM : public Distribution
{
public:
	GBM(double vol, double T) : Distribution(T), m_parameters{vol}, m_vol_2(m_parameters.m_vol * m_parameters.m_vol)
	{ }

//    [[nodiscard]] std::complex<double> GetChar(double u, double x) const
//    {
//        return std::exp(-u * m_T * (i * (m_r - m_q - 0.5 * m_vol_2) + 0.5 * m_vol_2 * u)) * std::exp(-1i * x * u);
//    }
    [[nodiscard]] std::complex<double> GetNonXChar(std::complex<double> u) const final
    {
        return std::exp(-m_T * 0.5 * m_vol_2 * u * (u - 1i));
    }

    [[nodiscard]] std::vector<std::complex<double>> GetNonXCharGradient(std::complex<double> u) const final
    {
	    return {-m_parameters.m_vol * m_T * (u * (u - 1i)) * GetNonXChar(u)};
	    throw std::runtime_error("Not implemented.");
    }
    [[nodiscard]] std::size_t GetNumberOfParameters() const final
    {
	    return GBMParameters::GetNumberOfParameters();
    }

    [[nodiscard]] double GetFirstCumulant() const final { return 1; }
    [[nodiscard]] double GetSecondCumulant() const final { return 1; }
    [[nodiscard]] double GetFourthCumulant() const final { return 1; }

    GBMParameters m_parameters;
    double m_vol_2;
};

// AKA B&S
class GBMDerivative : public Distribution
{
public:
    GBMDerivative(double vol, double T) : Distribution(T), m_parameters{vol}, m_vol_2(m_parameters.m_vol * m_parameters.m_vol)
    { }

//    [[nodiscard]] std::complex<double> GetChar(double u, double x) const
//    {
//        return std::exp(-u * m_T * (i * (m_r - m_q - 0.5 * m_vol_2) + 0.5 * m_vol_2 * u)) * std::exp(-1i * x * u);
//    }
    [[nodiscard]] std::complex<double> GetNonXChar(std::complex<double> u) const final
    {
        return -m_parameters.m_vol * m_T * (u * (u - 1i)) * std::exp(-m_T * 0.5 * m_vol_2 * u * (u - 1i));
    }

    [[nodiscard]] std::vector<std::complex<double>> GetNonXCharGradient(std::complex<double> /*u*/) const final
    {
//        return {GetNonXChar(u)};
        throw std::runtime_error("Not implemented.");
    }
    [[nodiscard]] std::size_t GetNumberOfParameters() const final
    {
        return GBMParameters::GetNumberOfParameters();
    }

    [[nodiscard]] double GetFirstCumulant() const final
    {
        double mu = -m_vol_2 / 2; // The r - q part is already taken into account in x0.
        return mu * m_T;
    }
    [[nodiscard]] double GetSecondCumulant() const final
    {
        return m_vol_2 * m_T;
    }
    [[nodiscard]] double GetFourthCumulant() const final
    {
        return 0;
    }

    GBMParameters m_parameters;
    double m_vol_2;
};

/// Parameters of the heston distribution
struct HestonParameters
{
public:
    HestonParameters(double k, double v_bar, double sigma, double rho, double v_0) : m_kappa(k), m_v_bar(v_bar), m_sigma(sigma), m_rho(rho), m_v0(v_0) {}

    static std::size_t GetNumberOfParameters()
    {
        return 5;
    }

	double m_kappa;         // mean reversion rate
	double m_v_bar;     // long term variance
	double m_sigma;     // variance of volatility
	double m_rho;       // correlation between spot and volatility
	double m_v0;        // initial variance
};

std::ostream& operator<<(std::ostream& out, HestonParameters const& heston_parameters);

class HestonDistribution : public Distribution
{
public:
	HestonDistribution(HestonParameters parameters, double T)
		: Distribution(T), m_parameters(parameters)
	{
		m_sigma_squared = m_parameters.m_sigma * m_parameters.m_sigma;
		m_sigma_times_rho = m_parameters.m_sigma * m_parameters.m_rho;
        m_kappa_v_bar_rho_T = m_parameters.m_kappa * m_parameters.m_v_bar * m_parameters.m_rho * T;
        m_kappa_v_bar_rho_T_over_sigma = m_kappa_v_bar_rho_T / m_parameters.m_sigma;
        m_half_T = 0.5 * T;
        m_two_kappa_v_bar_over_sigma_squared = 2 * m_parameters.m_kappa * m_parameters.m_v_bar / m_sigma_squared;
	}

	struct HelperVariables
	{
		std::complex<double> xi;
		std::complex<double> d;
		std::complex<double> A1;
		std::complex<double> A2;
		std::complex<double> A;
		std::complex<double> D;
	};

	HelperVariables GetHelperVariables(std::complex<double> u, double tau) const
	{
		auto const& [k, v_bar, sigma, rho, v_0] = m_parameters;
		std::complex<double> xi = k - sigma * rho * u * 1i;
		std::complex<double> d = std::sqrt(xi * xi + m_sigma_squared * (u * u + u * 1i));
		std::complex<double> A1 = (u * u + 1i * u) * std::sinh(0.5 * d * tau);
		std::complex<double> A2 = d * std::cosh(0.5 * d * tau) / v_0 + xi * std::sinh(d * tau * 0.5) / v_0;
		std::complex<double> A = A1 / A2;
		std::complex<double> D = std::log(d / v_0) + (k - d) * tau / 2.0 - std::log((d + xi) / (2 * v_0) + (d - xi) / (2 * v_0)
                                                                                                           * std::exp(-d * tau));

		return{xi, d, A1, A2, A, D};
	}

	[[nodiscard]] std::complex<double> GetNonXChar(std::complex<double> u) const final
    {
        u = -u; //TODO: Fix properly later on.
        return GetCuiCharExplicit(u, m_T);
//        return GetGatheralChar(u, m_T);
//        return GetHestonChar(u, m_T);
    }

    [[nodiscard]] std::vector<std::complex<double>> GetNonXCharGradient(std::complex<double> u) const final
    {
        u = -u; //TODO: Fix properly later on.
        return GetCuiGradient(u, m_T);
    }

	[[nodiscard]] std::complex<double> GetCuiChar(std::complex<double> u, double tau) const
    {
        auto const& [k, v_bar, sigma, rho, v_0] = m_parameters;
        auto const& [xi, d, A1, A2, A, D] = GetHelperVariables(u, tau);
	    return std::exp(- (k * v_bar * rho * tau * 1i * u) / sigma - A + (2 * k * v_bar * D) / (m_sigma_squared));
    }

    [[nodiscard]] std::complex<double> GetCuiCharExplicit(std::complex<double> u, double T) const
    {
        std::complex<double> ui = 1i * u;
        std::complex<double> u_squared = u * u;

        std::complex<double> xi = m_parameters.m_kappa - m_sigma_times_rho * ui; // xi = kappa - sigma * rho * u * i
        std::complex<double> xi2 = xi * xi;
        std::complex<double> m = ui + u_squared; // m = u * i + u^2;
        std::complex<double> d = std::sqrt(xi2 + m * m_sigma_squared); // d = sqrt(pow(xi,2) + m*pow(sigma,2));

        // alp, calp, salp
        std::complex<double> alpha = d * m_half_T;
        std::complex<double> cosh_alpha = cosh(alpha);
        std::complex<double> sinh_alpha = sinh(alpha);
        std::complex<double> A2_times_v0 = d * cosh_alpha + xi * sinh_alpha;
        std::complex<double> A1 = m * sinh_alpha;
        std::complex<double> A_over_v0 = A1 / A2_times_v0;

        std::complex<double> D = std::log(d) + (m_parameters.m_kappa - d) * m_half_T - std::log((d + xi) * 0.5 + (d - xi) * 0.5 * exp(-d * T));

//        std::complex<double> g = std::exp(m_kappa_v_bar_rho_T_over_sigma * ui);

        // g = exp(-kappa * b * rho * T * u * i / sigma);
        // char = std::exp(-A + (2 k v_bar / sigma^2) * D) * g
        std::complex<double> char_u = std::exp(-m_parameters.m_v0 * A_over_v0 + m_two_kappa_v_bar_over_sigma_squared * D - m_kappa_v_bar_rho_T_over_sigma * ui);
        return char_u;
    }
    [[nodiscard]] std::vector<std::complex<double>> GetCuiGradient(std::complex<double> u, double T) const
    {
        auto const& [kappa, v_bar, sigma, rho, v0] = m_parameters;

        std::complex<double> ui = 1i * u;
        std::complex<double> u_squared = u * u;

        std::complex<double> xi = m_parameters.m_kappa - m_sigma_times_rho * ui; // xi = kappa - sigma * rho * u * i
        std::complex<double> xi2 = xi * xi;
        std::complex<double> m = ui + u_squared; // m = u * i + u^2;
        std::complex<double> d = std::sqrt(xi2 + m * m_sigma_squared); // d = sqrt(pow(xi,2) + m*pow(sigma,2));

        // alp, calp, salp
        std::complex<double> alpha = d * m_half_T;
        std::complex<double> cosh_alpha = cosh(alpha);
        std::complex<double> sinh_alpha = sinh(alpha);
        std::complex<double> A2_times_v0 = d * cosh_alpha + xi * sinh_alpha;
        std::complex<double> A1 = m * sinh_alpha;
        std::complex<double> A_over_v0 = A1 / A2_times_v0;

        std::complex<double> D = std::log(d) + (m_parameters.m_kappa - d) * m_half_T - std::log((d + xi) * 0.5 + (d - xi) * 0.5 * exp(-d * T));

//        std::complex<double> g = exp(minus_kappa_v_bar_rho_T_over_sigma * ui);

        // F = S * e^((r - q) * T)
        // characteristic function: y1 = exp(i * log(F / S) * u) * exp(-A + 2 * kappa * b / pow(sigma, 2) * D) * g
        // But we only care about the second exponential, the rest depends only on market parameters and will be computed separately.
        std::complex<double> char_u = exp(-m_parameters.m_v0 * A_over_v0 + m_two_kappa_v_bar_over_sigma_squared * D - m_kappa_v_bar_rho_T_over_sigma * ui);

        // B = d * exp(kappa * T / 2) / (A2 * v0);
        double exp_kappa_times_half_T = exp(kappa * m_half_T); // exp(kappa * T / 2)
        std::complex<double> B = d * exp_kappa_times_half_T / A2_times_v0;

        // g = exp(-kappa * b * rho * T * u1 * i / sigma);
        double kappa_v_bar_rho_T = kappa * v_bar * rho * T;  // TAG: PRECOMPUTE
        double minus_kappa_v_bar_rho_T_over_sigma = -kappa_v_bar_rho_T / sigma;  // TAG: PRECOMPUTE

        std::complex<double> H = xi * cosh_alpha + d * sinh_alpha;

        // lnB = log(B);
        std::complex<double> lnB = D;

        // partial b: y3 = y1*(2*kappa*lnB/pow(sigma,2)-kappa*rho*T*u1*i/sigma);
        double two_kappa_over_sigma_squared = m_two_kappa_v_bar_over_sigma_squared / v_bar;
        double minus_kappa_rho_T_over_sigma = minus_kappa_v_bar_rho_T_over_sigma / v_bar;

        std::complex<double> h_v_bar = two_kappa_over_sigma_squared * lnB + minus_kappa_rho_T_over_sigma * ui;

        // partial rho:
        double minus_kappa_v_bar_t_over_sigma = minus_kappa_v_bar_rho_T_over_sigma / rho; //-kappa * v_bar * T/sigma;

        std::complex<double> sigma_ui_over_d = sigma * ui / d;
        std::complex<double> pd_prho = -xi * sigma_ui_over_d;
        std::complex<double> pA1_prho = m * cosh_alpha * m_half_T * pd_prho;
        std::complex<double> pA2_prho = -sigma_ui_over_d * H * (1.0 + xi * m_half_T);
        std::complex<double> pA_prho = (pA1_prho - A_over_v0 * pA2_prho) / A2_times_v0;
        std::complex<double> pd_phrho_minus_pA2_prho_times_d_over_A2 = pd_prho - pA2_prho * d / A2_times_v0;
        std::complex<double> pB_prho = exp_kappa_times_half_T / A2_times_v0 * pd_phrho_minus_pA2_prho_times_d_over_A2;
        std::complex<double> h_rho = -v0 * pA_prho + m_two_kappa_v_bar_over_sigma_squared * pd_phrho_minus_pA2_prho_times_d_over_A2 / d + minus_kappa_v_bar_t_over_sigma * ui;

        // partial kappa:
        double v_bar_rho_T_over_sigma = v_bar * rho * T / sigma;
        double two_v_bar_over_sigma_squared = m_two_kappa_v_bar_over_sigma_squared / kappa; // 2 * v_bar / sigma_squared;

        std::complex<double> minus_one_over_sigma_ui = -1.0 / (sigma * ui);
        std::complex<double> pB_pa = minus_one_over_sigma_ui * pB_prho + B * m_half_T;
        std::complex<double> h_kappa = -v0 * pA_prho * minus_one_over_sigma_ui + two_v_bar_over_sigma_squared * lnB + kappa * two_v_bar_over_sigma_squared * pB_pa / B - v_bar_rho_T_over_sigma * ui;

        // partial sigma:
        double rho_over_sigma = rho / sigma;
        double four_kappa_v_bar_over_sigma_cubed = 4 * kappa * v_bar / pow(sigma, 3);
        double kappa_v_bar_rho_T_over_sigma_squared = kappa_v_bar_rho_T / m_sigma_squared;
        std::complex<double> pd_pc = (rho_over_sigma - 1.0 / xi) * pd_prho + sigma * u_squared / d;
        std::complex<double> pA1_pc = m * cosh_alpha * m_half_T * pd_pc;
        std::complex<double> pA2_pc = rho_over_sigma * pA2_prho - 1.0 / ui * (2.0 / (T * xi) + 1.0) * pA1_prho + sigma * m_half_T * A1;
        std::complex<double> pA_pc = pA1_pc / A2_times_v0 - A_over_v0 / A2_times_v0 * pA2_pc;
        std::complex<double> h_sigma = -v0 * pA_pc - four_kappa_v_bar_over_sigma_cubed * lnB + m_two_kappa_v_bar_over_sigma_squared / d * (pd_pc - d / A2_times_v0 * pA2_pc) + kappa_v_bar_rho_T_over_sigma_squared * ui;

        return {char_u * h_kappa, char_u * h_v_bar, char_u * h_sigma, char_u * h_rho, -A_over_v0 * char_u};
    }

    std::complex<double> chf(double u, double T)
    {
        auto const& [kappa, vmean, sigma, rho, v0] = m_parameters;

        std::complex<double> ui = 1i * u;
        std::complex<double> u_squared = u * u;

        std::complex<double> xi = kappa + sigma * rho * ui;
        std::complex<double> m = -ui + u_squared;
        std::complex<double> d = std::sqrt(xi * xi + m * m_sigma_squared);
        std::complex<double> g = (xi - d) / (xi + d);
        std::complex<double> e1 = (v0 / pow(sigma, 2)) * ((1. - std::exp(-d * T)) / (1. - g * std::exp(-d * T))) * (kappa + rho * sigma * u * 1i - d);
        std::complex<double> e2 = ((kappa * vmean) / std::pow(sigma, 2)) * (T * (kappa + rho * sigma * u * 1i - d) - 2. * std::log((1. - g * std::exp(-d * T)) / (1. - g)));

        return std::exp(e1 + e2);
    }

	// The original paper used:
	// beta instead of xi
	// lambda instead of kappa
	// eta instead of sigma
	[[nodiscard]] std::complex<double> GetGatheralChar(std::complex<double> u, double tau) const
    {
        auto const& [k, v_bar, sigma, rho, v_0] = m_parameters;
        std::complex<double> xi = k - sigma * rho * u * 1i;
        std::complex<double> d = std::sqrt(xi * xi + m_sigma_squared * (u * u + u * 1i));

        std::complex<double> r_minus = (xi - d) / m_sigma_squared;
//        std::complex<double> r_plus = (xi + d) / m_sigma_squared;
        std::complex<double> g2 = (xi - d) / (xi + d); // r_minus / r_plus

        std::complex<double> exp_minus_d_times_tau = std::exp(-d * tau);
        std::complex<double> G2 = (1.0 - exp_minus_d_times_tau) / (1.0 - g2 * exp_minus_d_times_tau);

        std::complex<double> D = r_minus * G2;
        std::complex<double> C = k * (r_minus * tau - 2.0 / m_sigma_squared * std::log((1.0 - g2 * exp_minus_d_times_tau) / (1.0 - g2)));

        return std::exp(C * v_bar + D * v_0);
    }

	[[nodiscard]] std::complex<double> GetHestonChar(std::complex<double> u, double tau) const
	{
		auto const& [k, v_bar, sigma, rho, v_0] = m_parameters;
		std::complex<double> xi = k - sigma * rho * u * 1i;
		std::complex<double> d = std::sqrt(xi * xi + m_sigma_squared * (u * u + u * 1i));

        std::complex<double> r_minus = (xi - d) / m_sigma_squared;
        std::complex<double> r_plus = (xi + d) / m_sigma_squared;
        std::complex<double> g1 = r_plus / r_minus;
//        std::complex<double> g2 = r_minus / r_plus;

        std::complex<double> exp_d_times_tau = std::exp(d * tau);
        std::complex<double> G1 = (1.0 - exp_d_times_tau) / (1.0 - g1 * exp_d_times_tau);

        std::complex<double> D = r_plus * G1;
        std::complex<double> C = k * (r_plus * tau - 2.0 / m_sigma_squared * std::log((1.0 - g1 * exp_d_times_tau) / (1.0 - g1)));
		return std::exp(C * v_bar + D * v_0);
	}

    //https://arxiv.org/pdf/0909.3978.pdf
    [[nodiscard]] double GetFirstCumulant() const final
    {
//        c1=mu*T + (1.-exp(-kappa * T)) * (vmean - v0) / (2. * kappa) - 0.5 * vmean * T;
	    return -0.5 * m_parameters.m_v_bar * m_T;
    }
    [[nodiscard]] double GetSecondCumulant() const final
    {
//        p1= sigma * T * kappa * exp(-kappa * T) * (v0 - vmean) * (8. * kappa * rho - 4. * sigma);
//        p2= kappa * rho * sigma * (1. - exp(-kappa * T)) * (16. * vmean - 8. * v0);
//        p3= 2. * vmean * kappa * T * (-4. * kappa * rho * sigma + pow(sigma, 2) + 4. * pow(kappa, 2));
//        p4= pow(sigma, 2) * ((vmean - 2. * v0) * exp(-2. * kappa * T) + vmean * (6. * exp(-kappa * T) - 7.) + 2. * v0);
//        p5= 8. * pow(kappa, 2) * (v0 - vmean) * (1. - exp(-kappa * T));
//        c2= (1./(8.*pow(kappa, 3))) * (p1 + p2 + p3 + p4 + p5);

	    auto const& [kappa, s2, k, r, v] = m_parameters;
        double kappa2 = kappa * kappa, kappa3 = kappa2 * kappa;
        double k2 = k * k;
        double t = m_T;
	    return s2 / (8 * kappa3) * (-k2 * std::exp(-2 * kappa * t) + 4 * k * std::exp(- kappa * t) * (k - 2 * kappa * r) + 2 * kappa * t * (4 * kappa2 + k2 - 4 * kappa * k * r) + k * (8 * kappa * r - 3 * k));
    }
    [[nodiscard]] double GetFourthCumulant() const final
    {
        auto const& [a, s2, k, r, v] = m_parameters;
        double a2 = a * a, a3 = a2 * a, a4 = a3 * a;
        double k2 = k * k, k3 = k2 * k, k4 = k3 * k;
        double t = m_T, t2 = t * t;
        double r2 = r * r;
	    return (3 * k2 * s2) / (64 * std::pow(a, 7)) * (
	        -3 * k4 * std::exp(-4 * a * t)
	        -8 * k2 * std::exp(-3 * a * t) * (2 * a * k * t * (k - 2 * a * r) + 4 * a2 + k2 - 6 * a * k * r)
	        -4 * std::exp(-2 * a * t) * (4 * a2 * k2 * t2 * std::pow(k - 2 * a * r, 2) + 2 * a * k * t * (k3 - 16 * a3 * r -12 * a * k2 * r + 4 * a2 * k * (3 + 4 * r2)) + 8 * a4 - 3 * k4 - 32 * a3 * k * r + 8 * a * k3 * r + 16 * a2 * k2 * r2)
	        -8 * std::exp(-a * t) * (- 2 * a2 * k * t2 * std::pow(k - 2 * a * r, 3) - 8 * a * t * (k4 - 7 * a * k3 * r + 4 * a4 * r2 - 8 * a3 * k * r * (1 + r2) + a2 * k2 * (3 + 14 * r2)) - 9 * k4 + 70 * a * k3 * r + 32 * a3 * k * r * (4 + 3 * r2) - 16 * a4 * (1 + 4 * r2) - 4 * a2 * k2 * (9 + 40 * r2))
	        + 4 * a* t * (5 * k4 - 40 * a * k3 * r - 32 * a3 * k * r * (3 + 2 * r2) + 16 * a4 * (1 + 4 * r2) + 24 * a2 * k2 * (1 + 4 * r2))
	        -73 * k4 + 544 * a * k3 * r + 128 * a3 * k * r * (7 + 6 * r2) - 32 * a4 * (3 + 16 * r2) - 64 * a2 * k2 * (4 + 19 * r2));
    }


    [[nodiscard]] std::size_t GetNumberOfParameters() const final
    {
        return HestonParameters::GetNumberOfParameters();
    }
private:
	double m_sigma_squared;
	double m_sigma_times_rho;
    double m_kappa_v_bar_rho_T;
    double m_kappa_v_bar_rho_T_over_sigma;
    double m_half_T;
    double m_two_kappa_v_bar_over_sigma_squared;
	HestonParameters m_parameters;
};