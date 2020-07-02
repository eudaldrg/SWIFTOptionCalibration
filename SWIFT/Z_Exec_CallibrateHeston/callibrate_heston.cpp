#include <iostream>
#include <SWIFT/option_contracts.h>
#include <SWIFT/known_distribution_contract_combinations.h>
#include <SWIFT/swift.h>
#include <LEVMAR/levmar.h>
#include <SWIFT/quick_callibration_swift.h>
#include "vector"
#include "SWIFT/distributions.h"

struct ExpiryData
{
    ExpiryData(double T, std::vector<double>const& K, double S, OptionContract const& option_contract, Distribution const& distribution, double r, double q, double min_x0, double max_x0)
    : m_T(T), m_K(K), m_swift_params(m_m, distribution, min_x0, max_x0), m_swift_invariant_data(m_swift_params, T, option_contract, true, S, K, r, q)
    {
//        (void)S;
//        (void)option_contract;
//        (void)r;
//        (void)q;
    }
    double m_T;
    std::vector<double> m_K;
    std::size_t m_m = 5;
    Swift::SwiftParameters m_swift_params;//{m_m, -7, 8, 8, 7};
    Swift::SwiftInvariantData m_swift_invariant_data;
};

// market parameters: you may change the number of observations by modifying the size of T and K
struct MarketParameters{
    double S;
    double r;
    double q = 0;
    std::vector<ExpiryData> expiry_data;
};

// Jacobian (parameter, observation, dim_p, dim_x, arguments)
void GetHestonJacobianForLevenbergMarquard(double *p, double *jac, int m, int n_observations, void *adata) {
    auto* market_data_ptr = static_cast<MarketParameters const*>(adata);
    MarketParameters const& market_parameters = *market_data_ptr;

    int observation = 0;
    for (auto const& expiry_data : market_parameters.expiry_data)
    {
        HestonDistribution distribution({p[0], p[1], p[2], p[3], p[4]}, expiry_data.m_T);
        EuropeanOptionContract contract;

        Swift::QuickCallibrationSwiftEvaluator eval(expiry_data.m_swift_invariant_data, expiry_data.m_swift_params, distribution, contract);
        std::vector<std::vector<double>> gradients = eval.GetGradient(market_parameters.S, expiry_data.m_K, market_parameters.r, market_parameters.q);

//        Swift::SwiftParameters swift_params = expiry_data.m_swift_params;
        for (std::size_t i = 0; i < expiry_data.m_K.size(); ++i) {

//            double K = expiry_data.m_K[i];
//            Swift::SwiftEvaluator eval(swift_params, distribution, contract);
//            std::vector<double> pd = eval.GetGradient(market_parameters.S, K, market_parameters.r, market_parameters.q, true);

            std::vector<double> const& pd = gradients[i];
            jac[observation * m + 0] = pd[0];
            jac[observation * m + 1] = pd[1];
            jac[observation * m + 2] = pd[2];
            jac[observation * m + 3] = pd[3];
            jac[observation * m + 4] = pd[4];
            ++observation;
        }
    }
    if (observation != n_observations)
        throw std::runtime_error("Observations do not match");
}

void GetHestonPriceForLevMar(double *p, double *x, int /*m*/, int n_observations, void *adata)
{
    auto* market_data_ptr = static_cast<MarketParameters const*>(adata);
    MarketParameters const& market_parameters = *market_data_ptr;
    int observation = 0;
    for (auto const& expiry_data : market_parameters.expiry_data)
    {
        HestonDistribution distribution({p[0], p[1], p[2], p[3], p[4]}, expiry_data.m_T);
        EuropeanOptionContract contract;
        Swift::QuickCallibrationSwiftEvaluator eval(expiry_data.m_swift_invariant_data, expiry_data.m_swift_params, distribution, contract);
        std::vector<double> prices = eval.GetPrice(market_parameters.S, expiry_data.m_K, market_parameters.r, market_parameters.q);

//        Swift::SwiftParameters swift_params = expiry_data.m_swift_params;
        for (std::size_t i = 0; i < expiry_data.m_K.size(); ++i)
        {
//            Swift::SwiftEvaluator eval(swift_params, distribution, contract);
//            x[observation] = eval.GetPrice(market_parameters.S, expiry_data.m_K[i], market_parameters.r, market_parameters.q, true);
            x[observation] = prices[i];
            ++observation;
        }
    }
    if (observation != n_observations)
        throw std::runtime_error("Observations do not match");
}

int main()
{
    int M_parameters = 5;  // # of parameters
    int N_observations = 40; // # of observations (consistent with the struct MarketParameters)
//    int N_observations = 5; // # of observations (consistent with the struct MarketParameters)

    int n_tests = 100;
    std::vector<double> kappas;
    std::vector<double> v_bars;
    std::vector<double> sigmas;
    std::vector<double> rhos;
    std::vector<double> v0s;
    std::vector<double> iters;
    std::vector<double> times;
    std::vector<double> errors;


/////// FX
    double kappa = 0.5;           // |  mean reversion rate
    double v_bar = 0.04;          // |  long term variance
    double sigma = 1;          // |  variance of volatility
    double rho = -0.9;            // |  correlation between spot and volatility
    double v0 = 0.04;             // |  initial variance
/////// IR
//    double kappa = 0.3;           // |  mean reversion rate
//    double v_bar = 0.04;          // |  long term variance
//    double sigma = 0.9;          // |  variance of volatility
//    double rho = -0.5;            // |  correlation between spot and volatility
//    double v0 = 0.04;             // |  initial variance
/////// EQ
//    double kappa = 1;           // |  mean reversion rate
//    double v_bar = 0.09;          // |  long term variance
//    double sigma = 1;          // |  variance of volatility
//    double rho = 0.04;            // |  correlation between spot and volatility
//    double v0 = 0.09;             // |  initial variance

    for (int c_test = 0; c_test < n_tests; ++c_test)
    {
        MarketParameters market_parameters;
// spot and interest rate
        market_parameters.S = 1.0;
        market_parameters.r = 0.02;

        // you may set up your optimal model parameters here:
//    double kappa = 3.00;           // |  mean reversion rate
//    double v_bar = 0.10;          // |  long term variance
//    double sigma = 0.25;          // |  variance of volatility
//    double rho = -0.8;            // |  correlation between spot and volatility
//    double v0 = 0.08;             // |  initial variance
//    double p[5];
//    p[0] = 1.2000;
//    p[1] = 0.20000;
//    p[2] = 0.3000;
//    p[3] = -0.6000;
//    p[4] = 0.2000;

        // you may set up your initial point here:
        double p[5];
        p[0] = kappa * (1.0 + (((double) rand() / (RAND_MAX)) - 0.5) / 5);
        p[1] = v_bar * (1.0 + (((double) rand() / (RAND_MAX)) - 0.5) / 5);
        p[2] = sigma * (1.0 + (((double) rand() / (RAND_MAX)) - 0.5) / 5);
        p[3] = rho * (1.0 + (((double) rand() / (RAND_MAX)) - 0.5) / 5);
        p[4] = v0 * (1.0 + (((double) rand() / (RAND_MAX)) - 0.5) / 5);

        HestonParameters optimal_parameters{kappa, v_bar, sigma, rho, v0};
        double pstar[5];
        pstar[0] = optimal_parameters.m_kappa;
        pstar[1] = optimal_parameters.m_v_bar;
        pstar[2] = optimal_parameters.m_sigma;
        pstar[3] = optimal_parameters.m_rho;
        pstar[4] = optimal_parameters.m_v0;
//    p[0] = 2.9000;
//    p[1] = 0.15000;
//    p[2] = 0.2700;
//    p[3] = -0.7000;
//    p[4] = 0.1000;

//    std::vector<double> expiries = {
//    0.119047619047619,
//    0.238095238095238,
//    0.357142857142857,
//    0.476190476190476,
//    0.595238095238095,
//    0.714285714285714,
//    1.07142857142857,
//    1.42857142857143};
//    std::vector<std::vector<double>> K_over_S = {
//        {0.9371, 0.9956, 1.0427, 1.2287, 1.3939},
//        {0.8603, 0.9868, 1.0463, 1.2399, 1.4102},
//        {0.8112, 0.9728, 1.0499, 1.2485, 1.4291},
//        {0.7760, 0.9588, 1.0530, 1.2659, 1.4456},
//        {0.7470, 0.9464, 1.0562, 1.2646, 1.4603},
//        {0.7216, 0.9358, 1.0593, 1.2715, 1.4736},
//        {0.6699, 0.9175, 1.0663, 1.2859, 1.5005},
//        {0.6137, 0.9025, 1.0766, 1.3046, 1.5328}
//    };

        std::vector<double> expiries = {
            0.119047619047619, 0.119047619047619, 0.119047619047619, 0.119047619047619, 0.119047619047619,
            0.238095238095238, 0.238095238095238, 0.238095238095238, 0.238095238095238, 0.238095238095238,
            0.357142857142857, 0.357142857142857, 0.357142857142857, 0.357142857142857, 0.357142857142857,
            0.476190476190476, 0.476190476190476, 0.476190476190476, 0.476190476190476, 0.476190476190476,
            0.595238095238095, 0.595238095238095, 0.595238095238095, 0.595238095238095, 0.595238095238095,
            0.714285714285714, 0.714285714285714, 0.714285714285714, 0.714285714285714, 0.714285714285714,
            1.07142857142857, 1.07142857142857, 1.07142857142857, 1.07142857142857, 1.07142857142857,
            1.42857142857143, 1.42857142857143, 1.42857142857143, 1.42857142857143, 1.42857142857143};
        std::vector<std::vector<double>> K_over_S = {
            {0.9371}, {0.9956}, {1.0427}, {1.2287}, {1.3939},
            {0.8603}, {0.9868}, {1.0463}, {1.2399}, {1.4102},
            {0.8112}, {0.9728}, {1.0499}, {1.2485}, {1.4291},
            {0.7760}, {0.9588}, {1.0530}, {1.2659}, {1.4456},
            {0.7470}, {0.9464}, {1.0562}, {1.2646}, {1.4603},
            {0.7216}, {0.9358}, {1.0593}, {1.2715}, {1.4736},
            {0.6699}, {0.9175}, {1.0663}, {1.2859}, {1.5005},
            {0.6137}, {0.9025}, {1.0766}, {1.3046}, {1.5328}
        };

//    std::vector<double> expiries = {0.119047619047619};
//    std::vector<std::vector<double>> K_over_S = {{0.9371, 0.9956, 1.0427, 1.2287, 1.3939, 0.8603, 0.9868, 1.0463, 1.2399, 1.4102, 0.8112, 0.9728, 1.0499, 1.2485, 1.4291, 0.7760, 0.9588, 1.0530, 1.2659, 1.4456, 0.7470, 0.9464, 1.0562, 1.2646, 1.4603, 0.7216, 0.9358, 1.0593, 1.2715, 1.4736, 0.6699, 0.9175, 1.0663, 1.2859, 1.5005, 0.6137, 0.9025, 1.0766, 1.3046, 1.5328}};

        //// END INIT MARKET PARAMETERS
        // >>> Enter calibrating routine >>>
        double start_s = clock();

        for (std::size_t index = 0; index < expiries.size(); ++index)
        {
            double expiry = 1* expiries[index];
            std::vector<double> const& strikes = K_over_S[index];
            HestonDistribution distribution(optimal_parameters, expiry);
            double min_x0 = std::accumulate(strikes.begin(), strikes.end(), std::numeric_limits<double>::max(), [&market_parameters, expiry](double cum, auto const& it) { return std::min(cum, Distribution::GetXCompression(market_parameters.S, it, market_parameters.r, market_parameters.q, expiry)) ; });
            double max_x0 = std::accumulate(strikes.begin(), strikes.end(), std::numeric_limits<double>::lowest(), [&market_parameters, expiry](double cum, auto const& it) { return std::max(cum, Distribution::GetXCompression(market_parameters.S, it, market_parameters.r, market_parameters.q, expiry)) ; });
            market_parameters.expiry_data.emplace_back(expiry, strikes, market_parameters.S, EuropeanOptionContract{}, distribution, market_parameters.r, market_parameters.q, min_x0, max_x0);
        }

        double x[40];
        GetHestonPriceForLevMar(pstar, x, M_parameters, N_observations, &market_parameters);
//    for (std::size_t i = 0; i < K_over_S.size(); ++i)
//    {
//        double current_value = x[i];
//        double real_value = GetHestonEuropeanPriceCuiMyChar(optimal_parameters, market_parameters.S, market_parameters.expiry_data[0].m_K[i], market_parameters.r, market_parameters.q, market_parameters.expiry_data[0].m_T);
//        std::cout << "K " << market_parameters.expiry_data[0].m_K[i] << " T " << market_parameters.expiry_data[0].m_T << " value " << current_value << " real value " << real_value << std::endl;
//    }
//    double jacobian[40 * 5];
//    GetGBMJacobianForLevMar(pstar, jacobian, M_parameters, N_observations, &market_parameters);
//    for (std::size_t i = 0; i < K_over_S.size(); ++i)
//    {
////        double real_value = GetBSEuropeanPrice(market_parameters.expiry_data[0].m_K[i], market_parameters.S, market_parameters.r, market_parameters.expiry_data[0].m_T, pstar[0], true, market_parameters.q);
//        std::cout << "K " << market_parameters.expiry_data[0].m_K[i] << " T " << market_parameters.expiry_data[0].m_T << " jacobian " << jacobian[M_parameters * i] << " " << jacobian[M_parameters * i + 1] << " " << jacobian[M_parameters * i + 2]
//                  << " " << jacobian[M_parameters * i + 3]<< " " << jacobian[M_parameters * i + 4] << std::endl;
//    }

        // algorithm parameters
        double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
        opts[0]=LM_INIT_MU;
        // stopping thresholds for
        opts[1]=1E-10;       // ||J^T e||_inf
        opts[2]=1E-10;       // ||Dp||_2
        opts[3]=1E-10;       // ||e||_2
        opts[4]= LM_DIFF_DELTA; // finite difference if used

        std::cout << "\r-------- -------- -------- BS Model Calibrator -------- -------- --------"<< std::endl;
        std::cout << "Parameters:" << "\t         vol" << std::endl;
        std::cout << "\r Initial point:" << "\t"  << std::scientific << std::setprecision(8) << p[0]<< "\t" <<p[1]<< "\t" <<p[2]<< "\t" <<p[3]<< "\t" <<p[4]<< "\t" << std::endl;
        // Calibrate using analytical gradient
//    dlevmar_der(GetGBMPriceForLevMar, GetGBMJacobianForLevMar, p, x, M_parameters, N_observations, 100, opts, info, NULL, NULL, (void*) &market_parameters);
        dlevmar_der(GetHestonPriceForLevMar, GetHestonJacobianForLevenbergMarquard, p, x, M_parameters, N_observations, 100, opts, info, NULL, NULL, (void*) &market_parameters);

        double stop_s = clock();

        kappas.push_back(p[0]);
        v_bars.push_back(p[1]);
        sigmas.push_back(p[2]);
        rhos.push_back(p[3]);
        v0s.push_back(p[4]);

        iters.push_back(info[5]);
        times.push_back(double(stop_s - start_s) / CLOCKS_PER_SEC);
        errors.push_back(info[1]);

//        std::cout << "Optimum found:" << std::scientific << std::setprecision(8) << "\t"<< p[0]<< "\t" <<p[1]<< "\t" <<p[2]<< "\t" <<p[3]<< "\t" <<p[4]<< "\t" << std::endl;
//        std::cout << "Real optimum: " << optimal_parameters.m_kappa << ", " << optimal_parameters.m_v_bar << ", " << optimal_parameters.m_sigma << ", " << optimal_parameters.m_rho << ", " << optimal_parameters.m_v0 << std::endl;
//
//        if (int(info[6]) == 6) {
//            std::cout << "\r Solved: stopped by small ||e||_2 = "<< info[1] << " < " << opts[3]<< std::endl;
//        } else if (int(info[6]) == 1) {
//            std::cout << "\r Solved: stopped by small gradient J^T e = " << info[2] << " < " << opts[1]<< std::endl;
//        } else if (int(info[6]) == 2) {
//            std::cout << "\r Solved: stopped by small change Dp = " << info[3] << " < " << opts[2]<< std::endl;
//        } else if (int(info[6]) == 3) {
//            std::cout << "\r Unsolved: stopped by itmax " << std::endl;
//        } else if (int(info[6]) == 4) {
//            std::cout << "\r Unsolved: singular matrix. Restart from current p with increased mu"<< std::endl;
//        } else if (int(info[6]) == 5) {
//            std::cout << "\r Unsolved: no further error reduction is possible. Restart with increased mu"<< std::endl;
//        } else if (int(info[6]) == 7) {
//            std::cout << "\r Unsolved: stopped by invalid values, user error"<< std::endl;
//        }
//
//        std::cout << "\r-------- -------- -------- Computational cost -------- -------- --------"<< std::endl;
//        std::cout << "\r          Time cost: "<< double(stop_s - start_s) / CLOCKS_PER_SEC << " seconds "<< std::endl;
//        std::cout << "         Iterations: " << int(info[5]) << std::endl;
//        std::cout << "         pv  Evalue: " << int(info[7]) << std::endl;
//        std::cout << "         Jac Evalue: "<< int(info[8]) << std::endl;
//        std::cout << "# of lin sys solved: " << int(info[9])<< std::endl; //The attempts to reduce error
//        std::cout << "\r-------- -------- -------- Residuals -------- -------- --------"<< std::endl;
//        std::cout << " \r            ||e0||_2: " << info[0] << std::endl;
//        std::cout << "           ||e*||_2: " << info[1]<< std::endl;
//        std::cout << "          ||J'e||_inf: " << info[2]<< std::endl;
//        std::cout << "           ||Dp||_2: " << info[3]<< std::endl;
    }

    double avg_kappas = std::accumulate(kappas.begin(), kappas.end(), 0.0, [kappa](auto const& lhs, auto const& rhs){ return lhs + std::abs(rhs - kappa);});
    double avg_v_bars = std::accumulate(v_bars.begin(), v_bars.end(), 0.0, [v_bar](auto const& lhs, auto const& rhs){ return lhs + std::abs(rhs - v_bar);});
    double avg_sigmas = std::accumulate(sigmas.begin(), sigmas.end(), 0.0, [sigma](auto const& lhs, auto const& rhs){ return lhs + std::abs(rhs - sigma);});
    double avg_rhos = std::accumulate(rhos.begin(), rhos.end(), 0.0, [rho](auto const& lhs, auto const& rhs){ return lhs + std::abs(rhs - rho);});
    double avg_v0s = std::accumulate(v0s.begin(), v0s.end(), 0.0, [v0](auto const& lhs, auto const& rhs){ return lhs + std::abs(rhs - v0);});
    double avg_iters = std::accumulate(iters.begin(), iters.end(), 0.0, [](auto const& lhs, auto const& rhs){ return lhs + rhs;});
    double avg_times = std::accumulate(times.begin(), times.end(), 0.0, [](auto const& lhs, auto const& rhs){ return lhs + rhs;});
    double avg_errors = std::accumulate(errors.begin(), errors.end(), 0.0, [](auto const& lhs, auto const& rhs){ return lhs + rhs;});

    std::cout << "avg_kappas " << avg_kappas / kappas.size() << std::endl;
    std::cout << "avg_v_bars " << avg_v_bars / v_bars.size() << std::endl;
    std::cout << "avg_sigmas " << avg_sigmas / sigmas.size() << std::endl;
    std::cout << "avg_rhos " << avg_rhos / rhos.size() << std::endl;
    std::cout << "avg_v0s " << avg_v0s / v0s.size() << std::endl;
    std::cout << "avg_iters " << avg_iters / iters.size() << std::endl;
    std::cout << "avg_times " << avg_times / times.size() << std::endl;
    std::cout << "avg_errors " << avg_errors / errors.size() << std::endl;

}

////// PRICE
//K 0.9371 T 0.119048 value 0.0803314 real value 0.0803314
//K 0.9956 T 0.119048 value 0.0429609 real value 0.0429609
//K 1.0427 T 0.119048 value 0.0225191 real value 0.0225191
//K 1.2287 T 0.119048 value 0.000331382 real value 0.000331382
//K 1.3939 T 0.119048 value 4.31064e-07 real value 4.31064e-07

////// Gradient
//K 0.9371 T 0.119048 jacobian 0.000134411 0.0280474 0.00330929 -0.00115206 0.151462
//K 0.9956 T 0.119048 jacobian 0.0280474 0.00330929 -0.00115206 0.151462 0.000226969
//K 1.0427 T 0.119048 jacobian 0.00330929 -0.00115206 0.151462 0.000226969 0.0375877
//K 1.2287 T 0.119048 jacobian -0.00115206 0.151462 0.000226969 0.0375877 -0.000662676
//K 1.3939 T 0.119048 jacobian 0.151462 0.000226969 0.0375877 -0.000662676 -6.89813e-05