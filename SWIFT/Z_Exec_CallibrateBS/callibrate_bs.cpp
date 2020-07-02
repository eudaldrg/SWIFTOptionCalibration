#include <iostream>
#include <SWIFT/option_contracts.h>
#include <SWIFT/known_distribution_contract_combinations.h>
#include <SWIFT/swift.h>
#include <LEVMAR/levmar.h>
#include <SWIFT/quick_callibration_swift.h>
#include "vector"
#include "SWIFT/distributions.h"

// market parameters: you may change the number of observations by modifying the size of T and K
struct MarketParameters{
    double S;
    double r;
    double q = 0;
    std::vector<double> T;
    std::vector<double> K;
    std::vector<std::vector<std::complex<double>>> exp_u_j_two_to_the_m_x;
//    Swift::SwiftParameters swift_params{4, -15, 16, 8, 7};
    Swift::SwiftParameters swift_params{3, -7, 8, 8, 7};
    std::unique_ptr<Swift::SwiftInvariantData> swift_invariant_data;
};

// Jacobian (parameter, observation, dim_p, dim_x, arguments)
void GetHestonJacobianForLevenbergMarquard(double *p, double *jac, int m, int n_observations, void *adata) {
    GBMParameters parameters{p[0]};
    auto* market_data_ptr = static_cast<MarketParameters const*>(adata);
    MarketParameters const& market_parameters = *market_data_ptr;
    if (static_cast<std::size_t>(n_observations) != market_parameters.K.size() || static_cast<std::size_t>(n_observations) != market_parameters.T.size())
        throw std::runtime_error("Wrong parameters. The num of parameters doesn't match n_observations");

    double T = market_parameters.T[0];
    GBM distribution(parameters.m_vol, T);
//    Swift::SwiftParameters swift_params = market_parameters.swift_params;
    EuropeanOptionContract contract;
//    Swift::QuickCallibrationSwiftEvaluator eval(market_parameters.swift_params, distribution, contract, true);
    Swift::QuickCallibrationSwiftEvaluator eval(*market_parameters.swift_invariant_data, market_parameters.swift_params, distribution, contract);
    std::vector<std::vector<double>> gradients = eval.GetGradient(market_parameters.S, market_parameters.K, market_parameters.r, market_parameters.q);
    for (int observation = 0; observation < n_observations; ++observation) {
//        double K = market_parameters.K[observation];
//        double T = market_parameters.T[observation];
//        GBM distribution(parameters.m_vol, T);
//        Swift::SwiftParameters swift_params{3, -31, 32, 10, 10};
//        EuropeanOptionContract contract;
//        Swift::SwiftEvaluator eval(swift_params, distribution, contract);
//        std::vector<double> pd = eval.GetGradient(market_parameters.S, K, market_parameters.r, market_parameters.q, true);
        std::vector<double> pd = gradients[observation];
        jac[observation * m] = pd[0];
    }
}

void GetHestonPriceForLevMar(double *p, double *x, int /*m*/, int n_observations, void *adata)
{
    GBMParameters parameters{p[0]};
    auto* market_data_ptr = static_cast<MarketParameters const*>(adata);
    MarketParameters const& market_parameters = *market_data_ptr;
    if (static_cast<std::size_t>(n_observations) != market_parameters.K.size() || static_cast<std::size_t>(n_observations) != market_parameters.T.size())
        throw std::runtime_error("Wrong parameters. The num of parameters doesn't match n_observations");

    double T = market_parameters.T[0];
    GBM distribution(parameters.m_vol, T);
    EuropeanOptionContract contract;
//    Swift::SwiftEvaluator eval(market_parameters.swift_params, distribution, contract);
//    Swift::QuickCallibrationSwiftEvaluator eval(market_parameters.swift_params, distribution, contract, true);
    Swift::QuickCallibrationSwiftEvaluator eval(*market_parameters.swift_invariant_data, market_parameters.swift_params, distribution, contract);
    std::vector<double> prices = eval.GetPrice(market_parameters.S, market_parameters.K, market_parameters.r, market_parameters.q);
    for (int i = 0; i < n_observations; ++i)
    {
//        GBM distribution(parameters.m_vol, market_parameters.T[i]);
//        Swift::SwiftParameters swift_params{3, -31, 32, 10, 10};
//        EuropeanOptionContract contract;
//        Swift::SwiftEvaluator eval(swift_params, distribution, contract);
//        x[i] = eval.GetPrice(market_parameters.S, market_parameters.K[i], market_parameters.r, market_parameters.q, true);
        x[i] = prices[i];
//        std::cout << "i " << i << " x " << x[i] << std::endl;
    }
}

int main()
{
    int M_parameters = 1;  // # of parameters
    int N_observations = 40; // # of observations (consistent with the struct MarketParameters)
//    int N_observations = 4; // # of observations (consistent with the struct MarketParameters)

    MarketParameters market_parameters;

//    std::vector<double> K_over_S =
//    {
//        0.9371, 0.8603, 0.8112, 0.7760
//    };

    //// INIT MARKET PARAMETERS ////
    // array of strikes
    std::vector<double> K_over_S = {
        0.9371, 0.8603, 0.8112, 0.7760, 0.7470, 0.7216, 0.6699, 0.6137,
        0.9956, 0.9868, 0.9728, 0.9588, 0.9464, 0.9358, 0.9175, 0.9025,
        1.0427, 1.0463, 1.0499, 1.0530, 1.0562, 1.0593, 1.0663, 1.0766,
        1.2287, 1.2399, 1.2485, 1.2659, 1.2646, 1.2715, 1.2859, 1.3046,
        1.3939, 1.4102, 1.4291, 1.4456, 1.4603, 1.4736, 1.5005, 1.5328
    };


    // array of expiries
    std::vector<double> expiries = {0.119047619047619, 0.238095238095238, 0.357142857142857, 0.476190476190476, 0.595238095238095, 0.714285714285714, 1.07142857142857, 1.42857142857143,
                                    0.119047619047619, 0.238095238095238, 0.357142857142857, 0.476190476190476, 0.595238095238095, 0.714285714285714, 1.07142857142857, 1.42857142857143,
                                    0.119047619047619, 0.238095238095238, 0.357142857142857, 0.476190476190476, 0.595238095238095, 0.714285714285714, 1.07142857142857, 1.42857142857143,
                                    0.119047619047619, 0.238095238095238, 0.357142857142857, 0.476190476190476, 0.595238095238095, 0.714285714285714, 1.07142857142857, 1.42857142857143,
                                    0.119047619047619, 0.238095238095238, 0.357142857142857, 0.476190476190476, 0.595238095238095, 0.714285714285714, 1.07142857142857, 1.42857142857143};


    // spot and interest rate
    market_parameters.S = 1.0;
    market_parameters.r = 0.02;

    // strikes and expiries
    for (int j = 0; j < N_observations; ++j)
    {
        market_parameters.K.push_back(K_over_S[j]);
        market_parameters.T.push_back(expiries[j]);
    }
    market_parameters.swift_invariant_data = std::make_unique<Swift::SwiftInvariantData>(market_parameters.swift_params, market_parameters.T[0], EuropeanOptionContract{}, true,
        market_parameters.S, market_parameters.K, market_parameters.r, market_parameters.q);

    //// END INIT MARKET PARAMETERS

    // you may set up your optimal model parameters here:
    GBMParameters optimal_parameters{0.25};
    double pstar[1];
    pstar[0] = optimal_parameters.m_vol;

    // compute the market_parameters observations with pstar
    double x[40];
    GetHestonPriceForLevMar(pstar, x, M_parameters, N_observations, &market_parameters);
//    for (std::size_t i = 0; i < K_over_S.size(); ++i)
//    {
//        double current_value = x[i];
//        double real_value = GetBSEuropeanPrice(market_parameters.K[i], market_parameters.S, market_parameters.r, market_parameters.T[i], pstar[0], true, market_parameters.q);
//        std::cout << "K " << market_parameters.K[i] << " T " << market_parameters.T[i] << " value " << current_value << " real value " << real_value << std::endl;
//    }


//    double jacobian[40 * 5];
//    GetGBMJacobianForLevMar(pstar, jacobian, M_parameters, N_observations, &market_parameters);
//    for (int j = 0; j < N_observations; ++j)
//        std::cout << "K " << market_parameters.K[j] << " T " << market_parameters.T[j] << " jacobian " << jacobian[j] << " " << jacobian[j + 1] << " " << jacobian[j + 2] << " " << jacobian[j + 3]
//                  << " " << jacobian[j + 4] << " " << std::endl;
//
    // >>> Enter calibrating routine >>>
    double start_s = clock();

    // algorithm parameters
    double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
    opts[0]=LM_INIT_MU;
    // stopping thresholds for
    opts[1]=1E-10;       // ||J^T e||_inf
    opts[2]=1E-10;       // ||Dp||_2
    opts[3]=1E-10;       // ||e||_2
    opts[4]= LM_DIFF_DELTA; // finite difference if used

    // you may set up your initial point here:
    double p[1];
    p[0] = 0.4;

    std::cout << "\r-------- -------- -------- BS Model Calibrator -------- -------- --------"<< std::endl;
    std::cout << "Parameters:" << "\t         vol" << std::endl;
    std::cout << "\r Initial point:" << "\t"  << std::scientific << std::setprecision(8) << p[0] << std::endl;
    // Calibrate using analytical gradient
//    dlevmar_der(GetGBMPriceForLevMar, GetGBMJacobianForLevMar, p, x, M_parameters, N_observations, 100, opts, info, NULL, NULL, (void*) &market_parameters);
    dlevmar_der(GetHestonPriceForLevMar, GetHestonJacobianForLevenbergMarquard, p, x, M_parameters, N_observations, 5, opts, info, NULL, NULL, (void*) &market_parameters);

    double stop_s = clock();

    std::cout << "Optimum found:" << std::scientific << std::setprecision(8) << "\t"<< p[0]<< "\t" << std::endl;
    std::cout << "Real optimum: " << optimal_parameters.m_vol << std::endl;

    if (int(info[6]) == 6) {
        std::cout << "\r Solved: stopped by small ||e||_2 = "<< info[1] << " < " << opts[3]<< std::endl;
    } else if (int(info[6]) == 1) {
        std::cout << "\r Solved: stopped by small gradient J^T e = " << info[2] << " < " << opts[1]<< std::endl;
    } else if (int(info[6]) == 2) {
        std::cout << "\r Solved: stopped by small change Dp = " << info[3] << " < " << opts[2]<< std::endl;
    } else if (int(info[6]) == 3) {
        std::cout << "\r Unsolved: stopped by itmax " << std::endl;
    } else if (int(info[6]) == 4) {
        std::cout << "\r Unsolved: singular matrix. Restart from current p with increased mu"<< std::endl;
    } else if (int(info[6]) == 5) {
        std::cout << "\r Unsolved: no further error reduction is possible. Restart with increased mu"<< std::endl;
    } else if (int(info[6]) == 7) {
        std::cout << "\r Unsolved: stopped by invalid values, user error"<< std::endl;
    }

    std::cout << "\r-------- -------- -------- Computational cost -------- -------- --------"<< std::endl;
    std::cout << "\r          Time cost: "<< double(stop_s - start_s) / CLOCKS_PER_SEC << " seconds "<< std::endl;
    std::cout << "         Iterations: " << int(info[5]) << std::endl;
    std::cout << "         pv  Evalue: " << int(info[7]) << std::endl;
    std::cout << "         Jac Evalue: "<< int(info[8]) << std::endl;
    std::cout << "# of lin sys solved: " << int(info[9])<< std::endl; //The attempts to reduce error
    std::cout << "\r-------- -------- -------- Residuals -------- -------- --------"<< std::endl;
    std::cout << " \r            ||e0||_2: " << info[0] << std::endl;
    std::cout << "           ||e*||_2: " << info[1]<< std::endl;
    std::cout << "          ||J'e||_inf: " << info[2]<< std::endl;
    std::cout << "           ||Dp||_2: " << info[3]<< std::endl;
}
