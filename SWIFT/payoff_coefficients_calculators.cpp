#include "payoff_coefficients_calculators.h"

#include "option_contracts.h"
#include "swift.h"

namespace Swift {

PayoffCalculator::PayoffCalculator(OptionContract const& contract, SwiftParameters const& params) : m_contract(contract), m_params(params){}

ExplicitPayoffCalculator::ExplicitPayoffCalculator(OptionContract const& contract, SwiftParameters const& params) : PayoffCalculator(contract, params) { }

std::vector<double> ExplicitPayoffCalculator::GetNonKPayoffCoefs(bool is_call) const
{
    std::vector<double> c_m;
    c_m.reserve(m_params.m_k2 - m_params.m_k1 + 1);
    for (int k = m_params.m_k1; k <= m_params.m_k2; ++k)
    {
        std::complex<double> complex_val{0.0, 0.0};
        for (int j = 1; j <= m_params.m_J_payoff; ++j)
        {
            double w_j = (2.0 * j - 1.0) * MY_PI / m_params.m_N_payoff;
            double u = w_j * m_params.m_two_to_the_m;
            std::complex<double> factor = std::exp(1i * w_j * static_cast<double>(k));
            std::complex<double> integral = m_contract.GetDefiniteIntegral(GetBoundedFrom(m_params.m_payoff_from, is_call), GetBoundedTo(m_params.m_payoff_to, is_call), u, is_call);
            complex_val += factor * integral;
        }
        c_m.push_back(complex_val.real() * m_params.m_sqrt_two_to_the_m / m_params.m_J_payoff);
    }
    return c_m;
}

std::vector<double> ExplicitVietaPayoffCalculator::GetNonKPayoffCoefs(bool is_call) const
{
    std::vector<double> c_m;
    c_m.reserve(m_params.m_k2 - m_params.m_k1 + 1);
    for (int k = m_params.m_k1; k <= m_params.m_k2; ++k)
    {
        double c_m_k = 0;
        for (int i = 1; i <= m_params.m_J_payoff; ++i)
        {
            double w_j = (2.0 * i - 1.0) * MY_PI / m_params.m_N_payoff;
            double u = w_j * m_params.m_two_to_the_m;
            std::complex<double> factor = std::exp(1i * w_j * static_cast<double>(k));
            std::complex<double> integral = m_contract.GetDefiniteIntegral(GetBoundedFrom(m_params.m_payoff_from, is_call), GetBoundedTo(m_params.m_payoff_to, is_call), u, is_call);
            std::complex<double> cv = factor * integral;
            c_m_k += cv.real();
        }
        c_m.push_back(c_m_k * m_params.m_sqrt_two_to_the_m / m_params.m_J_payoff);
    }
    return c_m;
}

ExplicitVietaPayoffCalculator::ExplicitVietaPayoffCalculator(OptionContract const& contract, SwiftParameters const& params) : PayoffCalculator(contract, params)
{ }

FFTPayoffCalculator::FFTPayoffCalculator(OptionContract const& contract, SwiftParameters const& params) : PayoffCalculator(contract, params) { }

std::vector<double> FFTPayoffCalculator::GetNonKPayoffCoefs(bool is_call) const
{
    std::vector<double> c_m;
    c_m.reserve(m_params.m_k2 - m_params.m_k1 + 1);
    std::vector<std::complex<double>> frequencies;

    for (int i = 0; i < m_params.m_N_payoff; ++i)
    {
        if (i < m_params.m_J_payoff)
        {
            double u = (2.0 * i + 1.0) * MY_PI * static_cast<double>(m_params.m_two_to_the_m) / static_cast<double>(m_params.m_N_payoff);
            frequencies.push_back(m_contract.GetDefiniteIntegral(GetBoundedFrom(m_params.m_payoff_from, is_call), GetBoundedTo(m_params.m_payoff_to, is_call), u, is_call));
        }
        else
            frequencies.emplace_back(0, 0);
    }
    std::vector<std::complex<double>> times = MY_IDFT(frequencies, true);

    for (int k = m_params.m_k1; k <= m_params.m_k2; ++k)
    {
        std::size_t k_mod_N = Mod(k, m_params.m_N_payoff);
        std::complex<double> idf_part = times[k_mod_N];
        std::complex<double> c_m_k_complex_part_dft = std::exp(1i * static_cast<double>(k) * MY_PI / static_cast<double>(m_params.m_N_payoff)) * idf_part;
        c_m.push_back(c_m_k_complex_part_dft.real() * m_params.m_sqrt_two_to_the_m / m_params.m_J_payoff);
    }
    return c_m;
}

DCTAndDSTPayoffCalculator::DCTAndDSTPayoffCalculator(OptionContract const& contract, SwiftParameters const& params) : PayoffCalculator(contract, params) { }

std::vector<double> DCTAndDSTPayoffCalculator::GetNonKPayoffCoefs(bool is_call) const
{
    int i,j;
    double sup = GetBoundedTo(m_params.m_payoff_to, is_call);
    double inf = GetBoundedFrom(m_params.m_payoff_from, is_call);

    double *in,*out,*in2,*out2;
    fftw_plan plan1;
    fftw_plan plan2;

    double ea = exp(inf);
    double eb = exp(sup);

    double * d = (double*) fftw_malloc(sizeof(double)*m_params.m_J_payoff);
    double * e = (double*) fftw_malloc(sizeof(double)*m_params.m_J_payoff);

    in = (double*) fftw_malloc(sizeof(double)*m_params.m_N_payoff);
    out = (double*) fftw_malloc(sizeof(double)*m_params.m_N_payoff);
    in2 = (double*) fftw_malloc(sizeof(double)*m_params.m_N_payoff);
    out2 = (double*) fftw_malloc(sizeof(double)*m_params.m_N_payoff);

    for(j=0;j<m_params.m_J_payoff;j++)
    {
        double Co=((2*j+1.)/(2.*m_params.m_J_payoff))*MY_PI;
        double B=(1./(Co*m_params.m_two_to_the_m));
        double A=(Co*m_params.m_two_to_the_m)/(1.+pow(Co*m_params.m_two_to_the_m,2));
        double sb=std::sin(Co*m_params.m_two_to_the_m*sup);
        double sa=std::sin(Co*m_params.m_two_to_the_m*inf);
        double cb=std::cos(Co*m_params.m_two_to_the_m*sup);
        double ca=std::cos(Co*m_params.m_two_to_the_m*inf);
        double I11=eb*sb-ea*sa+B*eb*cb-B*ea*ca;
        double I12=-eb*cb+ea*ca+B*eb*sb-B*ea*sa;
        double I21=sb-sa;
        double I22=ca-cb;
        d[j]=A*I11-B*I21;
        e[j]=A*I12-B*I22;
    }

    //Calculem amb FFT

    plan1 = fftw_plan_r2r_1d(m_params.m_N_payoff, d, out, FFTW_REDFT10, FFTW_ESTIMATE);       //Here we set which kind of transformation we want to perform
    fftw_execute(plan1);                                                             //Execution of FFT

    plan2 = fftw_plan_r2r_1d(m_params.m_N_payoff, e, out2, FFTW_RODFT10, FFTW_ESTIMATE);       //Here we set which kind of transformation we want to perform
    fftw_execute(plan2);                                                             //Execution of FFT


    std::vector<double> payoffs(m_params.m_k2-m_params.m_k1 + 1);

    //Para k=0
    payoffs[-m_params.m_k1]=(1./m_params.m_J_payoff)*(out[0]/2.);
    printf("res[%d]=%lf\n",-m_params.m_k1,payoffs[-m_params.m_k1]);

    //Para k>0

    for(i=1;i<=m_params.m_k2;i++)
    {
        payoffs[-m_params.m_k1+i]=(1./m_params.m_J_payoff)*((out[i]+out2[i-1])/2.); //out viene multiplicado por 2 !!!
        printf("res[%d]=%lf\n",-m_params.m_k1+i,payoffs[-m_params.m_k1+i]);
    }

    //Para k<0

    for(i=1;i<=-m_params.m_k1;i++)
    {
        payoffs[-m_params.m_k1-i]=(1./m_params.m_J_payoff)*((out[i]-out2[i-1])/2.); //out viene multiplicado por 2 !!!
        printf("res[%d]=%lf\n",-m_params.m_k1-i,payoffs[-m_params.m_k1-i]);
    }

    fftw_destroy_plan(plan1);                                            //Destroy plan
    fftw_destroy_plan(plan2);
    free(d);
    free(e);
    free(in);                                                            //Free memory
    free(out);                                                           //Free memory
    free(in2);                                                           //Free memory
    free(out2);

    return payoffs;
}

}