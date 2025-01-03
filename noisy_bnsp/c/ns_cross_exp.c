#include <stdlib.h>
#include <math.h>
#include <float.h>

double ccf_exp(double u, double a, double tau1, double tau2)
{
    if (u > 0)
    {
        return 1 + a * ((tau1 * tau2) / (tau1 + tau2)) * exp(-tau2 * u);
    }
    else
    {
        return 1 + a * ((tau1 * tau2) / (tau1 + tau2)) * exp(tau1 * u);
    }
}

// integrate ccf from -r to r
double integrated_ccf_exp(double r, double a, double tau1, double tau2)
{
    return 2 * r +
           (a * (tau1 / (tau1 + tau2)) * (1 - exp(-tau2 * r))) +
           (a * (tau2 / (tau1 + tau2)) * (1 - exp(-tau1 * r)));
}

double cross_wl_exp(double r, double T, double a, double tau1, double tau2, int len_of_data1, int len_of_data2, double data1[], double data2[])
{
    double first_term = 0;
    double second_term = 0;
    double diff = 0;
    int start_j = 0;
    int flag = 0;
    double lambda1 = len_of_data1 / T;
    double lambda2 = len_of_data2 / T;

    for (int i = 0; i < len_of_data1; i++)
    {
        // inner edge correction
        if (data1[i] < r || data1[i] > T - r)
        {
            continue;
        }
        flag = 0; // Set flag = 1 when you find the first j that satisfies diff >= r, and start the next search for j from start_j.
        for (int j = start_j; j < len_of_data2; j++)
        {
            diff = data2[j] - data1[i]; // 単調増加
            if (diff >= r)
            {
                break;
            }
            if (diff >= -r)
            {
                if (flag == 0)
                {
                    start_j = j;
                    flag = 1;
                }
                first_term += log(ccf_exp(diff, a, tau1, tau2)) + log(lambda1) + log(lambda2);
            }
        }
    }
    second_term = (T - 2 * r) * lambda1 * lambda2 * integrated_ccf_exp(r, a, tau1, tau2);

    return first_term - second_term;
}
