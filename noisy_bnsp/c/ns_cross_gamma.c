#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_integration.h>
#include <stdio.h>

double bigamma_pdf_pos(double x, double a1, double a2, double l1, double l2)
{
    // clang-format off
    return pow(l1, a1) * pow(l2, a2) / gsl_sf_gamma(a1) * (
        gsl_sf_gamma(1 - (a1+a2)) / gsl_sf_gamma(1 - a1)
        * pow(x, a1+a2-1) * gsl_sf_exp(-x*l1)
        * gsl_sf_hyperg_1F1(a2, a1+a2, x*(l1+l2))
        + pow(l1 + l2, 1 - (a1 + a2))
        * gsl_sf_gamma(a1+a2-1) / gsl_sf_gamma(a2)
        * gsl_sf_exp(-x*l1)
        * gsl_sf_hyperg_1F1(1-a1, 2-(a1+a2), x*(l1+l2))
    );
    // clang-format on
}

double bigamma_pdf(double x, double a1, double a2, double l1, double l2)
{
    if (x > 0)
    {
        return bigamma_pdf_pos(x, a1, a2, l1, l2);
    }
    else
    {
        return bigamma_pdf_pos(-x, a2, a1, l2, l1);
    }
}

double ccf_gamma(double u, double a, double a1, double a2, double l1, double l2)
{
    return 1 + a * bigamma_pdf(u, a2, a1, l2, l1);
}

/* Calculate the integral of bilateral gamma density */
struct bigamma_params
{
    double a1;
    double a2;
    double l1;
    double l2;
};

double bigamma_pdf_wrapper(double x, void *params)
{
    struct bigamma_params *p = (struct bigamma_params *)params;
    return bigamma_pdf(x, p->a1, p->a2, p->l1, p->l2);
}

// FIXME: for large r, this integral often diverges and kill jupyter kernel when using Python wrapper
double bigamma_integrate(double start, double end, double a1, double a2, double l1, double l2)
{
    // FIXME: now start must be < 0 and end must be > 0
    int ws_size = 10000;
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(ws_size);

    double result, error;
    double singular_points[] = {start, 0.0, end};
    double len_of_singular_points = 3;
    struct bigamma_params params = {a1, a2, l1, l2};
    gsl_function F;
    F.function = &bigamma_pdf_wrapper;
    F.params = &params;

    gsl_integration_qagp(&F, singular_points, len_of_singular_points, 0, 1e-7, ws_size,
                         w, &result, &error);

    // printf("result          = % .18f\n", result);
    // printf("estimated error = % .18f\n", error);
    // printf("intervals       = %zu\n", w->size);

    gsl_integration_workspace_free(w);

    // 積分は理論上1を超えない
    if (result > 1)
    {
        result = 1;
    }
    return result;
}

// integrate ccf from -r to r
// FIXME: for large r, this integral often diverges and kill jupyter kernel when using Python wrapper
double integrated_ccf_gamma(double r, double a, double a1, double a2, double l1, double l2)
{
    return 2 * r + a * bigamma_integrate(-r, r, a2, a1, l2, l1);
}

double cross_wl_gamma(double r, double T, double a, double a1, double a2, double l1, double l2, int len_of_data1, int len_of_data2, double data1[], double data2[])
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
                first_term += log(ccf_gamma(diff, a, a1, a2, l1, l2) * lambda1 * lambda2);
            }
        }
    }
    second_term = (T - 2 * r) * lambda1 * lambda2 * integrated_ccf_gamma(r, a, a1, a2, l1, l2);

    return first_term - second_term;
}
