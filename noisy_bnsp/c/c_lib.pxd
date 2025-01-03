cdef extern from "emp.h":
    double pair_corr_estimator(double r, double T, double h, int len_of_data, double *data)
    double cross_corr_estimator(double r, double T, double h, int len_of_data1, int len_of_data2, double *data1, double *data2)

cdef extern from "ns_cross_exp.h":
    double cross_wl_exp(double r, double T, double a, double tau1, double tau2, int len_of_data1, int len_of_data2, double *data1, double *data2)

cdef extern from "ns_cross_gamma.h":
    double bigamma_pdf(double x, double a1, double a2, double l1, double l2)
    double cross_wl_gamma(double r, double T, double a, double a1, double a2, double l1, double l2, int len_of_data1, int len_of_data2, double *data1, double *data2)
    double ccf_gamma(double u, double a, double a1, double a2, double l1, double l2)
    double integrated_ccf_gamma(double r, double a, double a1, double a2, double l1, double l2)
