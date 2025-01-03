import numpy as np
cimport numpy as cnp
cimport c_lib

"""
Emperical analysis
"""

def pair_corr_estimator(r, T, h, cnp.ndarray[double, ndim=1, mode="c"] data):
    len_of_data = len(data)
    # dataがメモリ上連続して並ぶことを保証する
    #if not data.flags['C_CONTIGUOUS']:
    #    data = np.ascontiguousarray(data)
    cdef double r_c=r, T_c=T, h_c=h
    cdef double *data_c
    data_c = <double *> data.data
    cdef int len_of_data_c = len_of_data

    # Call to C function exposed via Cython
    cdef double result = c_lib.pair_corr_estimator(r_c, T_c, h_c, len_of_data_c, data_c)

    return result

def cross_corr_estimator(r, T, h, cnp.ndarray[double, ndim=1, mode="c"] data1, cnp.ndarray[double, ndim=1, mode="c"] data2):
    len_of_data1 = len(data1)
    len_of_data2 = len(data2)

    # dataがメモリ上連続して並ぶことを保証する
    if not data1.flags['C_CONTIGUOUS']:
        data1 = np.ascontiguousarray(data1)
    if not data2.flags['C_CONTIGUOUS']:
        data2 = np.ascontiguousarray(data2)

    cdef double r_c=r, T_c=T, h_c=h
    cdef double *data1_c
    data1_c = <double *> data1.data
    cdef double *data2_c
    data2_c = <double *> data2.data
    cdef int len_of_data1_c = len_of_data1
    cdef int len_of_data2_c = len_of_data2

    cdef double result = c_lib.cross_corr_estimator(r_c, T_c, h_c, len_of_data1_c, len_of_data2_c, data1_c, data2_c)

    return result


"""
For bivariate N-S process
"""

def cross_wl_exp(r, T, a, tau1, tau2, cnp.ndarray[double, ndim=1, mode="c"] data1, cnp.ndarray[double, ndim=1, mode="c"] data2):
    len_of_data1 = len(data1)
    len_of_data2 = len(data2)

    # dataがメモリ上連続して並ぶことを保証する
    if not data1.flags['C_CONTIGUOUS']:
        data1 = np.ascontiguousarray(data1)
    if not data2.flags['C_CONTIGUOUS']:
        data2 = np.ascontiguousarray(data2)

    cdef double r_c=r, T_c=T, a_c=a, tau1_c=tau1, tau2_c=tau2
    cdef double *data1_c
    data1_c = <double *> data1.data
    cdef double *data2_c
    data2_c = <double *> data2.data
    cdef int len_of_data1_c = len_of_data1
    cdef int len_of_data2_c = len_of_data2

    cdef double result = c_lib.cross_wl_exp(r_c, T_c, a_c, tau1_c, tau2_c, len_of_data1_c, len_of_data2_c, data1_c, data2_c)

    return result

def cross_wl_gamma(r, T, a, a1, a2, l1, l2, cnp.ndarray[double, ndim=1, mode="c"] data1, cnp.ndarray[double, ndim=1, mode="c"] data2):
    len_of_data1 = len(data1)
    len_of_data2 = len(data2)

    # dataがメモリ上連続して並ぶことを保証する
    if not data1.flags['C_CONTIGUOUS']:
        data1 = np.ascontiguousarray(data1)
    if not data2.flags['C_CONTIGUOUS']:
        data2 = np.ascontiguousarray(data2)

    cdef double r_c=r, T_c=T, a_c=a, a1_c=a1, a2_c=a2, l1_c=l1, l2_c=l2
    cdef double *data1_c
    data1_c = <double *> data1.data
    cdef double *data2_c
    data2_c = <double *> data2.data
    cdef int len_of_data1_c = len_of_data1
    cdef int len_of_data2_c = len_of_data2

    cdef double result = c_lib.cross_wl_gamma(r_c, T_c, a_c, a1_c, a2_c, l1_c, l2_c, len_of_data1_c, len_of_data2_c, data1_c, data2_c)

    return result


cpdef double gamma_ccf(double u, double a, double a1, double a2, double l1, double l2):
    return c_lib.ccf_gamma(u, a, a1, a2, l1, l2)

cpdef double gamma_integrate_ccf(double r, double a, double a1, double a2, double l1, double l2):
    return c_lib.integrated_ccf_gamma(r, a, a1, a2, l1, l2)

cpdef double bigamma_pdf(double x, double a1, double a2, double l1, double l2):
    return c_lib.bigamma_pdf(x, a1, a2, l1, l2)
