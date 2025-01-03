from typing import Literal

import numpy as np
import scipy.integrate
import scipy.special as sp


class Bigamma:
    CalcMethod = Literal["integral", "sp"]

    @staticmethod
    def bigamma_int_pos(x: float, a1: float, a2: float, l1: float, l2: float) -> float:
        def integrand(v):
            return v ** (a2 - 1) * (x + v / (l1 + l2)) ** (a1 - 1) * np.exp(-v)

        integral = scipy.integrate.quad(integrand, 0, np.inf)
        integral_val = integral[0]
        coef = (
            (l1) ** a1
            * (l2) ** a2
            * np.exp(-l1 * x)
            / ((l1 + l2) ** a2 * sp.gamma(a1) * sp.gamma(a2))
        )
        val = coef * integral_val
        # err = abs(coef) * integral[1]
        return val

    @staticmethod
    def whittaker_M(lam: float, mu: float, x: float) -> float:
        return (
            x ** (mu + 0.5)
            * np.exp(-0.5 * x)
            * sp.hyp1f1(mu - lam + 0.5, 2 * mu + 1, x)
        )

    @classmethod
    def whittaker_W(cls, lam: float, mu: float, x: float) -> float:
        return sp.gamma(-2 * mu) * cls.whittaker_M(lam, mu, x) / sp.gamma(
            0.5 - mu - lam
        ) + sp.gamma(2 * mu) * cls.whittaker_M(lam, -mu, x) / sp.gamma(0.5 + mu - lam)

    @classmethod
    def bigamma_sp_pos(
        cls, x: float, a1: float, a2: float, l1: float, l2: float
    ) -> float:
        return float(
            (l1) ** a1
            * (l2) ** a2
            / ((l1 + l2) ** ((a1 + a2) / 2) * sp.gamma(a1))
            * x ** ((a1 + a2) / 2 - 1)
            * np.exp(-x / 2 * (l1 - l2))
            * cls.whittaker_W(0.5 * (a1 - a2), 0.5 * (a1 + a2 - 1), x * (l1 + l2))
        )

    @classmethod
    def bigamma_pdf(
        cls, x: float, a1: float, a2: float, l1: float, l2: float, method: CalcMethod
    ) -> float:
        if x > 0:
            return getattr(cls, f"bigamma_{method}_pos")(x, a1, a2, l1, l2)
        if x < 0:
            return getattr(cls, f"bigamma_{method}_pos")(-x, a2, a1, l2, l1)
        return 0
        return 0
