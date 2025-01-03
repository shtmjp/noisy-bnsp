from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from logging import Logger
from typing import Literal

import numpy as np
import scipy

import c_func  # in noisy_bnsp/c/c_func.pyx

from .bigamma import Bigamma
from .utils import OptimizeHistory, get_mocked_logger


class BivariatePPModel(ABC):
    """Abstract class for bivariate point process model."""

    @abstractmethod
    def simulate(
        self,
        rng: np.random.Generator,
        T: float,
        full_params: np.ndarray,
        *,
        return_parent: bool,
    ) -> tuple[np.ndarray]:
        """Simulate point process model with given parameters."""

    @staticmethod
    @abstractmethod
    def convert_params(full_params: np.ndarray) -> np.ndarray:
        """Convert full parameters to parameters to be estimated."""

    @abstractmethod
    def ccf(self, u: float, params: np.ndarray) -> float:
        """(theoretical) cross correlation function."""

    @abstractmethod
    def integrated_ccf(self, r: float, params: np.ndarray) -> float:
        """Integrate cross correlation function on [-r, r]."""

    def wl(
        self,
        params: np.ndarray,
        data1: np.ndarray,
        data2: np.ndarray,
        T: float,
        r: float,
    ) -> float:
        """Calculate the Waagepetersen's type composite likelihood.

        cf. Waagepetersen, R. (2007).
        An Estimating Function Approach to Inference for Inhomogeneous Neyman-Scott Processes.
        Biometrics, 63-1, pp. 256-257.

        Args:
            params (np.ndarray): The (converted) parameters of the model.
            data1 (np.ndarray): The first component of the bivariate point process.
            data2 (np.ndarray): The second component of the bivariate point process.
            T (float): Terminal time.
            r (float): The range of the composite likelihood.

        Returns:
            float: The value of the composite likelihood.

        """
        first_term, second_term = 0.0, 0.0
        intensity1 = len(data1) / T
        intensity2 = len(data2) / T

        start_j = 0
        for i in range(len(data1)):
            if data1[i] < r or data1[i] > T - r:
                continue
            flag = 0
            for j in range(start_j, len(data2)):
                diff = data2[j] - data1[i]
                if diff >= r:
                    break
                if diff >= -r:
                    if flag == 0:
                        start_j = j
                        flag = 1
                    first_term += (
                        np.log(self.ccf(diff, params))
                        + np.log(intensity1)
                        + np.log(intensity2)
                    )

        second_term = (
            (T - 2.0 * r) * intensity1 * intensity2 * self.integrated_ccf(r, params)
        )
        return -(first_term - second_term)

    def wl_c(
        self,
        params: np.ndarray,
        data1: np.ndarray,
        data2: np.ndarray,
        T: float,
        r: float,
    ) -> float:
        """C implemented version of wl."""
        raise NotImplementedError


class NSExpModel(BivariatePPModel):
    """Noisy bivariate Neyman-Scott point process with exponential kernel.

    In the simulate() method, the noise is generated by homogenous Poisson processes.
    Note that, when estimating the parameters, the noise structure can be unknown
    (and not necessarily Poissonian).
    """

    @staticmethod
    def simulate(
        rng: np.random.Generator,
        T: float,
        full_params: np.ndarray,
        *,
        return_parent: bool,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate 2-variate NS point process with exponential kernel.

        Args:
            rng (np.random.Generator): random number generator
            T (float): Terminal time.
            full_params (np.ndarray): parameters of the model (See below for details).
            return_parent (bool): whether to return parent events or not.

        Returns:
            tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
            simulated N-S processes. if return_parent is True,
            return the parent events as well.

        """
        num_variate = 2
        lamb = full_params[0]  # intensity of the parent
        sigmas = full_params[1:3]  # number of children from each parent ~ Poi(sigma)
        taus = full_params[3:5]  # reciprocal of scale parameters of exp distribution
        nis = full_params[5:7]  # intensities of Poisson noise

        # generate parents
        # decide margin
        min_tau = min(taus)
        margin = 100 / min_tau
        parents = np.sort(rng.uniform(-margin, T, rng.poisson(lamb * (T + margin))))

        # generate children
        processes = [np.array([]) for _ in range(num_variate)]
        for i in range(num_variate):
            all_children = np.array([])
            # generate children for each event of parent
            for parent in parents:
                num_children = rng.poisson(lam=sigmas[i])
                children = (
                    rng.exponential(scale=1 / taus[i], size=num_children) + parent
                )
                # append results
                all_children = np.append(all_children, children)
            processes[i] = all_children[(all_children >= 0) * (all_children <= T)]

        # add Poisson noise
        for i in range(num_variate):
            noise = rng.uniform(0, T, rng.poisson(nis[i] * T))
            processes[i] = np.sort(np.append(processes[i], noise))

        if return_parent:
            return processes[0], processes[1], parents[parents > 0]
        return processes[0], processes[1]

    @staticmethod
    def ccf(u: float, params: np.ndarray) -> float:
        """(Theoretical) cross correlation function."""
        a, tau1, tau2 = params
        if u > 0:
            return 1 + a * (tau1 * tau2 / (tau1 + tau2)) * np.exp(-tau2 * abs(u))
        return 1 + a * (tau1 * tau2 / (tau1 + tau2)) * np.exp(-tau1 * abs(u))

    @staticmethod
    def integrated_ccf(r: float, params: np.ndarray) -> float:
        """Integrate cross correlation function on [-r, r]."""
        a, tau1, tau2 = params
        return 2 * r + a * (1 / (tau1 + tau2)) * (
            tau2 * (1 - np.exp(-tau1 * r)) + tau1 * (1 - np.exp(-tau2 * r))
        )

    @staticmethod
    def convert_params(full_params: np.ndarray) -> np.ndarray:
        """Convert full parameters to parameters to be estimated.

        Args:
            full_params (np.ndarray): full parameters.
                full_params[0]: intensity of the parent
                full_params[1:3]: number of children from each parent ~ Poi(sigma)
                full_params[3:5]: reciprocal of scale parameters of exp distribution
                full_params[5:7]: intensities of Poisson noise

        Returns:
            np.ndarray: parameters to be estimated.

        """
        lamb = full_params[0]  # intensity of the parent
        sigmas = full_params[1:3]  # number of children from each parent ~ Poi(sigma)
        taus = full_params[3:5]  # reciprocal of scale parameters of exp distribution
        nis = full_params[5:7]  # intensities of Poisson noise

        sn1 = lamb * sigmas[0] / (lamb * sigmas[0] + nis[0])
        sn2 = lamb * sigmas[1] / (lamb * sigmas[1] + nis[1])
        a = sn1 * sn2 / lamb
        return np.array([a, taus[0], taus[1]])

    def wl_c(
        self,
        params: np.ndarray,
        data1: np.ndarray,
        data2: np.ndarray,
        T: float,
        r: float,
    ) -> float:
        """C implemented version of wl."""
        return -c_func.cross_wl_exp(r, T, *params, data1, data2)


class NSGammaModel(BivariatePPModel):
    """Noisy bivariate Neyman-Scott point process with gamma kernel.

    In the simulate() method, the noise is generated by homogenous Poisson processes.
    Note that, when estimating the parameters, the noise structure can be unknown
    (and not necessarily Poissonian).
    """

    def __init__(self, bg_calc_method: Bigamma.CalcMethod = "sp") -> None:
        """Initialize the model."""
        # method to calculate bigamma density must be specified
        self.bg_calc_method: Bigamma.CalcMethod = bg_calc_method

    @staticmethod
    def simulate(
        rng: np.random.Generator,
        T: float,
        full_params: np.ndarray,
        *,
        return_parent: bool = False,
    ) -> tuple:
        """Generate 2-variate NS point process with gamma kernel.

        Args:
            rng (np.random.Generator): random number generator
            T (float): Terminal time.
            full_params (np.ndarray): parameters of the model (See below for details).
            return_parent (bool): whether to return parent events or not.

        Returns:
            tuple: simulated N-S processes.
            if return_parent is True, return the parent events as well.

        """
        num_variate = 2
        lamb = full_params[0]  # intensity of the parent
        sigmas = full_params[1:3]  # number of children from each parent ~ Poi(sigma)
        alphas = full_params[3:5]  # shape parameters of gamma distribution
        ls = full_params[5:7]  # reciprocal of scale parameters of gamma distribution
        nis = full_params[7:9]  # intensities of Poisson noise

        # generate parents
        # decide margin
        margin_1 = scipy.stats.gamma.ppf(0.9999, a=alphas[0], scale=1 / ls[0])
        margin_2 = scipy.stats.gamma.ppf(0.9999, a=alphas[1], scale=1 / ls[1])
        margin = max(margin_1, margin_2)
        parents = np.sort(rng.uniform(-margin, T, rng.poisson(lamb * (T + margin))))

        # generate children
        processes = [np.array([]) for _ in range(num_variate)]
        for i in range(num_variate):
            all_children = np.array([])
            # generate children for each event of parent
            for parent in parents:
                num_children = rng.poisson(lam=sigmas[i])
                children = (
                    rng.gamma(shape=alphas[i], scale=1 / ls[i], size=num_children)
                    + parent
                )
                # append results
                all_children = np.append(all_children, children)
            processes[i] = all_children[(all_children >= 0) * (all_children <= T)]

        # add Poisson noise
        for i in range(num_variate):
            noise = rng.uniform(0, T, rng.poisson(nis[i] * T))
            processes[i] = np.sort(np.append(processes[i], noise))
        if return_parent:
            return processes[0], processes[1], parents[parents > 0]
        return processes[0], processes[1]

    def ccf(self, u: float, params: np.ndarray) -> float:
        """(Theoretical) cross correlation function."""
        a, a1, a2, l1, l2 = params
        # Notice that the order of a1, a2, l1, l2 is different
        return 1 + a * Bigamma.bigamma_pdf(u, a2, a1, l2, l1, self.bg_calc_method)

    def integrated_ccf(self, r: float, params: np.ndarray) -> float:
        """Integrate cross correlation function on [-r, r]."""
        a, a1, a2, l1, l2 = params
        return (
            2 * r
            + a
            * scipy.integrate.quad(
                lambda x: Bigamma.bigamma_pdf(x, a2, a1, l2, l1, self.bg_calc_method),
                -r,
                r,
                points=[0],  # x=0 may be singular point for small a1, a2
            )[0]
        )

    @staticmethod
    def convert_params(full_params: np.ndarray) -> np.ndarray:
        """Convert full parameters to parameters to be estimated."""
        lamb = full_params[0]  # intensity of the parent
        sigmas = full_params[1:3]  # number of children from each parent ~ Poi(sigma)
        alphas = full_params[3:5]  # shape parameters of gamma distribution
        ls = full_params[5:7]  # reciprocal of scale parameters of gamma distribution
        nis = full_params[7:9]  # intensities of Poisson noise

        sn1 = lamb * sigmas[0] / (lamb * sigmas[0] + nis[0])
        sn2 = lamb * sigmas[1] / (lamb * sigmas[1] + nis[1])
        a = sn1 * sn2 / lamb
        return np.array([a, alphas[0], alphas[1], ls[0], ls[1]])

    def wl_c(
        self,
        params: np.ndarray,
        data1: np.ndarray,
        data2: np.ndarray,
        T: float,
        r: float,
    ) -> float:
        """C implemented version of wl."""
        return -c_func.cross_wl_gamma(r, T, *params, data1, data2)


@dataclass(frozen=True)
class EstConfig:
    """Configuration for the estimator."""

    T: float
    obj: Literal["wl", "wl_c"]  # kind of objective function
    r: float
    # will be passed to scipy.optimize.minimize()
    opt_method: Literal["Nelder-Mead"]
    initial_param: np.ndarray | Literal["true"]
    opt_bound: (
        list[tuple[float, float]] | Literal["auto"]
    )  # bound for scipy.optimize.minimize()
    true_param: np.ndarray | None = field(default=None)
    # for bigamma
    bigam_method: Bigamma.CalcMethod = field(default="sp")

    def __post_init__(self) -> None:
        """Check the configuration."""
        # initial_paramが"true"の場合はtrue_paramsを設定する
        if isinstance(self.initial_param, str):
            if (self.initial_param == "true") & hasattr(self, "true_param"):
                # frozenを貫通して無理やり値を代入するためにobject.__setattr__を使う
                object.__setattr__(self, "initial_param", self.true_param)
            else:
                msg = "initial_param is 'true', but true_param is not specified."
                raise ValueError(msg)
        if self.opt_bound == "auto":
            object.__setattr__(
                self, "opt_bound", [(0.0001, np.inf)] * len(self.initial_param)
            )


class Estimator:
    """Estimator for bivariate point process models."""

    def __init__(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        model: BivariatePPModel,
        cfg: EstConfig,
    ) -> None:
        """Initialize the estimator."""
        # check if data is sorted
        for data in [data1, data2]:
            if not np.all(data[:-1] <= data[1:]):
                msg = "data must be sorted in ascending order."
                raise ValueError(msg)
        self.data1 = data1
        self.data2 = data2
        self.cfg = cfg
        self.model = model

    # wrappers for the scipy.optimize.minimize
    def _objective_wl(self, params: np.ndarray) -> float:
        return self.model.wl(params, self.data1, self.data2, self.cfg.T, self.cfg.r)

    def _objective_wl_c(self, params: np.ndarray) -> float:
        return self.model.wl_c(params, self.data1, self.data2, self.cfg.T, self.cfg.r)

    def estimate(self, logger: Logger | None = None) -> np.ndarray:
        """Estimate parameters by the W-likelihood or C-implementation of W-likelihood.

        Args:
            logger: logger object.
            If the level is set to INFO, the estimation process will be logged.
            If DEBUG, the optimization history will also be logged.
            Note: callback function caluculates the objective function again,
            so the speed will be slower.

        Returns:
            np.ndarray: estimated parameters

        """
        # set mocked logger if not specified
        if logger is None:
            logger = get_mocked_logger("mocked_logger")

        # check config and set objective function
        if self.cfg.obj == "wl":
            objective = self._objective_wl
            logger.info("Start estimating by the W-likelihood")
            logger.info("cfg: %s", self.cfg.__dict__)
        elif self.cfg.obj == "wl_c":
            objective = self._objective_wl_c
            logger.info("Start estimating by the C-implementation of W-likelihood")
            logger.info("cfg: %s", self.cfg.__dict__)
        else:
            msg = f"Invalid objective function: {self.cfg.obj}"
            raise NotImplementedError(msg)

        # optimize
        if logger is not None:
            optimize_history = OptimizeHistory(objective, logger)

            optimize_result = scipy.optimize.minimize(
                objective,
                x0=self.cfg.initial_param,
                method=self.cfg.opt_method,
                bounds=self.cfg.opt_bound,
                callback=optimize_history.callback,
            )
            logger.info("Optimize result: %s", optimize_result.x)
        else:
            optimize_result = scipy.optimize.minimize(
                objective,
                x0=self.cfg.initial_param,
                method=self.cfg.opt_method,
                bounds=self.cfg.opt_bound,
            )
        return optimize_result.x
