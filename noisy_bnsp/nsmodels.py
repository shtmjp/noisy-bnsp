from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scipy

import c_func

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
        # return parameters to be estimated
        pass

    @abstractmethod
    def ccf(self, u, params: np.ndarray) -> float:
        # (theoretical) cross correlation function
        pass

    @abstractmethod
    def integrated_ccf(self, r, params: np.ndarray) -> float:
        # scipy.integrate.quad(ccf, -r, r)
        # TODO: 子クラスで具体的に書かれていない場合は, ここで数値積分するようにする
        pass

    def wl(
        self,
        params: np.ndarray,
        data1: np.ndarray,
        data2: np.ndarray,
        T: float,
        r: float,
    ) -> float:
        """Calculate the Waagepetersen's type composite likelihood.
        cf. Waagepetersen, R. (2007). An estimating function approach to inference for inhomogeneous spatial point processes. Biometrika, 94(4), 877-892.

        Args:
            params (np.ndarray): The (converted) parameters of the model.
            data1 (np.ndarray): The first data.
            data2 (np.ndarray): _description_
            T (float): _description_
            r (float): _description_

        Returns:
            float: _description_

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
        # C implemented version of wl
        raise NotImplementedError


class NSExpModel(BivariatePPModel):
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
    def ccf(u, params) -> float:
        """(Theoretical) cross correlation function."""
        a, tau1, tau2 = params
        if u > 0:
            return 1 + a * (tau1 * tau2 / (tau1 + tau2)) * np.exp(-tau2 * abs(u))
        return 1 + a * (tau1 * tau2 / (tau1 + tau2)) * np.exp(-tau1 * abs(u))

    @staticmethod
    def integrated_ccf(r, params) -> float:
        """Integrate cross correlation function on [-r, r]."""
        # scipy.integrate.quad(ccf, -r, r)
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
    def __init__(self, bg_calc_method: Bigamma.CalcMethod = "sp"):
        # method to calculate bigamma density must be specified
        self.bg_calc_method: Bigamma.CalcMethod = bg_calc_method

    @staticmethod
    def simulate(
        rng: np.random.Generator,
        T: float,
        full_params: np.ndarray,
        return_parent: bool = False,
    ) -> tuple:
        """Generate 2-variate NS point process with gamma kernel.

        Args:
            rng (np.random.Generator): random number generator
            T (float): Terminal time.
            full_params (np.ndarray): parameters of the model (See below for details).
            return_parent (bool): whether to return parent events or not.

        Returns:
            tuple: simulated N-S processes. if return_parent is True, return the parent events as well.

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
        a, a1, a2, l1, l2 = params
        # 変数が入れ替わっていることに注意
        return 1 + a * Bigamma.bigamma_pdf(u, a2, a1, l2, l1, self.bg_calc_method)

    def integrated_ccf(self, r: float, params: np.ndarray) -> float:
        a, a1, a2, l1, l2 = params
        # 変数が入れ替わっていることに注意
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
        lamb = full_params[0]  # intensity of the parent
        sigmas = full_params[1:3]  # number of children from each parent ~ Poi(sigma)
        alphas = full_params[3:5]  # shape parameters of gamma distribution
        ls = full_params[5:7]  # reciprocal of scale parameters of gamma distribution
        nis = full_params[7:9]  # intensities of Poisson noise

        sn1 = lamb * sigmas[0] / (lamb * sigmas[0] + nis[0])
        sn2 = lamb * sigmas[1] / (lamb * sigmas[1] + nis[1])
        a = sn1 * sn2 / lamb
        return np.array([a, alphas[0], alphas[1], ls[0], ls[1]])

    def wl_c(self, params, data1, data2, T, r):
        return -c_func.cross_wl_gamma(r, T, *params, data1, data2)


# information to be saved for estimation
@dataclass(frozen=True)
class EstConfig:
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

    def __post_init__(self):
        # initial_paramが"true"の場合はtrue_paramsを設定する
        if isinstance(self.initial_param, str):
            if (self.initial_param == "true") & hasattr(self, "true_param"):
                # frozenを貫通して無理やり値を代入するためにobject.__setattr__を使う
                object.__setattr__(self, "initial_param", self.true_param)
            else:
                raise ValueError(
                    f"initial_param is {self.initial_param}, "
                    + "but true_param is not specified."
                )
        if self.opt_bound == "auto":
            object.__setattr__(
                self, "opt_bound", [(0.0001, np.inf)] * len(self.initial_param)
            )


class Estimator:
    # cfgはmethodごとに変わるかもしれない
    def __init__(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        model: BivariatePPModel,
        cfg: EstConfig,
    ) -> None:
        assert np.all(data1[:-1] <= data1[1:])  # data1は時系列順に並んでいる必要がある
        assert np.all(data2[:-1] <= data2[1:])  # data2は時系列順に並んでいる必要がある
        self.data1 = data1
        self.data2 = data2
        self.cfg = cfg
        self.model = model

    # W-likelihood,
    # scipy.optimizeに渡す関数の引数は1次元のndarrayだけである必要があるので、wrapperを作る
    def objective_wl(self, params: np.ndarray):
        return self.model.wl(params, self.data1, self.data2, self.cfg.T, self.cfg.r)

    def objective_wl_c(self, params: np.ndarray):
        return self.model.wl_c(params, self.data1, self.data2, self.cfg.T, self.cfg.r)

    def estimate(self, logger=None) -> np.ndarray:
        # set mocked logger if not specified
        if logger is None:
            logger = get_mocked_logger("mocked_logger")

        # check config and set objective function
        if self.cfg.obj == "wl":
            objective = self.objective_wl
            logger.info("Start estimating by the W-likelihood")
            logger.info(f"cfg: {self.cfg.__dict__}")
        elif self.cfg.obj == "wl_c":
            objective = self.objective_wl_c
            logger.info("Start estimating by the C-implementation of W-likelihood")
            logger.info(f"cfg: {self.cfg.__dict__}")
        else:
            raise NotImplementedError(f"Method {self.cfg.obj} is not implemented.")

        # optimize
        optimize_history = OptimizeHistory(objective, logger)

        optimize_result = scipy.optimize.minimize(
            objective,
            x0=self.cfg.initial_param,
            method=self.cfg.opt_method,
            bounds=self.cfg.opt_bound,
            callback=optimize_history.callback,
        )
        logger.info(f"Optimize result: {optimize_result.x}")
        return optimize_result.x
