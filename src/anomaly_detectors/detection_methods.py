from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Protocol

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binom

# if TYPE_CHECKING:
#     from src.anomaly_detectors.anomaly_detector import Anomaly_Detector


class DetectionMethod(Protocol):
    @abc.abstractmethod
    def __call__(self, action_log_probs: list[float]) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def tune(
        self,
        val_log_probs: list[np.ndarray],
        false_positive_rate: float,
    ) -> None:
        raise NotImplementedError


class dm_prod(DetectionMethod):

    def __call__(self, action_log_probs: list[float]) -> np.ndarray:

        if len(action_log_probs) >= 3:
            anomal_robots_found = (
                np.vstack(action_log_probs).mean(axis=0) < self.fpr_threshold
            )
        else:
            anomal_robots_found = np.zeros_like(action_log_probs[0], dtype=bool)
        return anomal_robots_found

    def tune(
        self,
        val_log_probs: list[np.ndarray],
        false_positive_rate: float,
    ) -> None:

        mean_alps = np.hstack(
            [alp.sum(axis=0) - np.log(alp.shape[0]) for alp in val_log_probs]
        )

        cum, lp, _ = plt.hist(mean_alps, density=True, cumulative=True, bins=100)
        plt.hlines(
            false_positive_rate, lp.min(), lp.max(), color="black", linestyles="--"
        )
        plt.show()
        assert type(cum) == np.ndarray
        fpr_mask = cum < false_positive_rate
        lp = lp[1:]
        self.fpr_threshold = lp[fpr_mask].max()


class dm_mean(DetectionMethod):

    def __call__(self, action_log_probs: list[float]) -> np.ndarray:

        if len(action_log_probs) >= 3:
            anomal_robots_found = (
                np.vstack(action_log_probs).mean(axis=0) < self.fpr_threshold
            )
        else:
            anomal_robots_found = np.zeros_like(action_log_probs[0], dtype=bool)
        return anomal_robots_found

    def tune(
        self,
        val_log_probs: list[np.ndarray],
        false_positive_rate: float,
    ) -> None:

        mean_alps = np.hstack([alp.mean(axis=0) for alp in val_log_probs])

        cum, lp, _ = plt.hist(mean_alps, density=True, cumulative=True, bins=100)
        plt.hlines(
            false_positive_rate, lp.min(), lp.max(), color="black", linestyles="--"
        )
        plt.show()
        assert type(cum) == np.ndarray
        fpr_mask = cum < false_positive_rate
        lp = lp[1:]
        self.fpr_threshold = lp[fpr_mask].max()


class dm_one_step(DetectionMethod):
    def __call__(self, action_log_probs: list[float]) -> np.ndarray:

        anomal_robots_found = (np.vstack(action_log_probs) < self.fpr_threshold).any(
            axis=0
        )
        return anomal_robots_found

    def tune(
        self,
        val_log_probs: list[np.ndarray],
        false_positive_rate: float,
    ) -> None:
        """Find the log prob threshold corresponding to the selected false positive rate over all log probs in the validation data set, independent of the number of steps

        Parameters
        ----------
        val_log_probs : list[np.ndarray]
            log probabilities from the validation data set
        false_positive_rate : float
            false positive rate acceptable for the validation data
        """
        stacked_log_probs = np.hstack([lp.flatten() for lp in val_log_probs])
        cum, lp, _ = plt.hist(
            stacked_log_probs, density=True, cumulative=True, bins=100
        )
        plt.hlines(
            false_positive_rate, lp.min(), lp.max(), color="black", linestyles="--"
        )
        plt.show()
        assert type(cum) == np.ndarray
        fpr_mask = cum < false_positive_rate
        lp = lp[1:]
        self.fpr_threshold = lp[fpr_mask].max()


class dm_binom(DetectionMethod):
    def __call__(self, action_log_probs: list[float]) -> np.ndarray:

        n = len(action_log_probs)
        k = (np.vstack(action_log_probs) < self.fpr_threshold).sum(axis=0)
        p_anomal = binom.pmf(k, n, self.false_positive_rate)
        anomal_robots_found = p_anomal < self.false_positive_rate
        return anomal_robots_found

    def tune(
        self,
        val_log_probs: list[np.ndarray],
        false_positive_rate: float,
    ) -> None:

        stacked_log_probs = np.hstack([lp.flatten() for lp in val_log_probs])
        cum, lp, _ = plt.hist(
            stacked_log_probs, density=True, cumulative=True, bins=100
        )
        plt.hlines(
            false_positive_rate, lp.min(), lp.max(), color="black", linestyles="--"
        )
        plt.show()
        assert type(cum) == np.ndarray
        fpr_mask = cum < false_positive_rate
        lp = lp[1:]
        self.fpr_threshold = lp[fpr_mask].max()
        self.false_positive_rate = false_positive_rate
