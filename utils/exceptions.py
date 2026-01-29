"""
Centralized exception definitions for the training system.

All failures should be explicit, typed, and meaningful.
No silent or generic RuntimeErrors allowed.
"""


class TrainingException(RuntimeError):
    """
    Base class for all training-related exceptions.
    """

    exit_code = 1
    status = "FAILED_UNKNOWN"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    # -------- REPRESENTATION LAYER --------

    def __str__(self) -> str:
        """
        Human-readable form (used in console, UI, logs).
        """
        return f"[{self.status}] {self.message}"

    def __repr__(self) -> str:
        """
        Developer/debug form (used in tracebacks, debug logs).
        """
        return (
            f"{self.__class__.__name__}"
            f"(status={self.status!r}, message={self.message!r})"
        )

    def to_dict(self) -> dict:
        """
        Machine-readable form (future dashboards / JSON logs).
        """
        return {
            "type": self.__class__.__name__,
            "status": self.status,
            "message": self.message,
            "exit_code": self.exit_code,
        }


# -------------------------------------------------
# CONFIG / SETUP ERRORS
# -------------------------------------------------


class ConfigError(TrainingException):
    status = "FAILED_CONFIG"


class EnvironmentError(TrainingException):
    status = "FAILED_ENVIRONMENT"


# -------------------------------------------------
# DATASET ERRORS
# -------------------------------------------------


class DatasetError(TrainingException):
    status = "FAILED_DATASET"


class DatasetStructureError(DatasetError):
    pass


class DatasetIntegrityError(DatasetError):
    pass


class DatasetLabelError(DatasetError):
    pass


# -------------------------------------------------
# TRAINING ERRORS
# -------------------------------------------------


class TrainingRuntimeError(TrainingException):
    status = "FAILED_RUNTIME"


class OOMError(TrainingRuntimeError):
    status = "FAILED_OOM"


class TrainingCrashError(TrainingRuntimeError):
    pass


# -------------------------------------------------
# POST-TRAINING ERRORS
# -------------------------------------------------


class PlottingError(TrainingException):
    status = "FAILED_PLOTTING"


class MetricsError(TrainingException):
    status = "FAILED_METRICS"


# -------------------------------------------------
# UTILITIES
# -------------------------------------------------


def classify_exception(exc: Exception) -> TrainingException:
    """
    Convert unknown exceptions into typed TrainingException.
    """
    if isinstance(exc, TrainingException):
        return exc

    msg = str(exc).lower()

    if "out of memory" in msg or "cuda" in msg:
        return OOMError(str(exc))

    return TrainingCrashError(str(exc))
