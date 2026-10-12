"""Custom exceptions so the UI can show friendly messages instead of stack traces."""


class RecommenderError(Exception):
    """Base class for all application errors."""


class DatasetError(RecommenderError):
    """The MovieLens files are missing, unreadable, or have an unexpected format."""


class ModelNotTrainedError(RecommenderError):
    """Trained model artifacts were requested but have not been created yet."""


class UnknownMovieError(RecommenderError, LookupError):
    """A movie id or title is not present in the catalogue (or model)."""


class UnknownUserError(RecommenderError, LookupError):
    """A user id has no rating history in the training data (cold-start user)."""
