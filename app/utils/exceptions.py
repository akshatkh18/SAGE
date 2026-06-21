class SAGEBaseException(Exception):
    """Base exception for all SAGE errors."""
    pass

class DataLoadError(SAGEBaseException):
    """Raised when file upload or data reading fails."""
    pass

class ProfilingError(SAGEBaseException):
    """Raised when dataframe profiling fails."""
    pass

class InsightGenerationError(SAGEBaseException):
    """Raised when insight engine fails."""
    pass

class PreprocessingError(SAGEBaseException):
    """Raised when preprocessing pipeline fails."""
    pass

class ModelTrainingError(SAGEBaseException):
    """Raised when model arena fails."""
    pass

class ReportGenerationError(SAGEBaseException):
    """Raised when PDF report generation fails."""
    pass