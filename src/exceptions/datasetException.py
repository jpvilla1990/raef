class DatasetException(Exception):
    """
    Custom DatasetException
    """
    def __init__(self, message : str, error_code : int = None):
        super().__init__(message)
        self.error_code = error_code

    def __str__(self) -> str:
        if self.error_code:
            return f"[Error {self.error_code}]: {super().__str__()}"
        return super().__str__()