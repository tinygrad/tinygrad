from functools import wraps

class ErrorCode:
  INPUT = 'MODEL_INPUT'
  EXECUTION = 'MODEL_EXECUTION'
  ROLLBACK = 'MODEL_ROLLBACK'

class ModelError(Exception):
  code = ErrorCode.EXECUTION

  def __init__(self, message: str, *, code: str | None = None):
    self.code = code or self.code
    super().__init__(message)

class ModelInputError(ValueError, ModelError):
  code = ErrorCode.INPUT

  def __init__(self, message: str, *, code: str | None = None):
    ModelError.__init__(self, message, code=code)

class ModelExecutionError(ModelError):
  code = ErrorCode.EXECUTION

class ModelRollbackError(ModelError):
  code = ErrorCode.ROLLBACK

# Convert legacy ValueError at model boundaries into stable input errors.
def input_boundary(fn):
  @wraps(fn)
  def wrapped(*args, **kwargs):
    try: return fn(*args, **kwargs)
    except ModelError: raise
    except ValueError as error: raise ModelInputError(str(error)) from error
  return wrapped
