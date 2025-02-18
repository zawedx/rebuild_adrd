class MLFrame:
    def __init__(self):
        self._local_info = {}

    def set_local_info(self, key, value):
        self._local_info[key] = value

    def get_local_info(self, key):
        return self._local_info.get(key)

ml_frame = MLFrame()

import inspect

def with_local_info(func):
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        func_params = sig.parameters

        for param_name, param_value in ml_frame._local_info.items():
            if param_name in func_params and bound_args.arguments[param_name] is None:
                bound_args.arguments[param_name] = param_value

        return func(*bound_args.args, **bound_args.kwargs)
    return wrapper