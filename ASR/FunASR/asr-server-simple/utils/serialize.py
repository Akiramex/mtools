def to_python(obj):
    """Recursively convert numpy/torch types to JSON-serializable Python types."""
    import numpy as np

    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    if has_torch and isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    return obj
