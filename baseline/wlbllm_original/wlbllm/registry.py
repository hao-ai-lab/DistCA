# Hack to pass the information into FlashAttention.forward.
_registry = {}

def set(key, value):
    _registry[key] = value
    return

def get(key):
    return _registry[key]

def clear():
    _registry.clear()
    return