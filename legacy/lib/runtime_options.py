"""Environment switches used by the TEC-MoLLM backbone."""
import os


def runtime_option(name, default='', environ=None):
    """Read the PIGWN_<name> environment switch."""
    values = os.environ if environ is None else environ
    return values.get('PIGWN_' + name, default)


def gradient_checkpointing_enabled(environ=None):
    """Read the gradient-checkpointing switch."""
    value = runtime_option('ENABLE_GC', '0', environ)
    return str(value).lower() in {'1', 'true', 't', 'yes', 'y', 'on'}
