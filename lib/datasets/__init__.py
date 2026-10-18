from importlib import import_module

def load_dataset_base(name: str, args):
    mod = import_module(f"lib.datasets.{name.lower()}")
    return mod.load(args)

