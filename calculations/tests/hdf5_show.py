import h5py
import numpy as np

def show(filename):
    def attrs(obj, indent):
        for k, v in obj.attrs.items():
            print(f"{indent}  @{k} = {v!r}")

    def visit(name, obj):
        depth = name.count("/")
        indent = "    " * depth
        label = name.split("/")[-1]
        if isinstance(obj, h5py.Group):
            print(f"{indent}{label}/")
        else:
            print(f"{indent}{label}  shape={obj.shape} dtype={obj.dtype}")
        attrs(obj, indent)

    with h5py.File(filename, "r") as f:
        print("/")
        attrs(f, "")
        f.visititems(visit)

if __name__== "__main__":
    show('bs_completeness.hdf5')