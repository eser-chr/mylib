import numpy as np
import numpy.typing as npt
from typing import Tuple

class CommonVectors:
    e1= np.array([1,0,0])
    e2= np.array([0,1,0])
    e3= np.array([0,0,1])


def _project_on_surface(pos: npt.NDArray, n: npt.NDArray):
    n = n / np.linalg.norm(n)
    return pos - np.multiply(pos @ n, n.reshape(-1, 1)).T


def _split_to_components_of_surface(
    pos: npt.NDArray, n: npt.NDArray
) -> Tuple[npt.NDArray, npt.NDArray]:
    
    n = n / np.linalg.norm(n)
    vertical_component = np.multiply(pos @ n, n.reshape(-1, 1)).T
    on_surface_component = pos-vertical_component
    return (on_surface_component, vertical_component)


def project_on_surface(pos: npt.NDArray, n: npt.NDArray):
    if pos.shape != (3,) and pos.shape[1] != 3:
        raise AssertionError(
            "Shape of position is not correct. It should either be a 3d vector or (N,3)"
        )
    if n.shape != (3,):
        raise AssertionError("Shape of n should be (3,)")

    return _project_on_surface(pos, n)