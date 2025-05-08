from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import jax
import jax.numpy as jnp
import numpy as np
from celerite2.jax.terms import Term


@dataclass
class KernelParameterSpec:
    value: float
    fixed: bool = False
    prior: Optional[Callable[..., Any]] = None
    bounds: Optional[Tuple[float, float]] = None

    def __init__(
        self,
        value: float,
        fixed: bool = False,
        prior: Optional[Callable[..., Any]] = None,
        bounds: Optional[Tuple[float, float]] = None,
    ):
        self.value = value
        self.fixed = fixed
        self.prior = prior
        self.bounds = bounds


@dataclass
class KernelTermSpec:
    term_class: Type
    parameters: Dict[str, KernelParameterSpec]

    # init so type hints show up in vs code
    def __init__(self, term_class: Type, parameters: Dict[str, KernelParameterSpec]):
        self.term_class = term_class
        self.parameters = OrderedDict(parameters)


@dataclass
class KernelSpec:
    terms: List[KernelTermSpec]
    engine: str

    # init so type hints show up in vs code
    def __init__(self, terms: List[KernelTermSpec], engine: str):
        self.terms = terms
        self.engine = engine
        if issubclass(self.terms[0].term_class, Term):
            self.use_jax = True

    def update_params_from_array(self, array: Union[jax.Array, np.ndarray]) -> None:
        i = 0
        for term in self.terms:
            for name, param in term.parameters.items():
                if self.use_jax:
                    param.value = jnp.array(array[i])
                else:
                    param.value = float(array[i])
                i += 1

    def get_param_array(self) -> Union[np.ndarray, jax.Array]:

        values = []
        use_jax = False
        if issubclass(self.terms[0].term_class, Term):
            use_jax = True

        for term in self.terms:
            for param in term.parameters.values():
                if not param.fixed:
                    values.append(param.value)

        if use_jax:
            return jnp.array(values)
        else:
            return np.array(values, dtype=np.float64)

    def get_bounds_array(self) -> Union[np.ndarray, jax.Array]:
        bounds = []
        use_jax = False
        if issubclass(self.terms[0].term_class, Term):
            use_jax = True
        for term in self.terms:
            for param in term.parameters.values():
                if not param.fixed:
                    if param.bounds is None:
                        raise ValueError(f"Non-fixed parameter is missing bounds.")
                    bounds.append(param.bounds)

        return jnp.array(bounds) if use_jax else np.array(bounds, dtype=np.float64)

    def get_param_names(self) -> List[str]:
        names = []
        for i, term in enumerate(self.terms):
            for name, param in term.parameters.items():
                if not param.fixed:
                    # Prefix with term index to avoid name collisions
                    names.append(f"term{i}.{name}")
        return names
