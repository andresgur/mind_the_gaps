from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import celerite
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

    def __init__(
        self, term_class: Type, parameters: Dict[str, KernelParameterSpec], **kwargs
    ):
        self.term_class = term_class
        self.parameters = OrderedDict(parameters)
        self.extras = kwargs


@dataclass
class KernelSpec:
    terms: List[KernelTermSpec]

    def __init__(self, terms: List[KernelTermSpec]):
        self.terms = terms
        self.celerite2 = False

        if issubclass(self.terms[0].term_class, Term):
            self.celerite2 = True

    def __add__(self, other: "KernelSpec") -> "KernelSpec":
        if not isinstance(other, KernelSpec):
            raise TypeError(f"Cannot add KernelSpec with {type(other)}")

        combined_terms = self.terms + other.terms

        return KernelSpec(combined_terms)

    def update_params_from_array(self, array: Union[jax.Array, np.ndarray]) -> None:
        i = 0
        for term in self.terms:
            for name, param in term.parameters.items():
                if self.celerite2:
                    param.value = jnp.array(array[i])
                else:
                    param.value = float(array[i])
                i += 1

    def get_param_array(self) -> Union[np.ndarray, jax.Array]:

        values = []

        for term in self.terms:
            for param in term.parameters.values():
                if not param.fixed:
                    values.append(param.value)

        if self.celerite2:
            return jnp.array(values)
        else:
            return np.array(values, dtype=np.float64)

    def get_bounds_array(self) -> Union[np.ndarray, jax.Array]:
        bounds = []

        for term in self.terms:
            for param in term.parameters.values():
                if not param.fixed:
                    if param.bounds is None:
                        raise ValueError(f"Non-fixed parameter is missing bounds.")
                    bounds.append(param.bounds)

        return (
            jnp.array(bounds) if self.celerite2 else np.array(bounds, dtype=np.float64)
        )

    def get_param_names(self) -> List[str]:
        names = []
        for i, term in enumerate(self.terms):
            for name, param in term.parameters.items():
                if not param.fixed:
                    names.append(f"term{i}.{name}")
        return names

    def get_kernel(
        self, fit=True, rng_key=None
    ) -> Union[celerite.modeling.Model, Term]:

        if self.celerite2:
            return self._get_celerite2_kernel(fit=fit, rng_key=rng_key)
        else:
            return self._get_celerite_kernel()

    def _get_celerite2_kernel(self, fit=True, rng_key=None) -> Term:
        import numpyro

        rng_key = self.rng_key
        terms = []

        for i, term in enumerate(self.kernel_spec.terms):
            kwargs = {}

            for name, param_spec in term.parameters.items():
                full_name = f"terms[{i}]:log_{name}"

                if fit or param_spec.fixed:
                    val = param_spec.value
                else:
                    dist_cls = param_spec.prior
                    val = numpyro.sample(
                        full_name, dist_cls(*param_spec.bounds), rng_key=rng_key
                    )

                kwargs[name] = jnp.exp(val)
            terms.append(term.term_class(**kwargs))

        kernel = terms[0]
        for t in terms[1:]:
            kernel += t
        return kernel

    def _get_celerite_kernel(self) -> celerite.modeling.Model:
        terms = []
        bounds_dict = {}
        for i, term_spec in enumerate(self.terms):
            kwargs = {}
            for name, param_spec in term_spec.parameters.items():
                kwargs[name] = param_spec.value
                if param_spec.bounds is not None:
                    bounds_dict[name] = param_spec.bounds

            if bounds_dict:
                kwargs["bounds"] = bounds_dict

            term = term_spec.term_class(**kwargs, **term_spec.extras)
            terms.append(term)

        kernel = terms[0]
        for term in terms[1:]:
            kernel += term

        return kernel
