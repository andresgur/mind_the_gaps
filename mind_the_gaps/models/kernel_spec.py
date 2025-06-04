from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import celerite
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from celerite2.jax.terms import Term


@dataclass
class KernelParameterSpec:
    """Specification for a kernel parameter in a Gaussian Process.
    This class defines a parameter in a kernel specification, including its value,
    whether it is fixed, its prior distribution, and optional bounds.
    """

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
        """Initialize a kernel parameter specification.
        This class is used to define a parameter in a kernel specification, including its value,
        whether it is fixed, its prior distribution, and optional bounds.

        Parameters
        ----------
        value : float
            The initial value of the parameter.
        fixed : bool, optional
            Whether the parameter is fixed (not optimized), by default False
        prior : Optional[Callable[..., Any]], optional
            Prior distribution class for celerite2 kernels, by default None
        bounds : Optional[Tuple[float, float]], optional
            Bounds for the parameter, by default None
        """
        self.value = value
        self.fixed = fixed
        self.prior = prior
        self.bounds = bounds


@dataclass
class KernelTermSpec:
    """Specification for a kernel term in a Gaussian Process.
    This class defines a term in a kernel specification, including its class type,
    parameters, and any additional keyword arguments.
    Attributes
    ----------
    term_class : Type
        The class type of the kernel term (e.g., celerite2.jax.terms.RealTerm).
    parameters : Dict[str, KernelParameterSpec]
        A dictionary mapping parameter names to their specifications (KernelParameterSpec).
    """

    term_class: Type
    parameters: Dict[str, KernelParameterSpec]

    def __init__(
        self, term_class: Type, parameters: Dict[str, KernelParameterSpec], **kwargs
    ):
        """Initialize a kernel term specification.
        This class is used to define a term in a kernel specification, including its class type,
        parameters, and any additional keyword arguments.

        Parameters
        ----------
        term_class : Type
            The class type of the kernel term (e.g., celerite2.jax.terms.RealTerm).
        parameters : Dict[str, KernelParameterSpec]
            A dictionary mapping parameter names to their specifications (KernelParameterSpec).
        """
        self.term_class = term_class
        self.parameters = OrderedDict(parameters)
        self.extras = kwargs


@dataclass
class KernelSpec:
    """Specification for a kernel in a Gaussian Process.
    This class defines a kernel specification, which consists of multiple kernel terms.
    """

    terms: List[KernelTermSpec]

    def __init__(self, terms: List[KernelTermSpec]):
        """Initialize a kernel specification for a Gaussian Process.
        This class is used to define a kernel specification, which consists of multiple kernel terms.

        Parameters
        ----------
        terms : List[KernelTermSpec]
            A list of kernel term specifications (KernelTermSpec) that define the kernel.
        """
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
        """Update the values of the non-fixed parameters in the kernel specification from an array.

        Parameters
        ----------
        array : Union[jax.Array, np.ndarray]
            Array containing the new values for the non-fixed parameters.

        Raises
        ------
        ValueError
            If the length of the array does not match the number of non-fixed parameters in the kernel specification.
        """
        i = 0
        non_fixed = sum(
            len([param for name, param in term.parameters.items() if not param.fixed])
            for term in self.terms
        )
        if len(array) != non_fixed:
            raise ValueError(
                f"Array length {len(array)} does not match number of non-fixed parameters: "
                f"{non_fixed}"
            )

        if self.celerite2:
            for term in self.terms:
                for name, param in term.parameters.items():
                    if not param.fixed:
                        param.value = jnp.array(array[i])
                        i += 1
        else:
            for term in self.terms:
                for name, param in term.parameters.items():
                    if not param.fixed:

                        param.value = float(array[i])
                        i += 1

    def get_param_array(self) -> Union[np.ndarray, jax.Array]:
        """Get the values of the non-fixed parameters in the kernel specification.

        Returns
        -------
        Union[np.ndarray, jax.Array]
            Array of values for the non-fixed parameters in the kernel specification.
        """

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
        """Get the bounds for the non-fixed parameters in the kernel specification.

        Returns
        -------
        Union[np.ndarray, jax.Array]
            Array of bounds for the non-fixed parameters in the kernel specification.

        Raises
        ------
        ValueError
            If a non-fixed parameter is missing bounds.
        """
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
        """Get the names of the non_fixed parameters in the kernel specification.

        Returns
        -------
        List[str]
            List of parameter names in the format "termX.name" where X is the term index.
        """
        names = []
        for i, term in enumerate(self.terms):
            for name, param in term.parameters.items():
                if not param.fixed:
                    names.append(f"term{i}.{name}")
        return names

    def get_kernel(
        self, fit=True, rng_key=None
    ) -> Union[celerite.modeling.Model, Term]:
        """Get the kernel for the Gaussian Process based on the kernel specification.

        Parameters
        ----------
        fit : bool, optional
            For celerite2 kernels: whether the parameters are being fit or sampled, by default True
        rng_key : _type_, optional
        rng_key : jax.Array, optional
            For celerite2 kernels: Random key for sampling, by default None

        Returns
        -------
        Union[celerite.modeling.Model, Term]
            _description_
        """

        if self.celerite2:
            return self._get_celerite2_kernel(fit=fit, rng_key=rng_key)
        else:
            return self._get_celerite_kernel()

    def _get_celerite2_kernel(self, fit=True, rng_key=None) -> Term:
        """Get the celerite2 kernel for the Gaussian Process based on the kernel specification.

        Parameters
        ----------
        fit : bool, optional
            Whether the parameters are being fit or sampled, by default True
        rng_key : jax.Array, optional
            Random key for sampling, by default None

        Returns
        -------
        j_terms.Term
            The Celerite2 kernel term constructed from the kernel specification.

        Raises
        ------
        ValueError
            If a parameter in the kernel specification is not fixed and does not have a prior distribution.
        """
        terms = []

        for i, term in enumerate(self.terms):
            kwargs = {}

            for name, param_spec in term.parameters.items():
                full_name = f"terms[{i}]:log_{name}"
                if not param_spec.fixed and param_spec.prior is None:
                    raise ValueError(
                        f"Missing prior distribution for parameter '{full_name}'"
                    )
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
        """Get the celerite kernel for the Gaussian Process based on the kernel specification.

        Returns
        -------
        celerite.modeling.Model
            The Celerite kernel model constructed from the kernel specification.
        """
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
