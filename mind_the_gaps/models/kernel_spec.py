import itertools
import operator
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import celerite
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import tinygp
from celerite2.jax.terms import Term, TermSum
from jax import jit, vmap
from tinygp.kernels.base import Kernel


@jit
def get_psd_value_jax(
    alpha_real,
    beta_real,
    alpha_complex_real,
    alpha_complex_imag,
    beta_complex_real,
    beta_complex_imag,
    omega,
):
    w2 = omega**2
    p = 0.0

    # Real terms
    p += jnp.sum(alpha_real * beta_real / (beta_real**2 + w2))

    # Complex terms
    w02 = beta_complex_real**2 + beta_complex_imag**2
    num = (
        alpha_complex_real * beta_complex_real + alpha_complex_imag * beta_complex_imag
    ) * w02 + (
        alpha_complex_real * beta_complex_real - alpha_complex_imag * beta_complex_imag
    ) * w2
    denom = w2**2 + 2.0 * (beta_complex_real**2 - beta_complex_imag**2) * w2 + w02**2
    p += jnp.sum(num / denom)

    return jnp.sqrt(2.0 / jnp.pi) * p


# Vectorized for an array of omega
get_psd_jax = vmap(get_psd_value_jax, in_axes=(None, None, None, None, None, None, 0))


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
        zeroed: bool = False,
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
        zeroed : bool, optional
            Whether the parameter should be initialized to a very small value (e.g., -1.0e-10),
            by default False
        """
        self.value = value
        self.fixed = fixed
        self.prior = prior
        self.bounds = bounds
        self.zeroed = zeroed
        if self.zeroed:
            self.fixed = True


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

    def _resolve_params(self):

        self._zeroed_params = {}
        self._fixed_params = {}
        self._sampled_params = {}

        for name, param_spec in self.parameters.items():
            if param_spec.fixed:
                if param_spec.zeroed:
                    param_spec.value = -1.0e-10
                    self._zeroed_params[name] = param_spec
                else:
                    self._fixed_params[name] = param_spec
            else:
                if param_spec.prior is None:
                    raise ValueError(
                        f"Missing prior distribution for parameter '{name}'"
                    )
                self._sampled_params[name] = param_spec

    def _get_term_fit(self):
        _params = {}
        for name, param_spec in itertools.chain(
            self._fixed_params.items(),
            self._sampled_params.items(),
        ):
            _params[name] = jnp.exp(param_spec.value)
        for name, param_spec in self._zeroed_params.items():
            _params[name] = -1.0e-10

        return self.term_class(**_params)

    def _get_term_sampled(self, i):
        _params = {}
        for name, param_spec in self._sampled_params.items():

            sample = numpyro.sample(
                f"terms[{i}]:log_{name}", param_spec.prior(*param_spec.bounds)
            )

            _params[name] = jnp.exp(sample)
        for name, param_spec in self._zeroed_params.items():
            _params[name] = param_spec.value
        for name, param_spec in self._fixed_params.items():
            _params[name] = jnp.exp(param_spec.value)

        return self.term_class(**_params)

    def get_psd(self, freq_in_hz=True, real_eps=1e-5):
        """Get the Power Spectral Density (PSD) function for the kernel.

        Parameters
        ----------
        freq_in_hz : bool, optional
            If True, the frequencies are in Hz. If False, they are in radians.
        real_eps : float, optional
            Small value to avoid division by zero, by default 1e-5

        Returns
        -------
        Callable[[jnp.ndarray], jnp.ndarray]
            A function that computes the PSD for a given frequency array.

        Raises
        ------
        NotImplementedError
            If the kernel term is not supported for PSD computation.
        """

        term = self._get_term_fit()

        if not isinstance(term, tinygp.kernels.quasisep.Celerite):
            raise NotImplementedError(
                f"PSD only supported for tinygp Celerite terms. Got: {type(term)}"
            )

        a = term.a
        b = term.b
        c = term.c
        d = term.d

        is_real = jnp.abs(b) < real_eps and jnp.abs(d) < real_eps

        if is_real:
            alpha_real = jnp.array([a])
            beta_real = jnp.array([c])
            alpha_complex_real = jnp.zeros(1)
            alpha_complex_imag = jnp.zeros(1)
            beta_complex_real = jnp.zeros(1)
            beta_complex_imag = jnp.zeros(1)
        else:
            alpha_real = jnp.zeros(1)
            beta_real = jnp.zeros(1)
            alpha_complex_real = jnp.array([a])
            alpha_complex_imag = jnp.array([b])
            beta_complex_real = jnp.array([c])
            beta_complex_imag = jnp.array([d])

        def psd_fn(freqs):
            omega = 2 * jnp.pi * freqs if freq_in_hz else freqs
            return get_psd_jax(
                alpha_real,
                beta_real,
                alpha_complex_real,
                alpha_complex_imag,
                beta_complex_real,
                beta_complex_imag,
                omega,
            )

        return psd_fn


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
        self.use_jax = False

        if issubclass(self.terms[0].term_class, Term) or issubclass(
            self.terms[0].term_class, Kernel
        ):
            self.use_jax = True
            for term in self.terms:
                term._resolve_params()

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

        if self.use_jax:
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

        if self.use_jax:
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
                if param.fixed or param.zeroed:
                    pass
                else:
                    if param.bounds is None:
                        raise ValueError(f"Non-fixed parameter is missing bounds.")
                    bounds.append(param.bounds)

        return jnp.array(bounds) if self.use_jax else np.array(bounds, dtype=np.float64)

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

    def get_kernel(self, fit=True) -> celerite.modeling.Model | Term:
        """Get the kernel for the Gaussian Process based on the kernel specification.

        Parameters
        ----------
        fit : bool, optional

        Returns
        -------
        Union[celerite.modeling.Model, Term]
            _description_
        """

        if self.use_jax:
            return self._get_jax_kernel(fit=fit)
        else:
            return self._get_celerite_kernel()

    def _get_jax_kernel(self, fit=True) -> Term:
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
        if fit:
            return self._get_jax_kernel_fit()
        else:
            return self._get_jax_kernel_sample()

    def _get_jax_kernel_fit(self) -> Term:
        terms = []
        for i, term in enumerate(self.terms):
            terms.append(term._get_term_fit())
        kernel = terms[0]
        for t in terms[1:]:
            kernel += t

        return kernel

    def _get_jax_kernel_sample(self) -> Term:
        terms = []
        for i, term in enumerate(self.terms):
            terms.append(term._get_term_sampled(i))
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

    def get_psd_from_kernel(self, freq_in_hz=True, real_eps=1e-5):
        """
        Build a callable PSD function from a tinygp Celerite kernel.

        Args:
            kernel: A tinygp kernel (sum of Celerite terms)
            freq_in_hz: If True, expects frequencies in Hz and converts to omega
            real_eps: Threshold for deciding whether b and d are "zero" (real term)

        Returns:
            A function psd(omega) or psd(f) → PSD values
        """
        kernel = self.get_kernel(fit=True)
        if not isinstance(kernel, tinygp.kernels.base.Kernel):
            raise TypeError(
                f"get_psd_from_kernel should only be called on KernelSpecs constructued from tinyGP Celerite Kernels, git type {type(kernel)}. For celerite & celerite2 kernels use kernel.get_psd directly."
            )

        def _flatten_terms(kernel):
            if isinstance(kernel, tinygp.kernels.quasisep.Sum):
                return _flatten_terms(kernel.kernel1) + _flatten_terms(kernel.kernel2)
            elif hasattr(kernel, "terms"):
                flat_terms = []
                for term in kernel.terms:
                    flat_terms.extend(_flatten_terms(term))
                return flat_terms
            else:
                return [kernel]

        terms = _flatten_terms(kernel)

        if not self.use_jax:
            raise TypeError("PSD extraction only implemented for JAX (tinygp) kernels.")

        psd_terms = [
            term.get_psd(freq_in_hz=freq_in_hz, real_eps=real_eps)
            for term in self.terms
        ]

        def composite_psd_fn(freqs):
            return sum(psd(freqs) for psd in psd_terms)

        return composite_psd_fn

    def get_term_psds(self, freq_in_hz=True, real_eps=1e-5):
        """Return a list of PSD functions, one per term."""
        if not self.use_jax:
            raise TypeError("Term-level PSDs only supported for JAX (tinygp) kernels.")
        return [
            term.get_psd(freq_in_hz=freq_in_hz, real_eps=real_eps)
            for term in self.terms
        ]
