from collections import OrderedDict
from itertools import chain
from typing import Dict

import celerite2.jax.terms as jax_terms
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from celerite2.jax.terms import Term
from jax import random

from mind_the_gaps.models.celerite2.mean_terms import MeanFunction


def returns_type(type_):
    def decorator(func):
        func._return_type = type_
        return func

    return decorator


def _handle_mean(mean_model, params, fit, rng_key):
    """
    Handles the mean logic for kernel functions.

    Parameters:
    ----------
    mean_model : MeanFunction or float or jnp.ndarray
        The mean function object or a fixed mean value.
    params : jnp.array
        Array of parameters.
    fit : bool
        Whether we are in fitting mode.
    rng_key : int
        Random number generator key.

    Returns:
    -------
    tuple (mean_value, remaining_params)
        mean_value: The computed or sampled mean.
        remaining_params: The remaining parameters after extracting the mean parameters.
    """
    # if isinstance(mean_model, MeanFunction):
    #    mean_value = mean_model.compute_mean(params, fit, rng_key)
    #    if fit:
    #        return mean_value, params[mean_model.no_parameters :]

    # elif mean_model is None:
    #    return params[0], params[1:]

    if isinstance(mean_model, MeanFunction):
        mean_value = mean_model.compute_mean(params, fit, rng_key)
    else:
        raise ValueError(
            f"Expected 'mean_model' to be an instance of MeanFunction, "
            f"but got {type(mean_model).__name__} instead."
        )

    return mean_value, params


@returns_type(Term)
def real_kernel_fn(
    params=None,
    fit=False,
    rng_key=None,
    bounds: dict = None,
    mean_model: MeanFunction = None,
):
    """Create a celerite2 kernel with parameters either fixed (for optimization)
    or sampled (for MCMC).

    Parameters
    ----------
    params : _type_
        list or array of initial parameter values (not in log-space)
    fit : bool, optional
        Use fixed param values (for optimization); if False, initialize for MCMC, by default False
    rng_key : int, optional
        PRNG key for Numpyro, by default None
    bounds : dict, optional
        (min, max) bounds for parameters, by default None

    Returns
    -------
    celerite2.jax.terms.Term
        jax_terms.RealTerm(a=a, c=c)
    """

    mean_value, params = _handle_mean(mean_model, params, fit, rng_key)
    if fit:

        a, c = params  # [mean_model.no_parameters :]

    else:

        log_a = numpyro.sample(
            "log_a",
            dist.Uniform(jnp.log(bounds["a"][0]), jnp.log(bounds["a"][1])),
            rng_key=rng_key,
        )
        log_c = numpyro.sample(
            "log_c",
            dist.Uniform(jnp.log(bounds["c"][0]), jnp.log(bounds["c"][1])),
            rng_key=rng_key,
        )

        a = jnp.exp(log_a)
        c = jnp.exp(log_c)

    return jax_terms.RealTerm(a=a, c=c), mean_value


@returns_type(Term)
def complex_real_kernel_fn(
    params: jnp.array,
    fit: bool = False,
    rng_key: int = None,
    bounds: dict = None,
    mean_model: MeanFunction = None,
) -> Term:
    """Create a celerite2 kernel with parameters either fixed (for optimization)
    or sampled (for MCMC).

    Parameters
    ----------
    params : _type_
        list or array of initial parameter values (not in log-space)
    fit : bool, optional
        Use fixed param values (for optimization); if False, initialize for MCMC, by default False
    rng_key : int, optional
        PRNG key for Numpyro, by default None
    bounds : dict, optional
        (min, max) bounds for parameters, by default None

    Returns
    -------
    celerite2.jax.terms.Term
        celerite2 ComplexTerm(a=a, c=c, d=d, b=0.0) + jax_terms.RealTerm(a=a2, c=c2)
    """
    mean_value, params = _handle_mean(mean_model, params, fit, rng_key)
    if fit:
        a, c, d, a2, c2 = params
    else:

        # with numpyro.plate("kernel_plate", num_kernels):

        log_a = numpyro.sample(
            "log_a",
            dist.Uniform(jnp.log(bounds["a"][0]), jnp.log(bounds["a"][1])),
            rng_key=rng_key,
        )
        log_c = numpyro.sample(
            "log_c",
            dist.Uniform(jnp.log(bounds["c"][0]), jnp.log(bounds["c"][1])),
            rng_key=rng_key,
        )
        log_d = numpyro.sample(
            "log_d",
            dist.Uniform(jnp.log(bounds["d"][0]), jnp.log(bounds["d"][1])),
            rng_key=rng_key,
        )

        log_a2 = numpyro.sample(
            "log_a2",
            dist.Uniform(jnp.log(bounds["a2"][0]), jnp.log(bounds["a2"][1])),
            rng_key=rng_key,
        )
        log_c2 = numpyro.sample(
            "log_c2",
            dist.Uniform(jnp.log(bounds["c2"][0]), jnp.log(bounds["c2"][1])),
            rng_key=rng_key,
        )

        a = jnp.exp(log_a)
        c = jnp.exp(log_c)
        d = jnp.exp(log_d)
        a2 = jnp.exp(log_a2)
        c2 = jnp.exp(log_c2)

    return (
        jax_terms.ComplexTerm(a=a, c=c, d=d, b=0.0) + jax_terms.RealTerm(a=a2, c=c2),
        mean_value,
    )


@returns_type(Term)
def kernel_term(
    term_class,
    priors: dict,
    fit: bool = False,
    params: list = None,
    rng_key: jax.random.PRNGKey = None,
    prefix: str = "",
):
    """
    Build a celerite2.jax term using sampled or fixed parameters.

    Parameters
    ----------
    term_class : callable
        A celerite2.jax.terms class like RealTerm or ComplexTerm
    priors : dict
        Dictionary where each key is a param name, and each value is a tuple:
        (distribution constructor, bounds tuple)
    fit : bool
        Whether to use fixed params (True) or sample from priors (False)
    params : list
        If fit=True, list of fixed param values to use
    rng_key : jax.random.PRNGKey
        PRNG key for numpyro sampling
    prefix : str
        Prefix for naming sampled parameters (helps avoid name collisions)

    Returns
    -------
    term instance
        A constructed kernel term (e.g., RealTerm(a=..., c=...))
    """
    term_kwargs = {}

    if fit:
        for i, name in enumerate(priors):
            term_kwargs[name] = params[i]
    else:
        for name, (dist_class, bounds) in priors.items():
            sample_name = f"{prefix}_{name}" if prefix else name
            low, high = bounds
            dist_obj = (
                dist_class(jnp.log(low), jnp.log(high))
                if "log_" in name
                else dist_class(low, high)
            )
            val = numpyro.sample(sample_name, dist_obj, rng_key=rng_key)
            term_kwargs[name] = jnp.exp(val) if "log_" in name else val

    return term_class(**term_kwargs)
