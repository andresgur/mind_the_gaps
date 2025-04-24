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
        # if fit:
        #    params = params[mean_model.no_parameters :]
    elif mean_model is None:
        mean_value = None

        # mean_value =

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
