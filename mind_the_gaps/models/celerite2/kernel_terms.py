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


def returns_type(type_):
    def decorator(func):
        func._return_type = type_
        return func

    return decorator


@returns_type(Term)
def real_kernel_fn_norm(
    prior_sigma=0.1, params=None, fit=False, rng_key=None, bounds: dict = None
):
    if fit:

        a, c = params
    else:

        log_a = numpyro.sample(
            "log_a", dist.Normal(jnp.log(params[0]), prior_sigma), rng_key=rng_key
        )

        log_c = numpyro.sample(
            "log_c", dist.Normal(jnp.log(params[1]), prior_sigma), rng_key=rng_key
        )

        a = jnp.exp(log_a)
        c = jnp.exp(log_c)

    return jax_terms.RealTerm(a=a, c=c)


@returns_type(Term)
def real_kernel_fn(params=None, fit=False, rng_key=None, bounds: dict = None):
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
    if fit:

        a, c = params
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

    return jax_terms.RealTerm(a=a, c=c)


@returns_type(Term)
def complex_real_kernel_fn(
    params: jnp.array, fit: bool = False, rng_key: int = None, bounds: dict = None
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

    return jax_terms.ComplexTerm(a=a, c=c, d=d, b=0.0) + jax_terms.RealTerm(a=a2, c=c2)
