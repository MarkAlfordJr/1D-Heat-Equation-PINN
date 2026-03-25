import jax
import jax.numpy as jnp

def u_t(u_model, params, x, t):
    return jax.grad(lambda tt: u_model(params, x, tt))(t)

def u_x(u_model, params, x, t):
    return jax.grad(lambda xx: u_model(params, xx, t))(x)

def u_xx(u_model, params, x, t):
    return jax.grad(lambda xx: u_x(u_model, params, xx, t))(x)

def pde_residual(u_model, params, x, t, alpha):
    return u_t(u_model, params, x, t) - alpha * u_xx(u_model, params, x, t)

def exact_solution(x, t, alpha):
    return jnp.exp(-(jnp.pi**2) * alpha * t) * jnp.sin(jnp.pi * x)

