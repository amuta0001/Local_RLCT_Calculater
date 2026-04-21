import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as jtree
import optax 
from typing import NamedTuple
from dln import create_minibatches
from utils import (
    extrapolated_multiitemp_lambdahat,
    pack_params,
    param_l2_dist,
    param_lp_dist,
    unpack_params,
)

class SGLDConfig(NamedTuple):
  epsilon: float
  gamma: float
  num_steps: int
  num_chains: int = 1 
  batch_size: int = None

def mala_acceptance_probability(current_point, proposed_point, loss_and_grad_fn, step_size):
    """
    Calculate the acceptance probability for a MALA transition.

    Args:
    current_point: The current point in parameter space.
    proposed_point: The proposed point in parameter space.
    loss_and_grad_fn (function): Function to compute loss and loss gradient at a point.
    step_size (float): Step size parameter for MALA.

    Returns:
    float: Acceptance probability for the proposed transition.
    """
    # Compute the gradient of the loss at the current point
    current_loss, current_grad = loss_and_grad_fn(current_point)
    proposed_loss, proposed_grad = loss_and_grad_fn(proposed_point)

    # Compute the log of the proposal probabilities (using the Gaussian proposal distribution)
    log_q_proposed_to_current = -jnp.sum((current_point - proposed_point - (step_size * 0.5 * -proposed_grad)) ** 2) / (2 * step_size)
    log_q_current_to_proposed = -jnp.sum((proposed_point - current_point - (step_size * 0.5 * -current_grad)) ** 2) / (2 * step_size)

    # Compute the acceptance probability
    acceptance_log_prob = log_q_proposed_to_current - log_q_current_to_proposed + current_loss - proposed_loss
    return jnp.minimum(1.0, jnp.exp(acceptance_log_prob))

def run_sgld(rngkey, loss_fn, sgld_config, param_init, x_train, y_train, itemp=None, trace_batch_loss=True, compute_distance=False, compute_mala_acceptance=True, verbose=False, logging_period=200):
    num_training_data = len(x_train)
    if itemp is None:
        itemp = 1 / jnp.log(num_training_data)
    local_logprob = create_local_logposterior(
        avgnegloglikelihood_fn=loss_fn,
        num_training_data=num_training_data,
        w_init=param_init,
        gamma=sgld_config.gamma,
        itemp=itemp,
    )
    sgld_grad_fn = jax.jit(jax.value_and_grad(lambda w, x, y: -local_logprob(w, x, y), argnums=0))
    
    sgldoptim = optim_sgld(sgld_config.epsilon, rngkey)
    param = param_init
    if compute_mala_acceptance: # For memory efficiency, no need to store if not computing
        old_param = param.copy()

    loss_trace = []
    distances = []
    accept_probs = []
    opt_state = sgldoptim.init(param_init)
    t = 0
    while t < sgld_config.num_steps:
        for x_batch, y_batch in create_minibatches(x_train, y_train, batch_size=sgld_config.batch_size):

            if compute_distance: 
                distances.append(param_l2_dist(param_init, param))
            
            if trace_batch_loss:
                loss_val = loss_fn(param, x_batch, y_batch)
            else:
                loss_val = loss_fn(param, x_train, y_train)
            loss_trace.append(loss_val)
            
            if compute_mala_acceptance and t % 20 == 0: # Compute acceptance probability every 20 steps
                old_param_packed, pack_info = pack_params(old_param)
                param_packed, _ = pack_params(param)
                def grad_fn_packed(w):
                    nll, grad = sgld_grad_fn(unpack_params(w, pack_info), x_batch, y_batch)
                    grad_packed, _ = pack_params(grad)
                    return nll, grad_packed
                prob = mala_acceptance_probability(
                    old_param_packed, 
                    param_packed, 
                    grad_fn_packed, 
                    sgld_config.epsilon
                )
                accept_probs.append([t, prob])
            
            if t % logging_period == 0 and verbose:
                print(f"Step {t}, loss: {loss_val}")
            
            if jnp.isnan(loss_val) or jnp.isinf(loss_val):
                print(f"Step {t}, loss is NaN. Exiting.")
                return loss_trace, distances, accept_probs
            
            if compute_mala_acceptance:
                old_param = param.copy()

            _, grads = sgld_grad_fn(param, x_batch, y_batch)
            updates, opt_state = sgldoptim.update(grads, opt_state)
            param = optax.apply_updates(param, updates)
            t += 1
            if t >= sgld_config.num_steps:
                break
    return loss_trace, distances, accept_probs


def estimate_learning_coefficient(loss_trace, num_training_data, itemp=None, num_extrapolation=5, burn_in=0):
    """
    Estimate the local learning coefficient (RLCT) from an SGLD loss trace.

    Args:
    loss_trace: Sequence of minibatch/full-batch losses collected during SGLD.
    num_training_data: Training set size n used to define the inverse temperature.
    itemp: Inverse temperature used during SGLD. Defaults to 1 / log(n).
    num_extrapolation: Number of nearby inverse temperatures used for extrapolation.
    burn_in: Number of initial loss samples to discard before estimating RLCT.

    Returns:
    float or scipy LinregressResult: Estimated RLCT slope or the full regression result.
    """
    if itemp is None:
        itemp = 1 / jnp.log(num_training_data)
    losses = jnp.asarray(loss_trace)
    if burn_in:
        losses = losses[burn_in:]
    if losses.size == 0:
        raise ValueError("loss_trace is empty after burn-in; cannot estimate learning coefficient.")
    return extrapolated_multiitemp_lambdahat(
        losses=losses,
        n=num_training_data,
        itemp_og=float(itemp),
        num_extrapolation=num_extrapolation,
    )


def run_sgld_and_estimate_learning_coefficient(
    rngkey,
    loss_fn,
    sgld_config,
    param_init,
    x_train,
    y_train,
    itemp=None,
    trace_batch_loss=True,
    compute_distance=False,
    compute_mala_acceptance=True,
    verbose=False,
    logging_period=200,
    burn_in=0,
    num_extrapolation=5,
):
    """
    Convenience wrapper that runs SGLD and estimates the RLCT from its loss trace.
    """
    loss_trace, distances, accept_probs = run_sgld(
        rngkey=rngkey,
        loss_fn=loss_fn,
        sgld_config=sgld_config,
        param_init=param_init,
        x_train=x_train,
        y_train=y_train,
        itemp=itemp,
        trace_batch_loss=trace_batch_loss,
        compute_distance=compute_distance,
        compute_mala_acceptance=compute_mala_acceptance,
        verbose=verbose,
        logging_period=logging_period,
    )
    learning_coefficient = estimate_learning_coefficient(
        loss_trace=loss_trace,
        num_training_data=len(x_train),
        itemp=itemp,
        num_extrapolation=num_extrapolation,
        burn_in=burn_in,
    )
    return learning_coefficient, loss_trace, distances, accept_probs


def generate_rngkey_tree(key_or_seed, tree_or_treedef):
    rngseq = hk.PRNGSequence(key_or_seed)
    return jtree.tree_map(lambda _: next(rngseq), tree_or_treedef)

def optim_sgld(epsilon, rngkey_or_seed):
    @jax.jit
    def sgld_delta(g, rngkey):
        eta = jax.random.normal(rngkey, shape=g.shape) * jnp.sqrt(epsilon)
        return -epsilon * g / 2 + eta

    def init_fn(_):
        return rngkey_or_seed

    @jax.jit
    def update_fn(grads, state):
        rngkey, new_rngkey = jax.random.split(state)
        rngkey_tree = generate_rngkey_tree(rngkey, grads)
        updates = jtree.tree_map(sgld_delta, grads, rngkey_tree)
        return updates, new_rngkey
    return optax.GradientTransformation(init_fn, update_fn)


def create_local_logposterior(avgnegloglikelihood_fn, num_training_data, w_init, gamma, itemp):
    def helper(x, y):
        return jnp.sum((x - y)**2)

    def _logprior_fn(w):
        sqnorm = jax.tree_util.tree_map(helper, w, w_init)
        return jax.tree_util.tree_reduce(lambda a,b: a + b, sqnorm)

    def logprob(w, x, y):
        loglike = -num_training_data * avgnegloglikelihood_fn(w, x, y)
        logprior = -gamma / 2 * _logprior_fn(w)
        return itemp * loglike + logprior
    return logprob
