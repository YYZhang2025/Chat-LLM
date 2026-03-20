import math

import torch
import torch.distributed as dist
import torch.nn as nn

from chat_llm.utils.common import print_master
from chat_llm.utils.dist import get_dist_info, is_ddp_initialized


@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: torch.Tensor,
    p_grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step_t: torch.Tensor,
    lr_t: torch.Tensor,
    weight_decay: torch.Tensor,
    beta1_t: torch.Tensor,
    beta2_t: torch.Tensor,
    eps_t: torch.Tensor,
):
    p.mul_(1 - lr_t * weight_decay)
    # Update running averages (lerp_ is cleaner and fuses well)
    exp_avg.lerp_(p_grad, 1 - beta1_t)
    exp_avg_sq.lerp_(p_grad.square(), 1 - beta2_t)
    # Bias corrections
    bias1 = 1 - beta1_t**step_t
    bias2 = 1 - beta2_t**step_t
    # Compute update and apply
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: torch.Tensor,  # (12, 768, 3072) - stacked gradients
    stacked_params: torch.Tensor,  # (12, 768, 3072) - stacked parameters
    momentum_buffer: torch.Tensor,  # (12, 768, 3072) - first moment buffer
    second_momentum_buffer: torch.Tensor,  # (12, 768, 1) or (12, 1, 3072) - factored second moment
    momentum_t: torch.Tensor,  # () - 0-D CPU torch.Tensor, momentum coefficient
    lr_t: torch.Tensor,  # () - 0-D CPU torch.Tensor, learning rate
    wd_t: torch.Tensor,  # () - 0-D CPU torch.Tensor, weight decay
    beta2_t: torch.Tensor,  # () - 0-D CPU tensor, beta2 for second moment
    ns_steps: int,  # 5 - number of Newton-Schulz/Polar Express iterations
    red_dim: int,  # -1 or -2 - reduction dimension for variance
) -> None:
    """
    Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_update
    All in one compiled graph to eliminate Python overhead between ops.
    Some of the constants are 0-D CPU tensors to avoid recompilation when values change.
    """

    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Polar express
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1):  # Tall matrix
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:  # Wide matrix (original math)
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    # Variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class Optimizer(torch.optim.Optimizer):
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})

        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _reduce_adamw(self, group: dict, world_size: int) -> dict:
        """Launch async reduce ops for AdamW group. Returns info dict with per-param infos."""
        param_infos = {}

        for p in group["params"]:
            grad = p.grad
            if p.numel() < 1024:
                # Why average operation?
                #
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(
                    future=future,
                    grad_slice=grad,
                    is_small=True,
                )
            else:
                assert grad.shape[0] % world_size == 0, (
                    "For large params, the first dimension must be divisible by world size"
                )

                rank_size = grad.shape[0] // world_size  # chunk size for this param
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(
                    grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True
                ).get_future()

                param_infos[p] = dict(
                    future=future,
                    grad_slice=grad_slice,
                    is_small=False,
                )

        return dict(param_infos=param_infos)

    def _reduce_muon(self, group: dict, world_size: int) -> dict:
        """Launch async reduce op for Muon group. Returns info dict."""
        params = group["params"]
        chunk_size = (len(params) + world_size - 1) // world_size
        padded_num_params = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # Stack grads and zero-pad to padded_num_params
        grad_stack = torch.stack([p.grad for p in params])
        stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
        stacked_grads[: len(params)].copy_(grad_stack)
        if len(params) < padded_num_params:
            stacked_grads[len(params) :].zero_()

        # Reduce_scatter to get this rank's chunk
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(
            grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True
        ).get_future()

        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)

    def _compute_adamw(self, group: dict, info: dict, gather_list: list, rank: int, world_size: int) -> None:
        """Wait for reduce, compute AdamW updates, launch gathers for large params."""
        param_infos = info["param_infos"]
        for p in group["params"]:
            pinfo = param_infos[p]
            pinfo["future"].wait()  # Wait for the async reduce to complete for this param

            grad_slice = pinfo["grad_slice"]
            state = self.state[p]

            # For small params, operate on full param; for large, operate on slice
            if pinfo["is_small"]:
                p_slice = p
            else:
                # For large params, the reduced slice is only a portion of the full param. We will need to all_gather later to get the full updated param back.
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size : (rank + 1) * rank_size]

            # State init
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p_slice)
                state["exp_avg_sq"] = torch.zeros_like(p_slice)
            state["step"] += 1

            # Fill 0-D tensors and run fused kernel
            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])
            adamw_step_fused(
                p_slice,
                grad_slice,
                state["exp_avg"],
                state["exp_avg_sq"],
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_wd_t,
            )

            # Large params need all_gather
            if not pinfo["is_small"]:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_list.append(dict(future=future, params=None))

    def _compute_muon(self, group: dict, info: dict, gather_list: list, rank: int) -> None:
        """Wait for reduce, compute Muon updates, launch gather."""
        info["future"].wait()
        params = group["params"]
        chunk_size = info["chunk_size"]
        grad_chunk = info["grad_chunk"]
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # How many params does this rank own?
        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))

        # Get or create group-level state
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Build output buffer for all_gather
        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)

            # Fill 0-D tensors and run fused kernel
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
            self._muon_wd_t.fill_(group["weight_decay"])
            muon_step_fused(
                grad_chunk[:num_owned],
                stacked_owned,
                state["momentum_buffer"][:num_owned],
                state["second_momentum_buffer"][:num_owned],
                self._muon_momentum_t,
                self._muon_lr_t,
                self._muon_wd_t,
                self._muon_beta2_t,
                group["ns_steps"],
                red_dim,
            )
            updated_params[:num_owned].copy_(stacked_owned)

        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()

        # Reuse stacked_grads buffer for all_gather output
        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    def _finish_gathers(self, gather_list: list) -> None:
        """Wait for all gathers and copy Muon params back."""
        for info in gather_list:
            info["future"].wait()
            if info["params"] is not None:
                # Muon: copy from stacked buffer back to individual params
                torch._foreach_copy_(
                    info["params"], list(info["stacked_params"][: len(info["params"])].unbind(0))
                )

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Phase 1: launch all async reduce ops
        # Compute Gradient
        reduce_infos: list[dict] = []
        for group in self.param_groups:
            if group["kind"] == "adamw":
                reduce_infos.append(self._reduce_adamw(group, world_size))
            elif group["kind"] == "muon":
                reduce_infos.append(self._reduce_muon(group, world_size))
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        # Phase 2: wait for reduces, compute updates, launch gathers
        gather_list: list[dict] = []
        for group, info in zip(self.param_groups, reduce_infos):
            if group["kind"] == "adamw":
                self._compute_adamw(group, info, gather_list, rank, world_size)
            elif group["kind"] == "muon":
                self._compute_muon(group, info, gather_list, rank)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        # Phase 3: wait for gathers, copy back
        self._finish_gathers(gather_list)


from chat_llm.model.llm import LLMModel


def set_optimizer(
    model: LLMModel,
    un_embedding_lr: float = 0.004,
    embedding_lr: float = 0.02,
    matrix_lr: float = 0.02,
    weight_decay: float = 0.0,
    scalar_lr: float = 0.02,
    **kwargs,
):
    model_dim = model.config.embed_dim
    value_embeds_params = list(model.value_embeds.parameters())
    embedding_params = list(model.transformer.wte.parameters())
    lm_head_params = list(model.lm_head.parameters())
    x0_params = [model.x0_lambdas]
    resid_params = [model.resid_lambdas]

    matrix_params = list(model.transformer.h.parameters())

    assert len(list(model.parameters())) == len(matrix_params) + len(embedding_params) + len(
        lm_head_params
    ) + len(resid_params) + len(x0_params) + len(value_embeds_params), (
        f"Parameter count mismatch, with {len(list(model.parameters()))} parameters in the model but {len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(resid_params) + len(x0_params) + len(value_embeds_params)} parameters in the optimizer groups"
    )

    dmodel_lr_scale = (model_dim / 768) ** -0.5
    print_master(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

    # Build param_groups with all required fields explicit
    param_groups = [
        # AdamW groups (embeddings, lm_head, scalars)
        dict(
            kind="adamw",
            params=lm_head_params,
            lr=un_embedding_lr * dmodel_lr_scale,
            betas=(0.8, 0.96),
            eps=1e-10,
            weight_decay=0.01,
        ),
        dict(
            kind="adamw",
            params=embedding_params,
            lr=embedding_lr * dmodel_lr_scale,
            betas=(0.8, 0.995),
            eps=1e-10,
            weight_decay=0.001,
        ),
        dict(
            kind="adamw",
            params=value_embeds_params,
            lr=embedding_lr * dmodel_lr_scale * 0.5,
            betas=(0.8, 0.995),
            eps=1e-10,
            weight_decay=0.01,
        ),
        dict(
            kind="adamw",
            params=resid_params,
            lr=scalar_lr * 0.01,
            betas=(0.8, 0.95),
            eps=1e-10,
            weight_decay=0.05,
        ),
        dict(
            kind="adamw", params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0
        ),  # higher beta1 for x0
    ]
    # Muon groups (matrix params, grouped by shape for stacking)
    for shape in sorted({p.shape for p in matrix_params}):
        group_params = [p for p in matrix_params if p.shape == shape]
        param_groups.append(
            dict(
                kind="muon",
                params=group_params,
                lr=matrix_lr,
                momentum=0.95,
                ns_steps=5,
                beta2=0.9,
                weight_decay=weight_decay,
            )
        )

    optimizer = Optimizer(param_groups)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    return optimizer


def optimizer_step(
    optimizer,
    scaler=None,
):
    if scaler is not None:
        scaler.unscale_(optimizer)
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()


# Momentum scheduler for Muon optimizer (warms up to 0.97 over the first 400 steps)
def get_muon_momentum(it):
    frac = min(it / 400, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.97
    return momentum


# Weight decay scheduler for Muon optimizer (cosine decay to zero over the course of training)
def get_weight_decay(it, num_iterations, weight_decay_scaled):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))


# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it, warmup_steps, warmdown_ratio, num_iterations, final_lr_frac):
    warmup_iters = warmup_steps
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac


def update_optimizer_state(
    optimizer,
    cur_step,
    warmup_steps,
    warmdown_ratio,
    num_iterations,
    final_lr_frac,
    weight_decay_scaled,
):
    lrm = get_lr_multiplier(cur_step, warmup_steps, warmdown_ratio, num_iterations, final_lr_frac)
    muon_momentum = get_muon_momentum(cur_step)
    muon_weight_decay = get_weight_decay(cur_step, num_iterations, weight_decay_scaled)

    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group["kind"] == "muon":
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
