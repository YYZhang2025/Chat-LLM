def get_target_tokens_num(num_params, target_param_data_ratio):
    # Given a number of model parameters and a target parameter-to-data ratio, return the target number of tokens to train on.
    return int(num_params * target_param_data_ratio)


import math


def get_target_batch_size(num_params, target_tokens, D_REF, B_REF):
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio**0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size))
    return total_batch_size


def get_target_learning_rate(model, target_param_data_ratio, base_lr):
    pass


def get_target_weight_decay(weight_decay, total_batch_size, B_REF, D_REF, targets_tokens_nums):
    return weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / targets_tokens_nums)


def get_num_scaling_params(model) -> int:
    wte = sum(p.numel() for p in model.transformer.wte.parameters())
    value_embeds = sum(p.numel() for p in model.value_embeds.parameters())
    lm_head = sum(p.numel() for p in model.lm_head.parameters())
    transformer_matrices = sum(p.numel() for p in model.transformer.h.parameters())
    scalars = model.resid_lambdas.numel() + model.x0_lambdas.numel()
    total = wte + value_embeds + lm_head + transformer_matrices + scalars
    assert total == sum(p.numel() for p in model.parameters()), "Parameter count mismatch"
    params_count = {
        "wte": wte,
        "value_embeds": value_embeds,
        "lm_head": lm_head,
        "transformer_matrices": transformer_matrices,
        "scalars": scalars,
        "total": total,
    }

    return (
        params_count["transformer_matrices"] + params_count["lm_head"]
    )  # we focus on scaling laws with respect to the "scaling parameters" which are the transformer weights and the lm head weights, excluding the token embedding and value embedding parameters which do not scale with model size in the same way and the scalar parameters which are negligible in count
