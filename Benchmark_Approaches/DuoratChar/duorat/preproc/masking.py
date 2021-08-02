from dataclasses import dataclass, replace

import torch

from duorat.preproc.relations import (
    mask_target_relation_tensor,
    mask_memory_relation_tensor,
    TargetRelation,
    MemoryRelation,
)
from duorat.types import (
    FrozenDict,
    DuoRATDecoderBatch,
)


class MaskSamplingConfig(object):
    pass


@dataclass(order=True, frozen=True)
class UniformMaskConfig(MaskSamplingConfig):
    pass


@dataclass(order=True, frozen=True)
class BernoulliMaskConfig(MaskSamplingConfig):
    p_mask: float


@dataclass(order=True, frozen=True)
class CustomMaskConfig(MaskSamplingConfig):
    custom_target_mask: torch.Tensor


@dataclass(order=True, frozen=True)
class NoMaskConfig(MaskSamplingConfig):
    pass


class MaskSampling(object):
    pass


@dataclass(order=True, frozen=True)
class UniformMask(MaskSampling):
    target_key_padding_mask: torch.Tensor


@dataclass(order=True, frozen=True)
class BernoulliMask(MaskSampling):
    p_mask: float
    target_key_padding_mask: torch.Tensor


@dataclass(order=True, frozen=True)
class CustomMask(MaskSampling):
    custom_target_mask: torch.Tensor


@dataclass(order=True, frozen=True)
class NoMask(MaskSampling):
    pass


def mask_duo_rat_decoder_batch(
    batch: DuoRATDecoderBatch,
    action_relation_types: FrozenDict[TargetRelation, int],
    memory_relation_types: FrozenDict[MemoryRelation, int],
    mask_sampling_config: MaskSamplingConfig,
    mask_value: int,
) -> DuoRATDecoderBatch:
    if isinstance(mask_sampling_config, UniformMaskConfig):
        mask_sampling = UniformMask(
            target_key_padding_mask=batch.target_key_padding_mask
        )
    elif isinstance(mask_sampling_config, BernoulliMaskConfig):
        mask_sampling = BernoulliMask(
            p_mask=mask_sampling_config.p_mask,
            target_key_padding_mask=batch.target_key_padding_mask,
        )
    elif isinstance(mask_sampling_config, CustomMaskConfig):
        mask_sampling = CustomMask(
            custom_target_mask=mask_sampling_config.custom_target_mask
        )
    elif isinstance(mask_sampling_config, NoMaskConfig):
        mask_sampling = NoMask()
    else:
        raise ValueError(
            f"Unrecognized mask sampling configuration: {mask_sampling_config}"
        )

    mask_token_mask = _mask_token_mask_like(
        input=batch.target, mask_sampling=mask_sampling,
    )

    return replace(
        batch,
        masked_target=torch.masked_fill(
            batch.target, mask=mask_token_mask, value=mask_value,
        ),
        target_relations=mask_target_relation_tensor(
            batch.target_relations, action_relation_types, mask_token_mask
        ),
        memory_relations=mask_memory_relation_tensor(
            batch.memory_relations, memory_relation_types, mask_token_mask
        ),
    )


def _mask_token_mask_like(
    input: torch.tensor, mask_sampling: MaskSampling
) -> torch.Tensor:
    assert input.dim() == 2

    # Sample one MASK token per example, uniformly among all tokens of the example
    if isinstance(mask_sampling, UniformMask):
        assert input.shape == mask_sampling.target_key_padding_mask.shape
        return torch.zeros_like(input, dtype=torch.bool).scatter(
            dim=1,
            index=torch.multinomial(mask_sampling.target_key_padding_mask.float(), 1),
            value=True,
        )

    # Each token is independently converted to MASK with probability p_mask
    elif isinstance(mask_sampling, BernoulliMask):
        assert input.shape == mask_sampling.target_key_padding_mask.shape
        return (
            torch.bernoulli(
                torch.full_like(
                    input, fill_value=mask_sampling.p_mask, dtype=torch.float
                )
            )
            .bool()
            .masked_fill(mask=~mask_sampling.target_key_padding_mask, value=False)
        )

    # Custom mask
    elif isinstance(mask_sampling, CustomMask):
        return torch.zeros_like(input, dtype=torch.bool).scatter(
            dim=1, index=mask_sampling.custom_target_mask, value=True
        )

    # No mask
    elif isinstance(mask_sampling, NoMask):
        return torch.zeros_like(input, dtype=torch.bool)

    # Unsupported mask
    else:
        raise ValueError(
            f"Unrecognized value for mask_sampling_method: {mask_sampling}"
        )
