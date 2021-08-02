# lm_duorat
# Raymond Li, 2020-08-31
# Copyright (c) 2020 Element AI Inc. All rights reserved.
import logging
from typing import List, Tuple

import torch
import torch.utils.data

from duorat.models.duorat import DuoRATModel
from duorat.preproc.duorat import duo_rat_decoder_batch, duo_rat_encoder_batch
from duorat.types import (
    RATPreprocItem,
    DuoRATBatch,
    DuoRATDecoderBatch,
)
from duorat.utils import registry


logger = logging.getLogger(__name__)


@registry.register("model", "LMDuoRAT")
class LMDuoRATModel(DuoRATModel):
    def compute_loss(
        self, preproc_items: List[RATPreprocItem], debug=False
    ) -> torch.Tensor:

        items = self.preproc_items_to_duorat_items(preproc_items)
        decoder_batch = duo_rat_decoder_batch(
            items=tuple(item.decoder_item for item in items)
        )
        memory, output = self.forward(
            batch=DuoRATBatch(
                encoder_batch=duo_rat_encoder_batch(
                    items=tuple(item.encoder_item for item in items)
                ),
                decoder_batch=decoder_batch,
            )
        )
        assert not torch.isnan(memory).any()
        assert not torch.isnan(output).any()
        return self._compute_loss(
            memory=memory,
            output=output,
            target_key_padding_mask=decoder_batch.target_key_padding_mask,
            valid_copy_mask=decoder_batch.valid_copy_mask,
            copy_target_mask=decoder_batch.copy_target_mask,
            valid_actions_mask=decoder_batch.valid_actions_mask,
            target=decoder_batch.target,
        ).mean()

    @staticmethod
    def _get_targets_as_input(batch: DuoRATDecoderBatch) -> torch.Tensor:
        return batch.shifted_target

    @staticmethod
    def _get_memory_relations(batch: DuoRATDecoderBatch) -> torch.Tensor:
        return batch.shifted_memory_relations
