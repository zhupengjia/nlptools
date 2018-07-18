#!/usr/bin/env python

import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

class Decoder_Base(nn.Module):
    """Base class for decoders."""

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

    def forward(self, prev_output_tokens, encoder_out):
        raise NotImplementedError

    def get_normalized_probs(self, net_output, log_probs, _):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def max_positions(self):
        """Maximum input length supported by the decoder."""
        raise NotImplementedError

    def upgrade_state_dict(self, state_dict):
        return state_dict


class IncrementalDecoder(Decoder_Base):
    """Base class for incremental decoders."""

    def __init__(self, vocab):
        super().__init__(vocab)

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        raise NotImplementedError

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder incremental state.

        This should be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """
        def apply_reorder_incremental_state(module):
            if module != self and hasattr(module, 'reorder_incremental_state'):
                module.reorder_incremental_state(
                    incremental_state,
                    new_order,
                )
        self.apply(apply_reorder_incremental_state)

    def set_beam_size(self, beam_size):
        """Sets the beam size in the decoder and all children."""
        if getattr(self, '_beam_size', -1) != beam_size:
            def apply_set_beam_size(module):
                if module != self and hasattr(module, 'set_beam_size'):
                    module.set_beam_size(beam_size)
            self.apply(apply_set_beam_size)
            self._beam_size = beam_size




