#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, output_embed_dim):
        super().__init__()

        self.input_proj = nn.Linear(input_embed_dim, output_embed_dim, bias=False)
        self.output_proj = nn.Linear(2*output_embed_dim, output_embed_dim, bias=False)

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x output_embed_dim

        # x: bsz x output_embed_dim
        x = self.input_proj(input)

        # compute attention
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)

        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)

        x = F.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores


class LSTMDecoder(nn.Module):
    """LSTM decoder."""
    def __init__(
        self, embedding, encoder_output_units=512, out_embed_dim=512, hidden_size=512, 
        num_layers=1, attention=True, dropout=0.1, share_embed=False):
        super().__init__()
        self.dropout = dropout
        self.hidden_size = hidden_size

        self.embedding = embedding
        num_embeddings = self.embedding.num_embeddings
        embed_dim = self.embedding.embedding_dim
        

        self.encoder_output_units = encoder_output_units
        assert encoder_output_units == hidden_size, \
            'encoder_output_units ({}) != hidden_size ({})'.format(encoder_output_units, hidden_size)
        # TODO another Linear layer if not equal

        self.layers = nn.ModuleList([
            nn.LSTMCell(
                input_size=encoder_output_units + embed_dim if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])
        self.attention = AttentionLayer(encoder_output_units, hidden_size) if attention else None
        if hidden_size != out_embed_dim:
            self.additional_fc = nn.Linear(hidden_size, out_embed_dim)
        self.fc_out = nn.Linear(out_embed_dim, num_embeddings, dropout=dropout)
        if share_embed:
            self.fc_out.weight = self.embed_tokens.weight


    def forward(self, prev_output_tokens, encoder_out, encoder_padding_mask):
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, _, _ = encoder_out[:3]
        srclen = encoder_outs.size(0)

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        _, encoder_hiddens, encoder_cells = encoder_out[:3]
        num_layers = len(self.layers)
        prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
        prev_cells = [encoder_cells[i] for i in range(num_layers)]
        input_feed = x.data.new(bsz, self.encoder_output_units).zero_()

        attn_scores = x.data.new(srclen, seqlen, bsz).zero_()
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = torch.cat((x[j, :, :], input_feed), dim=1)

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs, encoder_padding_mask)
            else:
                out = hidden
            out = F.dropout(out, p=self.dropout, training=self.training)

            # input feeding
            input_feed = out

            # save final output
            outs.append(out)

        
        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        attn_scores = attn_scores.transpose(0, 2)

        # project back to size of vocabulary
        if hasattr(self, 'additional_fc'):
            x = self.additional_fc(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc_out(x)

        return x, attn_scores


    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number



