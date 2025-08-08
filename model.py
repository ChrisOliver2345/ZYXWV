from torch import nn
from torch_geometric.nn import GATConv
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Dict, Mapping, Optional, Tuple, Any, Union
from flash import FlashTransformerEncoderLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Gene_Transformer(nn.Module):
    def __init__(self,
                 args,
                 ntoken: int,
                 d_model: int,
                 nhead: int,
                 d_hid: int,
                 nlayers: int,
                 n_cls: int,
                 vocab: Any = None,
                 dropout: float = 0.5,
                 pad_token: str = "<pad>",
                 pad_value: int = 0,
                 n_input_bins: Optional[int] = None,
                 ):
        super().__init__()
        self.args = args
        self.d_model = d_model
        self.encoder = GeneEncoder(ntoken, d_model, padding_idx=vocab[pad_token])
        self.value_encoder = CategoryValueEncoder(n_input_bins, d_model, padding_idx=pad_value)
        self.bn = nn.BatchNorm1d(d_model, eps=6.1e-5)

        encoder_layers = FlashTransformerEncoderLayer(
            d_model,
            nhead,
            d_hid,
            dropout,
            batch_first=True,
            norm_scheme='pre',
        )

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.decoder = ExprDecoder(
            d_model=d_model,
            explicit_zero_prob=False,
            use_batch_labels=False,
        )

        self.cls_decoder = ClsDecoder(d_model=d_model, n_cls=n_cls, nlayers=2)

        self.sim = Similarity(temp=0.5)  # TODO: auto set temp
        self.creterion_cce = nn.CrossEntropyLoss()

        def _encode(
                self,
                src: Tensor,
                values: Tensor,
                src_key_padding_mask: Tensor,
                batch_labels: Optional[Tensor] = None,  # (batch,)
        ) -> Tensor:
            self._check_batch_labels(batch_labels)

            src = self.encoder(src)  # (batch, seq_len, embsize)
            self.cur_gene_token_embs = src

            values = self.value_encoder(values)  # (batch, seq_len, embsize)
            if self.input_emb_style == "scaling":
                values = values.unsqueeze(2)
                total_embs = src * values
            else:
                total_embs = src + values

            if getattr(self, "dsbn", None) is not None:
                batch_label = int(batch_labels[0].item())
                total_embs = self.dsbn(total_embs.permute(0, 2, 1), batch_label).permute(
                    0, 2, 1
                )  # the batch norm always works on dim 1
            elif getattr(self, "bn", None) is not None:
                total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

            output = self.transformer_encoder(
                total_embs, src_key_padding_mask=src_key_padding_mask
            )
            return output  # (batch, seq_len, embsize)

        def forward(
                self,
                src: Tensor,
                values: Tensor,
                src_key_padding_mask: Tensor,
                batch_labels: Optional[Tensor] = None,
                do_sample: bool = False,
        ) -> Mapping[str, Tensor]:

            transformer_output = self._encode(
                src, values, src_key_padding_mask, batch_labels
            )
            output = {}
            mlm_output = self.decoder(transformer_output)

            output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)

            cell_emb = self._get_cell_emb_from_layer(transformer_output, values)
            output["cell_emb"] = cell_emb

            # if CLS:
            output["cls_output"] = self.cls_decoder(cell_emb)  # (batch, n_cls)

            # if CCE:
            cell1 = cell_emb
            transformer_output2 = self._encode(
                src, values, src_key_padding_mask, batch_labels
            )
            cell2 = self._get_cell_emb_from_layer(transformer_output2)
            # TODO: should detach the second run cls2? Can have a try
            cos_sim = self.sim(cell1.unsqueeze(1), cell2.unsqueeze(0))  # (batch, batch)
            labels = torch.arange(cos_sim.size(0)).long().to(cell1.device)
            output["loss_cce"] = self.creterion_cce(cos_sim, labels)

            return output


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class CategoryValueEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.fc(x).squeeze(-1)  # (batch, seq_len)

        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        zero_logits = self.zero_logit(x).squeeze(-1)  # (batch, seq_len)
        zero_probs = torch.sigmoid(zero_logits)
        return dict(pred=pred_value, zero_probs=zero_probs)
        # TODO: note that the return currently is only for training. Since decoder
        # is not used in the test setting for the integration task, the eval/inference
        # logic is not implemented yet. However, remember to implement it when
        # the decoder is used in any test setting. The inference logic will need
        # to sample from the bernoulli distribution with the zero_probs.


class ClsDecoder(nn.Module):
    """
    Decoder for classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


def _get_cell_emb_from_layer(
        self, layer_output: Tensor, weights: Tensor = None
) -> Tensor:
    """
    Args:
        layer_output(:obj:`Tensor`): shape (batch, seq_len, embsize)
        weights(:obj:`Tensor`): shape (batch, seq_len), optional and only used
            when :attr:`self.cell_emb_style` is "w-pool".

    Returns:
        :obj:`Tensor`: shape (batch, embsize)
    """
    if self.cell_emb_style == "cls":
        cell_emb = layer_output[:, 0, :]  # (batch, embsize)
    elif self.cell_emb_style == "avg-pool":
        cell_emb = torch.mean(layer_output, dim=1)
    elif self.cell_emb_style == "w-pool":
        if weights is None:
            raise ValueError("weights is required when cell_emb_style is w-pool")
        if weights.dim() != 2:
            raise ValueError("weights should be 2D")
        cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
        cell_emb = F.normalize(cell_emb, p=2, dim=1)  # (batch, embsize)

    return cell_emb