import torch.nn as nn
from transformers import AutoModelForSequenceClassification

from .hierbert import HierarchicalBert


class BERT(nn.Module):
    """BERT

    Args:
        num_classes (int): Total number of classes.
        dropout (float): The dropout rate of the word embedding. Defaults to 0.1.
        lm_weight (str): Pretrained model name or path. Defaults to 'bert-base-cased'.
    """
    def __init__(
        self,
        num_classes,
        dropout=0.1,
        lm_weight='bert-base-cased',
        hierarchical=False,
        max_segments=64,
        max_seg_length=128,
        **kwargs
    ):
        super().__init__()
        self.lm = AutoModelForSequenceClassification.from_pretrained(lm_weight,
                                                                     num_labels=num_classes,
                                                                     hidden_dropout_prob=dropout,
                                                                     torchscript=True)
        self.hierarchical = hierarchical
        if self.hierarchical:
            segment_encoder = self.lm.bert
            model_encoder = HierarchicalBert(encoder=segment_encoder,
                                             max_segments=max_segments,
                                             max_segment_length=max_seg_length)
            self.lm.bert = model_encoder

    def forward(self, input):
        input_ids = input['input_ids']
        attention_mask = input['attention_mask']
        x = self.lm(input_ids, attention_mask=attention_mask)[0] # (batch_size, num_classes)
        return {'logits': x}
