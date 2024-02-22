""" MS-EQA model based on DebertaV2 Encoder model """

from transformers.models.deberta_v2 import DebertaV2ForQuestionAnswering
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from typing import Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss
import torch


class DebertaXXLForQuestionAnswering(DebertaV2ForQuestionAnswering):
    """ inherits from DebertaV2ForQuestionAnswering class """

    def __init__(self, config):
        """ DebertaV2ForQuestionAnswering constructor """
        super().__init__(config)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            start_positions: Optional[torch.Tensor] = None,
            end_positions: Optional[torch.Tensor] = None,
            sequence_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
            input_ids: input sequence tokens ids
            attention_mask: mask to avoid performing attention on padding token indices
            token_type_ids, position_ids, head_mask, inputs_embeds: NOT required

            start_positions: tensor (BATCH_SIZE, MAX_SEQ_LENGTH) with 1 where answers start (multiple)
            end_positions: tensor (BATCH_SIZE, MAX_SEQ_LENGTH) with 1 where answers end (multiple)
            sequence_ids: tensor (BATCH_SIZE, MAX_SEQ_LENGTH) with 1 if CLS or passage token, 0 otherwise

            start_positions and end_positions required to compute loss during training,
            but not used in eval mode to compute metrics
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # retrieving last_hidden_state of shape: (batch_size, sequence_length=MAX_SEQ_LENGTH)
        sequence_output = outputs[0]
        # passing now through QA prediction layers
        logits = self.qa_outputs(sequence_output)
        # splitting logits
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None

        # if are passed during training we compute loss
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split adds a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            loss_fct = BCEWithLogitsLoss(reduction='none')
            # with reduction='none' has shape (BATCH_SIZE, MAX_SEQ_LENGTH)
            start_loss = loss_fct(start_logits, start_positions.to(start_logits.dtype))
            end_loss = loss_fct(end_logits, end_positions.to(end_logits.dtype))

            # sequence_ids shape (BATCH_SIZE, MAX_SEQ_LENGTH)
            # 1 if CLS or passage token, 0 otherwise
            # 0-ing loss for tokens that are not passage tokens or CLS
            # is different to masking the logit to 0
            start_loss = torch.mul(start_loss, sequence_ids)
            end_loss = torch.mul(end_loss, sequence_ids)

            # averaging loss per batch
            number_of_valid_tokens = torch.count_nonzero(sequence_ids)
            start_loss_scalar = torch.sum(start_loss) / number_of_valid_tokens
            end_loss_scalar = torch.sum(end_loss) / number_of_valid_tokens

            total_loss = (start_loss_scalar + end_loss_scalar) / 2  # average of the two

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
