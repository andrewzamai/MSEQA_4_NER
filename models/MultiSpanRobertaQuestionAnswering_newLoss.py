""" custom RoBERTa model for MULTI-SPAN EQA - new loss """

from typing import List, Optional, Tuple, Union

from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.models.roberta.modeling_roberta import RobertaForQuestionAnswering

import torch
from torch.nn import BCEWithLogitsLoss

class MultiSpanRobertaQuestionAnswering(RobertaForQuestionAnswering):
    """
    input_ids: indices of input sequence tokens in the vocabulary
    attention_mask: mask to avoid performing attention on padding token indices
    token_type_ids, position_ids, head_mask, inputs_embeds: NOT required
    start_positions: tensor (BATCH_SIZE, MAX_SEQ_LENGTH) with 1 where answers start
    end_positions: tensor (BATCH_SIZE, MAX_SEQ_LENGTH) with 1 where answers end
    sequence_ids: tensor (BATCH_SIZE, MAX_SEQ_LENGTH) with 1 if CLS or passage token, 0 otherwise

    start_positions and end_positions required to compute loss during training,
    but not required in eval mode to compute metrics
    """
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            start_positions: Optional[torch.FloatTensor] = None,
            end_positions: Optional[torch.FloatTensor] = None,
            sequence_ids: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
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

            ''' outer sum between start & end logits --> score(start, end) = start_logits + end_logits '''
            # why not product? because two <0 logits could give >0 score when multiplied
            # and because in inference the det. alg. sorts start-end pairs within a passage
            # by score(start,end) = start_logits + end_logits
            # For this reasons we sum all logits pair-wise
            s_e_logits = torch.einsum('ai,aj->aij', start_logits, torch.ones_like(end_logits)) + torch.einsum('ai,aj->aij', torch.ones_like(start_logits), end_logits)
            # disqualify answers where end < start (it will be done also at inference time)
            # i.e. set the lower triangular matrix to low value, excluding diagonal
            max_seq_len = s_e_logits.shape[-1]
            indices = torch.tril_indices(max_seq_len, max_seq_len, offset=-1, device=s_e_logits.device)
            s_e_logits[:, indices[0][:], indices[1][:]] = -888
            # disqualify answers where answer span is greater than max_answer_length
            # i.e. set the upper triangular matrix to low value, excluding diagonal
            MAX_ANSW_LENGTH_IN_TOK = 20
            indices_long_span = torch.triu_indices(max_seq_len, max_seq_len, offset=MAX_ANSW_LENGTH_IN_TOK, device=s_e_logits.device)
            s_e_logits[:, indices_long_span[0][:], indices_long_span[1][:]] = -777
            # disqualify answers where start=0, but end != 0 (i.e. first row of matrix)
            s_e_logits[:, 0, 1:] = -666
            # disqualify spans where either start and/or end is on an invalid token
            # sequence_ids shape (BATCH_SIZE, MAX_SEQ_LENGTH), 1 if CLS or passage token, 0 otherwise
            sequence_ids_outer = torch.einsum('bi,bj->bij', sequence_ids, sequence_ids)  # outer product
            s_e_logits = torch.where(sequence_ids_outer == 0, -1000, s_e_logits)

            # now we construct similarly the new start_end_pairs to be hit
            # we must pay attention to not mark as valid start-end pairs where:
            # - end preceeds start
            # - end too far from start
            # - not mixing start-end of different pairs, especially for near consecutive spans (e.g. in CONSULTANT)
            start_positions_indices = torch.nonzero(start_positions, as_tuple=False)
            end_positions_indices = torch.nonzero(end_positions, as_tuple=False)
            # 1st col is batch idx, 2nd col is row idx, 3rd col is batch idx again(deleted), 4th column is col idx
            s_e_pairs_indices = torch.cat((start_positions_indices, end_positions_indices), dim=-1)[:, (0, 1, -1)]
            # constructing matrix of zeros of shape (BATCH_SIZE, MAX_SEQ_LENGTH, MAX_SEQ_LENGTH)
            # FloatTensor so that in loss we don't need conversion to float16
            s_e_positions = torch.zeros((start_positions.shape[0], start_positions.shape[1], start_positions.shape[1]), device=start_positions.device, dtype=torch.float16)
            s_e_positions[s_e_pairs_indices[:, 0], s_e_pairs_indices[:, 1], s_e_pairs_indices[:, 2]] = 1

            # sequence_ids_outer already computed above
            # sequence_ids_outer = torch.einsum('bi,bj->bij', sequence_ids, sequence_ids)
            # we don't compute loss for positions where end < start
            # we will discard immediately those pairs during inference
            # and the s_e_position matrix in not symmetric, would be imp for the model to model 2 scores for same pair
            max_seq_length = sequence_ids_outer.shape[1]
            indices = torch.tril_indices(max_seq_length, max_seq_length, offset=-1, device=sequence_ids_outer.device)
            sequence_ids_outer[:, indices[0][:], indices[1][:]] = 0
            # nor for too far away start-end pairs
            indices_long_span = torch.triu_indices(max_seq_length, max_seq_length, offset=MAX_ANSW_LENGTH_IN_TOK, device=sequence_ids_outer.device)
            sequence_ids_outer[:, indices_long_span[0][:], indices_long_span[1][:]] = 0

            loss_fct = BCEWithLogitsLoss(reduction='none')
            # with reduction='none' has shape (BATCH_SIZE, MAX_SEQ_LENGTH, MAX_SEQ_LENGTH)
            start_end_pairs_loss = loss_fct(s_e_logits, s_e_positions)  # s_e_positions.to(torch.float16)

            # 0ing loss for token pairs which we don't want to compute loss on
            start_end_pairs_loss = torch.mul(start_end_pairs_loss, sequence_ids_outer)

            total_loss = torch.sum(start_end_pairs_loss) / torch.count_nonzero(sequence_ids_outer)

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
