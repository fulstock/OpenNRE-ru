import logging
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from .base_encoder import BaseEncoder

class BERTEncoder(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True, mask_entity=False):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model

        Automatically detects and supports BERT, RoBERTa, and similar models.
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.hidden_size = 768
        self.mask_entity = mask_entity

        # Auto-detect model type (BERT, RoBERTa, etc.)
        logging.info(f'Loading pre-trained model from {pretrain_path}')
        config = AutoConfig.from_pretrained(pretrain_path)
        self.model_type = config.model_type.lower()
        logging.info(f'Detected model type: {self.model_type}')

        self.bert = AutoModel.from_pretrained(pretrain_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path)

        # Get special tokens based on model type
        self._setup_special_tokens()

        # CRITICAL FIX: Set model to training mode!
        self.bert.train()

    def _setup_special_tokens(self):
        """Setup special tokens based on model type (BERT vs RoBERTa)"""
        if 'roberta' in self.model_type:
            self.cls_token = self.tokenizer.cls_token or '<s>'
            self.sep_token = self.tokenizer.sep_token or '</s>'
            self.pad_token_id = self.tokenizer.pad_token_id or 1

            # RoBERTa doesn't have [unused] tokens by default, we need to add them
            special_tokens = {
                'additional_special_tokens': [
                    '<e1>', '<e2>', '<e3>', '<e4>',  # Entity markers
                    '<e1-end>', '<e2-end>', '<e3-end>', '<e4-end>',  # Entity end markers
                    '<e-mask-1>', '<e-mask-2>'  # Entity mask tokens
                ]
            }
            num_added = self.tokenizer.add_special_tokens(special_tokens)
            if num_added > 0:
                logging.info(f'Added {num_added} special tokens for RoBERTa')
                self.bert.resize_token_embeddings(len(self.tokenizer))

            # Map to our marker system
            self.marker_tokens = ['<e1>', '<e2>', '<e3>', '<e4>']
            self.mask_tokens = ['<e-mask-1>', '<e-mask-2>']

        else:  # BERT and similar
            self.cls_token = '[CLS]'
            self.sep_token = '[SEP]'
            self.pad_token_id = 0
            self.marker_tokens = ['[unused0]', '[unused1]', '[unused2]', '[unused3]']
            self.mask_tokens = ['[unused4]', '[unused5]']

        logging.info(f'Using special tokens: CLS={self.cls_token}, SEP={self.sep_token}, PAD_ID={self.pad_token_id}')

    def forward(self, token, att_mask, pos1, pos2):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        _, x = self.bert(token, attention_mask=att_mask, return_dict=False)
        return x

    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        # Sentence -> token
        if 'text' in item:
            sentence = item['text']
            is_token = False
        else:
            sentence = item['token']
            is_token = True
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        # FIXED: Handle nested entities correctly by sorting markers by position
        # Determine which entity comes first
        head_first = pos_head[0] <= pos_tail[0]

        # Create marker list with positions and types
        # Use dynamic marker tokens based on model type
        marker0 = self.marker_tokens[0]
        marker1 = self.marker_tokens[1]
        marker2 = self.marker_tokens[2]
        marker3 = self.marker_tokens[3]

        markers = []
        if not is_token:
            markers.append((pos_head[0], marker0 if head_first else marker2, False, 'head'))
            markers.append((pos_head[1], marker1 if head_first else marker3, True, 'head'))
            markers.append((pos_tail[0], marker2 if head_first else marker0, False, 'tail'))
            markers.append((pos_tail[1], marker3 if head_first else marker1, True, 'tail'))
        else:
            markers.append((pos_head[0], marker0 if head_first else marker2, False, 'head'))
            markers.append((pos_head[1], marker1 if head_first else marker3, True, 'head'))
            markers.append((pos_tail[0], marker2 if head_first else marker0, False, 'tail'))
            markers.append((pos_tail[1], marker3 if head_first else marker1, True, 'tail'))

        # Sort markers by position (and by end markers after start markers at same position)
        markers.sort(key=lambda x: (x[0], x[2]))  # Sort by position, then by is_end

        # Build token sequence with markers
        re_tokens = [self.cls_token]
        prev_pos = 0
        pos1_marker = marker0 if head_first else marker2  # Head start marker
        pos2_marker = marker2 if head_first else marker0  # Tail start marker
        pos1 = None
        pos2 = None

        if self.mask_entity:
            # Replace markers with mask tokens
            mask1 = self.mask_tokens[0]
            mask2 = self.mask_tokens[1]
            marker_map = {
                marker0: mask1 if head_first else mask2,
                marker1: mask1 if head_first else mask2,
                marker2: mask2 if head_first else mask1,
                marker3: mask2 if head_first else mask1
            }

        for char_pos, marker_token, is_end, entity_type in markers:
            # Add text segment before this marker
            if not is_token:
                if char_pos > prev_pos:
                    segment_tokens = self.tokenizer.tokenize(sentence[prev_pos:char_pos])
                    re_tokens.extend(segment_tokens)
            else:
                if char_pos > prev_pos:
                    segment_tokens = self.tokenizer.tokenize(' '.join(sentence[prev_pos:char_pos]))
                    re_tokens.extend(segment_tokens)

            # Add marker
            if self.mask_entity and not is_end:
                re_tokens.append(marker_map[marker_token])
                if marker_token == pos1_marker:
                    pos1 = len(re_tokens) - 1
                elif marker_token == pos2_marker:
                    pos2 = len(re_tokens) - 1
            elif not self.mask_entity:
                re_tokens.append(marker_token)
                if marker_token == pos1_marker:
                    pos1 = len(re_tokens) - 1
                elif marker_token == pos2_marker:
                    pos2 = len(re_tokens) - 1

            # Update prev_pos only for end markers
            if is_end:
                prev_pos = char_pos

        # Add remaining text after last marker
        if not is_token:
            if prev_pos < len(sentence):
                segment_tokens = self.tokenizer.tokenize(sentence[prev_pos:])
                re_tokens.extend(segment_tokens)
        else:
            if prev_pos < len(sentence):
                segment_tokens = self.tokenizer.tokenize(' '.join(sentence[prev_pos:]))
                re_tokens.extend(segment_tokens)

        re_tokens.append(self.sep_token)

        # Ensure pos1 and pos2 are set
        if pos1 is None:
            pos1 = 1  # Default to position after CLS
        if pos2 is None:
            pos2 = 1

        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()

        # Padding
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(self.pad_token_id)
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask, pos1, pos2


class BERTEntityEncoder(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True, mask_entity=False):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model

        Automatically detects and supports BERT, RoBERTa, and similar models.
        Uses entity marker positions for relation classification.
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.hidden_size = 768 * 2
        self.mask_entity = mask_entity

        # Auto-detect model type (BERT, RoBERTa, etc.)
        logging.info(f'Loading pre-trained model from {pretrain_path}')
        config = AutoConfig.from_pretrained(pretrain_path)
        self.model_type = config.model_type.lower()
        logging.info(f'Detected model type: {self.model_type}')

        self.bert = AutoModel.from_pretrained(pretrain_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path)

        # Get special tokens based on model type
        self._setup_special_tokens()

        # CRITICAL FIX: Set model to training mode!
        self.bert.train()

        self.linear = nn.Linear(self.hidden_size, self.hidden_size)

    def _setup_special_tokens(self):
        """Setup special tokens based on model type (BERT vs RoBERTa)"""
        if 'roberta' in self.model_type:
            self.cls_token = self.tokenizer.cls_token or '<s>'
            self.sep_token = self.tokenizer.sep_token or '</s>'
            self.pad_token_id = self.tokenizer.pad_token_id or 1

            # RoBERTa doesn't have [unused] tokens by default, we need to add them
            special_tokens = {
                'additional_special_tokens': [
                    '<e1>', '<e2>', '<e3>', '<e4>',  # Entity markers
                    '<e1-end>', '<e2-end>', '<e3-end>', '<e4-end>',  # Entity end markers
                    '<e-mask-1>', '<e-mask-2>'  # Entity mask tokens
                ]
            }
            num_added = self.tokenizer.add_special_tokens(special_tokens)
            if num_added > 0:
                logging.info(f'Added {num_added} special tokens for RoBERTa')
                self.bert.resize_token_embeddings(len(self.tokenizer))

            # Map to our marker system
            self.marker_tokens = ['<e1>', '<e2>', '<e3>', '<e4>']
            self.mask_tokens = ['<e-mask-1>', '<e-mask-2>']

        else:  # BERT and similar
            self.cls_token = '[CLS]'
            self.sep_token = '[SEP]'
            self.pad_token_id = 0
            self.marker_tokens = ['[unused0]', '[unused1]', '[unused2]', '[unused3]']
            self.mask_tokens = ['[unused4]', '[unused5]']

        logging.info(f'Using special tokens: CLS={self.cls_token}, SEP={self.sep_token}, PAD_ID={self.pad_token_id}')

    def forward(self, token, att_mask, pos1, pos2):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
            pos1: (B, 1), position of the head entity starter
            pos2: (B, 1), position of the tail entity starter
        Return:
            (B, 2H), representations for sentences
        """
        hidden, _ = self.bert(token, attention_mask=att_mask, return_dict=False)

        # Get entity start hidden state
        onehot_head = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_tail = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_head = onehot_head.scatter_(1, pos1, 1)
        onehot_tail = onehot_tail.scatter_(1, pos2, 1)
        head_hidden = (onehot_head.unsqueeze(2) * hidden).sum(1)  # (B, H)
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden).sum(1)  # (B, H)
        x = torch.cat([head_hidden, tail_hidden], 1)  # (B, 2H)
        x = self.linear(x)
        return x

    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        # Sentence -> token
        if 'text' in item:
            sentence = item['text']
            is_token = False
        else:
            sentence = item['token']
            is_token = True
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        # FIXED: Handle nested entities correctly by sorting markers by position
        # Determine which entity comes first
        head_first = pos_head[0] <= pos_tail[0]

        # Create marker list with positions and types
        # Use dynamic marker tokens based on model type
        marker0 = self.marker_tokens[0]
        marker1 = self.marker_tokens[1]
        marker2 = self.marker_tokens[2]
        marker3 = self.marker_tokens[3]

        markers = []
        if not is_token:
            markers.append((pos_head[0], marker0 if head_first else marker2, False, 'head'))
            markers.append((pos_head[1], marker1 if head_first else marker3, True, 'head'))
            markers.append((pos_tail[0], marker2 if head_first else marker0, False, 'tail'))
            markers.append((pos_tail[1], marker3 if head_first else marker1, True, 'tail'))
        else:
            markers.append((pos_head[0], marker0 if head_first else marker2, False, 'head'))
            markers.append((pos_head[1], marker1 if head_first else marker3, True, 'head'))
            markers.append((pos_tail[0], marker2 if head_first else marker0, False, 'tail'))
            markers.append((pos_tail[1], marker3 if head_first else marker1, True, 'tail'))

        # Sort markers by position (and by end markers after start markers at same position)
        markers.sort(key=lambda x: (x[0], x[2]))  # Sort by position, then by is_end

        # Build token sequence with markers
        re_tokens = [self.cls_token]
        prev_pos = 0
        pos1_marker = marker0 if head_first else marker2  # Head start marker
        pos2_marker = marker2 if head_first else marker0  # Tail start marker
        pos1 = None
        pos2 = None

        if self.mask_entity:
            # Replace markers with mask tokens
            mask1 = self.mask_tokens[0]
            mask2 = self.mask_tokens[1]
            marker_map = {
                marker0: mask1 if head_first else mask2,
                marker1: mask1 if head_first else mask2,
                marker2: mask2 if head_first else mask1,
                marker3: mask2 if head_first else mask1
            }

        for char_pos, marker_token, is_end, entity_type in markers:
            # Add text segment before this marker
            if not is_token:
                if char_pos > prev_pos:
                    segment_tokens = self.tokenizer.tokenize(sentence[prev_pos:char_pos])
                    re_tokens.extend(segment_tokens)
            else:
                if char_pos > prev_pos:
                    segment_tokens = self.tokenizer.tokenize(' '.join(sentence[prev_pos:char_pos]))
                    re_tokens.extend(segment_tokens)

            # Add marker
            if self.mask_entity and not is_end:
                re_tokens.append(marker_map[marker_token])
                if marker_token == pos1_marker:
                    pos1 = len(re_tokens) - 1
                elif marker_token == pos2_marker:
                    pos2 = len(re_tokens) - 1
            elif not self.mask_entity:
                re_tokens.append(marker_token)
                if marker_token == pos1_marker:
                    pos1 = len(re_tokens) - 1
                elif marker_token == pos2_marker:
                    pos2 = len(re_tokens) - 1

            # Update prev_pos only for end markers
            if is_end:
                prev_pos = char_pos

        # Add remaining text after last marker
        if not is_token:
            if prev_pos < len(sentence):
                segment_tokens = self.tokenizer.tokenize(sentence[prev_pos:])
                re_tokens.extend(segment_tokens)
        else:
            if prev_pos < len(sentence):
                segment_tokens = self.tokenizer.tokenize(' '.join(sentence[prev_pos:]))
                re_tokens.extend(segment_tokens)

        re_tokens.append(self.sep_token)

        # Ensure pos1 and pos2 are set
        if pos1 is None:
            pos1 = 1  # Default to position after CLS
        if pos2 is None:
            pos2 = 1

        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        # Position
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()

        # Padding
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(self.pad_token_id)
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask, pos1, pos2
