import logging
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from .base_encoder import BaseEncoder

class BERTEncoder(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True, mask_entity=False):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.hidden_size = 768
        self.mask_entity = mask_entity
        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)

        # CRITICAL FIX: Set BERT to training mode!
        self.bert.train()

    def forward(self, token, att_mask, pos1, pos2):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        _, x = self.bert(token, attention_mask=att_mask, return_dict=False)
        #return_dict=fault is set to adapt to the new version of transformers
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
        # Format: (char_position, marker_token, is_end, entity_type)
        markers = []
        if not is_token:
            markers.append((pos_head[0], '[unused0]' if head_first else '[unused2]', False, 'head'))
            markers.append((pos_head[1], '[unused1]' if head_first else '[unused3]', True, 'head'))
            markers.append((pos_tail[0], '[unused2]' if head_first else '[unused0]', False, 'tail'))
            markers.append((pos_tail[1], '[unused3]' if head_first else '[unused1]', True, 'tail'))
        else:
            # For token-based input, convert token positions to character-like positions
            # by treating each token as a unit
            markers.append((pos_head[0], '[unused0]' if head_first else '[unused2]', False, 'head'))
            markers.append((pos_head[1], '[unused1]' if head_first else '[unused3]', True, 'head'))
            markers.append((pos_tail[0], '[unused2]' if head_first else '[unused0]', False, 'tail'))
            markers.append((pos_tail[1], '[unused3]' if head_first else '[unused1]', True, 'tail'))

        # Sort markers by position (and by end markers after start markers at same position)
        markers.sort(key=lambda x: (x[0], x[2]))  # Sort by position, then by is_end

        # Build token sequence with markers
        re_tokens = ['[CLS]']
        prev_pos = 0
        pos1_marker = '[unused0]' if head_first else '[unused2]'  # Head start marker
        pos2_marker = '[unused2]' if head_first else '[unused0]'  # Tail start marker
        pos1 = None
        pos2 = None

        if self.mask_entity:
            # Replace markers with mask tokens
            marker_map = {
                '[unused0]': '[unused4]' if head_first else '[unused5]',
                '[unused1]': '[unused4]' if head_first else '[unused5]',
                '[unused2]': '[unused5]' if head_first else '[unused4]',
                '[unused3]': '[unused5]' if head_first else '[unused4]'
            }

        for char_pos, marker_token, is_end, entity_type in markers:
            # Add text segment before this marker
            if not is_token:
                if char_pos > prev_pos:
                    segment_tokens = self.tokenizer.tokenize(sentence[prev_pos:char_pos])
                    re_tokens.extend(segment_tokens)
            else:
                # For token-based input
                if char_pos > prev_pos:
                    segment_tokens = self.tokenizer.tokenize(' '.join(sentence[prev_pos:char_pos]))
                    re_tokens.extend(segment_tokens)

            # Add marker
            if self.mask_entity and not is_end:
                # Only add mask for start markers
                re_tokens.append(marker_map[marker_token])
                # Update positions
                if marker_token == pos1_marker:
                    pos1 = len(re_tokens) - 1
                elif marker_token == pos2_marker:
                    pos2 = len(re_tokens) - 1
            elif not self.mask_entity:
                re_tokens.append(marker_token)
                # Track positions for entity start markers
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

        re_tokens.append('[SEP]')

        # Ensure pos1 and pos2 are set
        if pos1 is None:
            pos1 = 1  # Default to position after [CLS]
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
                indexed_tokens.append(0)  # 0 is id for [PAD]
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
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.hidden_size = 768 * 2
        self.mask_entity = mask_entity
        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)

        # CRITICAL FIX: Set BERT to training mode!
        # By default, from_pretrained loads in eval mode
        self.bert.train()

        self.linear = nn.Linear(self.hidden_size, self.hidden_size)

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
        # Format: (char_position, marker_token, is_end, entity_type)
        markers = []
        if not is_token:
            markers.append((pos_head[0], '[unused0]' if head_first else '[unused2]', False, 'head'))
            markers.append((pos_head[1], '[unused1]' if head_first else '[unused3]', True, 'head'))
            markers.append((pos_tail[0], '[unused2]' if head_first else '[unused0]', False, 'tail'))
            markers.append((pos_tail[1], '[unused3]' if head_first else '[unused1]', True, 'tail'))
        else:
            # For token-based input, convert token positions to character-like positions
            # by treating each token as a unit
            markers.append((pos_head[0], '[unused0]' if head_first else '[unused2]', False, 'head'))
            markers.append((pos_head[1], '[unused1]' if head_first else '[unused3]', True, 'head'))
            markers.append((pos_tail[0], '[unused2]' if head_first else '[unused0]', False, 'tail'))
            markers.append((pos_tail[1], '[unused3]' if head_first else '[unused1]', True, 'tail'))

        # Sort markers by position (and by end markers after start markers at same position)
        markers.sort(key=lambda x: (x[0], x[2]))  # Sort by position, then by is_end

        # Build token sequence with markers
        re_tokens = ['[CLS]']
        prev_pos = 0
        pos1_marker = '[unused0]' if head_first else '[unused2]'  # Head start marker
        pos2_marker = '[unused2]' if head_first else '[unused0]'  # Tail start marker
        pos1 = None
        pos2 = None

        if self.mask_entity:
            # Replace markers with mask tokens
            marker_map = {
                '[unused0]': '[unused4]' if head_first else '[unused5]',
                '[unused1]': '[unused4]' if head_first else '[unused5]',
                '[unused2]': '[unused5]' if head_first else '[unused4]',
                '[unused3]': '[unused5]' if head_first else '[unused4]'
            }

        for char_pos, marker_token, is_end, entity_type in markers:
            # Add text segment before this marker
            if not is_token:
                if char_pos > prev_pos:
                    segment_tokens = self.tokenizer.tokenize(sentence[prev_pos:char_pos])
                    re_tokens.extend(segment_tokens)
            else:
                # For token-based input
                if char_pos > prev_pos:
                    segment_tokens = self.tokenizer.tokenize(' '.join(sentence[prev_pos:char_pos]))
                    re_tokens.extend(segment_tokens)

            # Add marker
            if self.mask_entity and not is_end:
                # Only add mask for start markers
                re_tokens.append(marker_map[marker_token])
                # Update positions
                if marker_token == pos1_marker:
                    pos1 = len(re_tokens) - 1
                elif marker_token == pos2_marker:
                    pos2 = len(re_tokens) - 1
            elif not self.mask_entity:
                re_tokens.append(marker_token)
                # Track positions for entity start markers
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

        re_tokens.append('[SEP]')

        # Ensure pos1 and pos2 are set
        if pos1 is None:
            pos1 = 1  # Default to position after [CLS]
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
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask, pos1, pos2
