#!/usr/bin/env python3
"""
BRAT Document-to-Sentence TACRED Converter

This script converts document-level BRAT format to sentence-level TACRED format.
It splits documents into sentences and outputs one TACRED JSON per relation,
where both entities appear in the same sentence.

Key features:
- Splits documents into sentences using NLTK
- Only outputs relations where both entities are in the same sentence
- Uses 'text' field with CHARACTER-LEVEL positions (not token-level)
- OpenNRE's BERT tokenizer handles tokenization internally

Output format:
    {"text": "sentence text", "h": {"pos": [char_start, char_end], ...}, ...}

This avoids tokenization mismatch between different tokenizers.

Usage:
    python brat2tacred-sent.py \
        --input_dir /path/to/brat/data \
        --output_file output.txt \
        --rel2id rel2id.json
"""

import os
import json
import argparse
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

# Use NLTK sentence splitter only (no word tokenization needed)
# OpenNRE's BERT tokenizer will handle tokenization
try:
    from nltk.data import load

    ru_tokenizer = load("tokenizers/punkt/russian.pickle")
except Exception as e:
    print("ERROR: NLTK punkt tokenizer not found. Install with:")
    print("  pip install nltk")
    print("  python -c 'import nltk; nltk.download(\"punkt\")'")
    exit(1)


def parse_brat_ann(ann_path: str) -> Tuple[Dict[str, Dict], List[Dict], int]:
    """
    Parse a BRAT .ann file and extract entities and relations.

    Skips discontinuous entities (e.g., "присвоил ... звание" with text in between).

    Args:
        ann_path: Path to .ann file

    Returns:
        entities: Dict mapping entity ID (T1, T2, ...) to entity info
        relations: List of relation dicts
        skipped_discontinuous: Count of discontinuous entities skipped
    """
    entities = {}
    relations = []
    skipped_discontinuous = 0

    with open(ann_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) < 2:
                continue

            ann_id = parts[0]

            # Parse entity annotation (T1, T2, ...)
            if ann_id.startswith('T'):
                ann_parts = parts[1].split()
                if len(ann_parts) < 3:
                    continue

                entity_type = ann_parts[0]
                entity_text = parts[2] if len(parts) > 2 else ""

                # Skip discontinuous entities (contain semicolon in span)
                # Example: "VERB 118 126;134 141" means "присвоил ... звание"
                # These are hard to handle correctly, so we skip them
                if ';' in parts[1]:
                    skipped_discontinuous += 1
                    continue

                # Parse continuous entity span
                try:
                    start_char = int(ann_parts[1])
                    end_char = int(ann_parts[2])
                except (ValueError, IndexError):
                    # Invalid format, skip
                    continue

                entities[ann_id] = {
                    'id': ann_id,
                    'type': entity_type,
                    'start': start_char,
                    'end': end_char,
                    'text': entity_text
                }

            # Parse relation annotation (R1, R2, ...)
            elif ann_id.startswith('R'):
                rel_parts = parts[1].split()
                if len(rel_parts) < 3:
                    continue

                rel_type = rel_parts[0]
                head_id = rel_parts[1].replace('Arg1:', '')
                tail_id = rel_parts[2].replace('Arg2:', '')

                relations.append({
                    'id': ann_id,
                    'type': rel_type,
                    'head': head_id,
                    'tail': tail_id
                })

    return entities, relations, skipped_discontinuous


# Note: We no longer need word tokenization or token span finding.
# OpenNRE's BERT tokenizer will handle this internally when we provide
# the 'text' field with character-level positions.


def split_into_sentences(text: str) -> List[Tuple[int, int, str]]:
    """
    Split text into sentences using NLTK.

    Args:
        text: Input text

    Returns:
        List of (start, end, sentence_text) tuples
    """
    sentence_spans = list(ru_tokenizer.span_tokenize(text))
    sentences = [(s, e, text[s:e]) for s, e in sentence_spans]
    return sentences


def convert_brat_to_tacred_sent(
    brat_dir: str,
    output_file: str,
    rel2id_output: Optional[str] = None,
    ner2id_output: Optional[str] = None,
    only_with_relations: bool = True,
    min_entities: int = 1
):
    """
    Convert document-level BRAT format to sentence-level TACRED format.

    Key features:
    - Splits documents into sentences using NLTK
    - Only outputs relations where both entities are in the same sentence
    - Uses 'text' field with character-level positions (sentence-relative)
    - No tokenization - OpenNRE's BERT tokenizer handles this

    Args:
        brat_dir: Directory containing .ann and .txt files
        output_file: Output file path (one JSON per line)
        rel2id_output: Optional path to save relation to ID mapping
        ner2id_output: Optional path to save NER type to ID mapping
        only_with_relations: If True, only output sentences with relations
        min_entities: Minimum entities per sentence to include
    """

    # Find all .txt files
    txt_files = []
    for root, dirs, files in os.walk(brat_dir):
        for f in files:
            if f.endswith('.txt'):
                txt_files.append(os.path.join(root, f))

    print(f"Found {len(txt_files)} .txt files in {brat_dir}")

    # Track all relation types and entity types
    all_relations = set()
    all_entity_types = set()

    # Output data
    output_data = []

    # Statistics
    stats = {
        'total_documents': 0,
        'total_document_sentences': 0,
        'total_sentences_with_entities': 0,
        'total_sentences_with_relations': 0,
        'total_relations': 0,
        'skipped_no_relations': 0,
        'skipped_entity_not_found': 0,
        'skipped_too_few_entities': 0,
        'skipped_discontinuous': 0,
        'nested_entities': 0
    }

    for txt_path in tqdm(txt_files, desc="Converting files"):
        stats['total_documents'] += 1

        # Read text
        with open(txt_path, 'r', encoding='utf-8') as f:
            doc_text = f.read()

        # Parse annotations
        ann_path = txt_path.replace('.txt', '.ann')
        if not os.path.exists(ann_path):
            continue

        entities, relations, skipped_disc = parse_brat_ann(ann_path)
        stats['skipped_discontinuous'] += skipped_disc

        # Track types
        for ent in entities.values():
            all_entity_types.add(ent['type'])
        for rel in relations:
            all_relations.add(rel['type'])

        # Split into sentences
        sentences = split_into_sentences(doc_text)
        stats['total_document_sentences'] += len(sentences)

        # Process each sentence
        for sent_idx, (sent_start, sent_end, sent_text) in enumerate(sentences):
            # Find entities in this sentence
            # No tokenization needed - we'll use character positions
            sent_entities = {}
            for ent_id, ent in entities.items():
                # Check if entity is completely within this sentence
                if ent['start'] >= sent_start and ent['end'] <= sent_end:
                    # Store entity with sentence-relative character positions
                    sent_entities[ent_id] = {
                        **ent,
                        'sent_char_start': ent['start'] - sent_start,
                        'sent_char_end': ent['end'] - sent_start
                    }

            # Skip if too few entities
            if len(sent_entities) < min_entities:
                stats['skipped_too_few_entities'] += 1
                continue

            stats['total_sentences_with_entities'] += 1

            # Find relations in this sentence
            sent_relations = []
            for rel in relations:
                head_id = rel['head']
                tail_id = rel['tail']

                # Check if both entities are in this sentence
                if head_id in sent_entities and tail_id in sent_entities:
                    sent_relations.append(rel)

            # Skip if no relations and only_with_relations is True
            if only_with_relations and len(sent_relations) == 0:
                stats['skipped_no_relations'] += 1
                continue

            stats['total_sentences_with_relations'] += 1

            # Create TACRED entries for each relation in this sentence
            for rel in sent_relations:
                stats['total_relations'] += 1

                head_ent = sent_entities[rel['head']]
                tail_ent = sent_entities[rel['tail']]

                # Check for nested entities (using character positions)
                head_span = (head_ent['sent_char_start'], head_ent['sent_char_end'])
                tail_span = (tail_ent['sent_char_start'], tail_ent['sent_char_end'])

                # Detect nesting
                if (head_span[0] <= tail_span[0] and tail_span[1] <= head_span[1]) or \
                   (tail_span[0] <= head_span[0] and head_span[1] <= tail_span[1]):
                    stats['nested_entities'] += 1

                # Output format: 'text' field with character positions
                # This allows OpenNRE's BERT tokenizer to handle tokenization correctly
                tacred_entry = {
                    'text': sent_text,
                    'h': {
                        'name': head_ent['text'],
                        'pos': [head_ent['sent_char_start'], head_ent['sent_char_end']],
                        'type': head_ent['type']
                    },
                    't': {
                        'name': tail_ent['text'],
                        'pos': [tail_ent['sent_char_start'], tail_ent['sent_char_end']],
                        'type': tail_ent['type']
                    },
                    'relation': rel['type']
                }

                output_data.append(tacred_entry)

    # Print statistics
    print("\n" + "=" * 80)
    print("CONVERSION STATISTICS")
    print("=" * 80)
    print(f"Total documents processed: {stats['total_documents']}")
    print(f"Total sentences in documents: {stats['total_document_sentences']}")
    print(f"Sentences with entities (>= {min_entities}): {stats['total_sentences_with_entities']}")
    print(f"Sentences with relations: {stats['total_sentences_with_relations']}")
    print(f"Total relations output: {stats['total_relations']}")
    print(f"\nSkipped:")
    print(f"  - Discontinuous entities: {stats['skipped_discontinuous']}")
    print(f"  - Too few entities: {stats['skipped_too_few_entities']}")
    print(f"  - No relations: {stats['skipped_no_relations']}")
    print(f"  - Entity not found: {stats['skipped_entity_not_found']}")
    print(f"\nNested entity pairs: {stats['nested_entities']}")
    print(f"\nUnique entity types: {len(all_entity_types)}")
    print(f"Unique relation types: {len(all_relations)}")
    print("=" * 80)

    # Write output file (one JSON per line)
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in output_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"✓ Wrote {len(output_data)} entries to {output_file}")

    # Create rel2id mapping
    if rel2id_output:
        rel2id = {'Na': 0}  # No relation
        for idx, rel in enumerate(sorted(all_relations), start=1):
            rel2id[rel] = idx

        with open(rel2id_output, 'w', encoding='utf-8') as f:
            json.dump(rel2id, f, ensure_ascii=False, indent=2)

        print(f"✓ Wrote relation mapping to {rel2id_output}")

    # Create ner2id mapping
    if ner2id_output:
        ner2id = {}
        for idx, ner in enumerate(sorted(all_entity_types)):
            ner2id[ner] = idx

        with open(ner2id_output, 'w', encoding='utf-8') as f:
            json.dump(ner2id, f, ensure_ascii=False, indent=2)

        print(f"✓ Wrote NER type mapping to {ner2id_output}")

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description='Convert document-level BRAT to sentence-level TACRED format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert train split
  python brat2tacred-sent.py --input_dir /path/to/nerel/train --output_file train.txt

  # Convert with mappings
  python brat2tacred-sent.py \\
      --input_dir /path/to/nerel/train \\
      --output_file train.txt \\
      --rel2id rel2id.json \\
      --ner2id ner2id.json

  # Include sentences without relations
  python brat2tacred-sent.py \\
      --input_dir /path/to/nerel/train \\
      --output_file train.txt \\
      --include_no_relations

Key Differences from brat2tacred.py:
  - Splits documents into sentences (document-level → sentence-level)
  - Only outputs relations where both entities are in same sentence
  - Adjusts entity positions relative to sentence (not document)
  - Uses NLTK tokenizer (consistent with brat_doc2sent.py)

Output Format (TACRED):
  One JSON per line:
  {
    "token": ["Иван", "Петров", "работает", "в", "Google"],
    "h": {"name": "Иван Петров", "pos": [0, 2], "type": "PERSON"},
    "t": {"name": "Google", "pos": [4, 5], "type": "ORGANIZATION"},
    "relation": "WORKPLACE"
  }
        """
    )

    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory containing BRAT .ann and .txt files'
    )

    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Output file path (will write one JSON per line)'
    )

    parser.add_argument(
        '--rel2id',
        type=str,
        default=None,
        help='Output path for relation to ID mapping (JSON)'
    )

    parser.add_argument(
        '--ner2id',
        type=str,
        default=None,
        help='Output path for NER type to ID mapping (JSON)'
    )

    parser.add_argument(
        '--include_no_relations',
        action='store_true',
        help='Include sentences without relations (default: only sentences with relations)'
    )

    parser.add_argument(
        '--min_entities',
        type=int,
        default=1,
        help='Minimum entities per sentence to include (default: 1)'
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"ERROR: Input directory not found: {args.input_dir}")
        exit(1)

    # Create output directory if needed
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert
    convert_brat_to_tacred_sent(
        brat_dir=args.input_dir,
        output_file=args.output_file,
        rel2id_output=args.rel2id,
        ner2id_output=args.ner2id,
        only_with_relations=not args.include_no_relations,
        min_entities=args.min_entities
    )

    print("\n✅ Conversion complete!")
    print("\nYou can now train OpenNRE with:")
    print(f"  python example/train_supervised_bert.py \\")
    print(f"      --train_file {args.output_file} \\")
    print(f"      --pooler entity \\")
    print(f"      --pretrain_path DeepPavlov/rubert-base-cased")


if __name__ == '__main__':
    main()
