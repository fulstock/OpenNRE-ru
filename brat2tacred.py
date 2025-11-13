#!/usr/bin/env python3
"""
BRAT to TACRED Format Converter for OpenNRE

Converts BRAT standoff format (.ann + .txt files) to TACRED-like JSON format
for use with OpenNRE sentence-level relation extraction.

This script is optimized for:
- NEREL dataset (Russian text)
- Nested entity handling
- OpenNRE with entity markers (ent.marker-ent configuration)
- Uses 'text' field with CHARACTER-LEVEL positions (not token-level)
- OpenNRE's BERT tokenizer handles tokenization internally

Usage:
    python brat2tacred.py --input_dir /path/to/brat/data --output_file train.txt

Output Format (one JSON per line):
{
  "text": "Иван Петров работает в Google",
  "h": {"name": "Иван Петров", "pos": [0, 11], "type": "PERSON"},
  "t": {"name": "Google", "pos": [23, 29], "type": "ORGANIZATION"},
  "relation": "WORKPLACE"
}

Note: positions are character indices, not token indices.
"""

import os
import json
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable

# Note: We use razdel only for sentence splitting (if needed).
# For tokenization, OpenNRE's BERT tokenizer handles it internally.
try:
    import razdel
except ImportError:
    print("ERROR: razdel library not found. Install with: pip install razdel")
    print("razdel is required for Russian text sentence splitting.")
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


def sentence_split_russian(text: str) -> List[Tuple[int, int, str]]:
    """
    Split Russian text into sentences using razdel.

    Args:
        text: Input text

    Returns:
        List of (start, end, sentence_text) tuples
    """
    sent_objs = list(razdel.sentenize(text))
    sentences = [(s.start, s.stop, s.text) for s in sent_objs]
    return sentences


# Note: No tokenization or token span finding needed.
# We use character-level positions directly, and OpenNRE's BERT tokenizer
# handles subword tokenization internally.


def convert_brat_to_tacred(
    brat_dir: str,
    output_file: str,
    rel2id_output: Optional[str] = None,
    ner2id_output: Optional[str] = None,
    only_with_relations: bool = True,
    merge_multi_sentence: bool = True,
    generate_all_pairs: bool = False
):
    """
    Convert BRAT format files to TACRED format for OpenNRE.

    Args:
        brat_dir: Directory containing .ann and .txt files
        output_file: Output file path (one JSON per line)
        rel2id_output: Optional path to save relation to ID mapping
        ner2id_output: Optional path to save NER type to ID mapping
        only_with_relations: If True, only output sentences with relations (ignored if generate_all_pairs=True)
        merge_multi_sentence: If True, merge sentences when relation spans multiple sentences
        generate_all_pairs: If True, generate ALL entity pairs (for inference), relation will be 'Na' if no gold
    """

    # Find all .ann files
    ann_files = []
    for root, dirs, files in os.walk(brat_dir):
        for f in files:
            if f.endswith('.ann'):
                ann_files.append(os.path.join(root, f))

    print(f"Found {len(ann_files)} .ann files in {brat_dir}")

    # Track all relation types and entity types
    all_relations = set()
    all_entity_types = set()

    # Output data
    output_data = []

    # Statistics
    stats = {
        'total_files': 0,
        'total_sentences': 0,
        'total_relations': 0,
        'skipped_no_relations': 0,
        'skipped_entity_not_found': 0,
        'skipped_multi_sentence': 0,
        'skipped_discontinuous': 0,
        'nested_entities': 0
    }

    for ann_path in tqdm(ann_files, desc="Converting files"):
        stats['total_files'] += 1

        # Check if .txt file exists
        txt_path = ann_path.replace('.ann', '.txt')
        if not os.path.exists(txt_path):
            continue

        # Read text
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Parse annotations
        entities, relations, skipped_disc = parse_brat_ann(ann_path)
        stats['skipped_discontinuous'] += skipped_disc

        # Track types
        for ent in entities.values():
            all_entity_types.add(ent['type'])
        for rel in relations:
            all_relations.add(rel['type'])

        # Split into sentences
        sentences = sentence_split_russian(text)

        # Process each sentence
        for sent_idx, (sent_start, sent_end, sent_text) in enumerate(sentences):
            # Find entities in this sentence
            # No tokenization needed - we use character positions
            sent_entities = {}
            for ent_id, ent in entities.items():
                # Check if entity is in this sentence
                if ent['start'] >= sent_start and ent['end'] <= sent_end:
                    # Store entity with sentence-relative character positions
                    sent_entities[ent_id] = {
                        **ent,
                        'sent_char_start': ent['start'] - sent_start,
                        'sent_char_end': ent['end'] - sent_start
                    }

            # Find relations in this sentence
            sent_relations = []
            for rel in relations:
                head_id = rel['head']
                tail_id = rel['tail']

                # Check if both entities are in this sentence
                if head_id in sent_entities and tail_id in sent_entities:
                    sent_relations.append(rel)

            # Create gold relation lookup map
            gold_rel_map = {}
            for rel in sent_relations:
                gold_rel_map[(rel['head'], rel['tail'])] = rel['type']

            if generate_all_pairs:
                # Generate ALL entity pairs for inference
                if len(sent_entities) < 2:
                    continue

                entity_list = list(sent_entities.values())
                for i in range(len(entity_list)):
                    for j in range(len(entity_list)):
                        if i == j:
                            continue

                        head_ent = entity_list[i]
                        tail_ent = entity_list[j]

                        # Check if there's a gold relation for this pair
                        gold_rel = gold_rel_map.get((head_ent['id'], tail_ent['id']), 'Na')

                        # Track gold relations
                        if gold_rel != 'Na':
                            stats['total_relations'] += 1

                        # Check for nested entities
                        head_span = (head_ent['sent_char_start'], head_ent['sent_char_end'])
                        tail_span = (tail_ent['sent_char_start'], tail_ent['sent_char_end'])

                        if (head_span[0] <= tail_span[0] and tail_span[1] <= head_span[1]) or \
                           (tail_span[0] <= head_span[0] and head_span[1] <= tail_span[1]):
                            stats['nested_entities'] += 1

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
                            'relation': gold_rel
                        }

                        output_data.append(tacred_entry)
                        stats['total_sentences'] += 1
            else:
                # Original behavior: only output pairs with gold relations
                # Skip sentences without relations if requested
                if only_with_relations and len(sent_relations) == 0:
                    stats['skipped_no_relations'] += 1
                    continue

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
                    stats['total_sentences'] += 1

    # Print statistics
    print("\n" + "=" * 80)
    print("CONVERSION STATISTICS")
    print("=" * 80)
    print(f"Total files processed: {stats['total_files']}")
    print(f"Total entity pairs output: {stats['total_sentences']}")
    print(f"  - With gold relations: {stats['total_relations']} ({stats['total_relations']/stats['total_sentences']*100:.1f}%)")
    print(f"  - Without relations (Na): {stats['total_sentences'] - stats['total_relations']} ({(stats['total_sentences'] - stats['total_relations'])/stats['total_sentences']*100:.1f}%)")
    print(f"\nSkipped:")
    print(f"  - Discontinuous entities: {stats['skipped_discontinuous']}")
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
        description='Convert BRAT format to TACRED format for OpenNRE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert train split
  python brat2tacred.py --input_dir /path/to/nerel/train --output_file train.txt

  # Convert with mappings
  python brat2tacred.py \\
      --input_dir /path/to/nerel/train \\
      --output_file train.txt \\
      --rel2id rel2id.json \\
      --ner2id ner2id.json

  # Include sentences without relations
  python brat2tacred.py \\
      --input_dir /path/to/nerel/train \\
      --output_file train.txt \\
      --include_no_relations

Output Format:
  One JSON per line (TACRED format):
  {
    "token": ["Иван", "Петров", "работает", "в", "Google"],
    "h": {"name": "Иван Петров", "pos": [0, 2], "type": "PERSON"},
    "t": {"name": "Google", "pos": [4, 5], "type": "ORGANIZATION"},
    "relation": "WORKPLACE"
  }

  Note: pos is [start, end) with EXCLUSIVE end (Python slice convention)
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
        '--all_pairs',
        action='store_true',
        help='Generate ALL entity pairs for inference (not just pairs with gold relations). Use this for inference on new data.'
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
    convert_brat_to_tacred(
        brat_dir=args.input_dir,
        output_file=args.output_file,
        rel2id_output=args.rel2id,
        ner2id_output=args.ner2id,
        only_with_relations=not args.include_no_relations,
        generate_all_pairs=args.all_pairs
    )

    print("\n✅ Conversion complete!")


if __name__ == '__main__':
    main()
