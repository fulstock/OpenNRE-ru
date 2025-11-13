#!/usr/bin/env python3
"""
Convert OpenNRE predictions back to BRAT format.

Takes predictions from predict.py and converts them back to BRAT .ann files,
preserving original entity annotations and adding predicted relations.

Usage:
    python predictions2brat.py \
        --predictions predictions.json \
        --original_brat_dir /path/to/original/brat/files \
        --output_dir /path/to/output/brat/files \
        --min_confidence 0.5
"""

import argparse
import json
import os
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_original_entities(ann_path: str) -> Dict[Tuple[int, int, str], str]:
    """
    Parse entity annotations from original BRAT file.

    Returns:
        Dict mapping (start, end, text) -> entity_id
    """
    entity_map = {}

    with open(ann_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('T'):
                continue

            parts = line.split('\t')
            if len(parts) < 3:
                continue

            entity_id = parts[0]
            ann_parts = parts[1].split()

            if len(ann_parts) < 3:
                continue

            # Skip discontinuous entities
            if ';' in parts[1]:
                continue

            try:
                start = int(ann_parts[1])
                end = int(ann_parts[2])
                text = parts[2]

                entity_map[(start, end, text)] = entity_id
            except (ValueError, IndexError):
                continue

    return entity_map


def find_entity_in_sentence(entity_info: Dict, sent_start: int, entity_map: Dict) -> str:
    """
    Find entity ID in original BRAT file.

    Args:
        entity_info: Entity dict with 'name' and 'pos' (sentence-relative positions)
        sent_start: Character offset where sentence starts in document
        entity_map: Mapping from (start, end, text) to entity_id

    Returns:
        Entity ID or None if not found
    """
    # Convert sentence-relative position to document-absolute position
    doc_start = sent_start + entity_info['pos'][0]
    doc_end = sent_start + entity_info['pos'][1]
    text = entity_info['name']

    # Try exact match first
    entity_id = entity_map.get((doc_start, doc_end, text))
    if entity_id:
        return entity_id

    # Try fuzzy match (sometimes whitespace differs)
    for (start, end, ent_text), ent_id in entity_map.items():
        if start == doc_start and end == doc_end:
            # Position matches, text might differ slightly
            if ent_text.strip() == text.strip():
                return ent_id

    return None


def find_sentence_start(text_content: str, sentence_text: str) -> int:
    """
    Find where a sentence starts in the document.

    Args:
        text_content: Full document text
        sentence_text: Sentence to find

    Returns:
        Character offset where sentence starts, or -1 if not found
    """
    # Try exact match first
    pos = text_content.find(sentence_text)
    if pos != -1:
        return pos

    # Try with normalized whitespace
    normalized_sent = ' '.join(sentence_text.split())
    normalized_doc = ' '.join(text_content.split())

    pos = normalized_doc.find(normalized_sent)
    if pos != -1:
        # Convert position back to original text
        # This is approximate, but should work for most cases
        return text_content.find(sentence_text.split()[0])

    return -1


def convert_predictions_to_brat(
    predictions_file: str,
    original_brat_dir: str,
    output_dir: str,
    min_confidence: float = 0.0,
    skip_na: bool = True
):
    """
    Convert predictions to BRAT format.

    Args:
        predictions_file: JSON file with predictions
        original_brat_dir: Directory with original BRAT files
        output_dir: Output directory for new BRAT files
        min_confidence: Minimum confidence threshold (if predictions have confidence)
        skip_na: Skip predictions with relation='Na'
    """

    # Load predictions
    print(f"Loading predictions from {predictions_file}")
    predictions = []
    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            predictions.append(json.loads(line))

    print(f"Loaded {len(predictions)} predictions")

    # Group predictions by file (need to determine which file each prediction belongs to)
    # Since predictions don't have file names, we'll try to match by text content

    # First, build a mapping from text snippets to files
    print(f"\nScanning original BRAT files in {original_brat_dir}")
    text_files = []
    for root, dirs, files in os.walk(original_brat_dir):
        for f in files:
            if f.endswith('.txt'):
                text_files.append(os.path.join(root, f))

    print(f"Found {len(text_files)} .txt files")

    # Group predictions by finding which file they belong to
    predictions_by_file = defaultdict(list)
    unmatched_predictions = []

    print("\nMatching predictions to files...")
    for pred_idx, pred in enumerate(predictions):
        if pred_idx % 100 == 0:
            print(f"  Processed {pred_idx}/{len(predictions)} predictions", end='\r')

        matched = False
        pred_text = pred['text']

        # Try to find this text in one of the files
        for txt_path in text_files:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if pred_text in content:
                predictions_by_file[txt_path].append(pred)
                matched = True
                break

        if not matched:
            unmatched_predictions.append(pred)

    print(f"\n  Matched {len(predictions) - len(unmatched_predictions)} predictions")
    print(f"  Unmatched: {len(unmatched_predictions)}")

    if unmatched_predictions:
        print(f"\nWarning: {len(unmatched_predictions)} predictions could not be matched to files")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each file
    print(f"\nGenerating BRAT files in {output_dir}")
    stats = {
        'files_processed': 0,
        'relations_added': 0,
        'entities_not_found': 0,
        'skipped_na': 0,
        'skipped_low_confidence': 0
    }

    for txt_path, file_predictions in predictions_by_file.items():
        stats['files_processed'] += 1

        # Get corresponding .ann file
        ann_path = txt_path.replace('.txt', '.ann')
        if not os.path.exists(ann_path):
            print(f"Warning: No .ann file found for {txt_path}")
            continue

        # Read original entities
        entity_map = parse_original_entities(ann_path)

        # Read text content for sentence matching
        with open(txt_path, 'r', encoding='utf-8') as f:
            text_content = f.read()

        # Copy original .txt file
        output_txt_path = os.path.join(output_dir, os.path.basename(txt_path))
        shutil.copy2(txt_path, output_txt_path)

        # Create new .ann file
        output_ann_path = os.path.join(output_dir, os.path.basename(ann_path))

        # Copy original entity annotations
        entity_lines = []
        with open(ann_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('T'):
                    entity_lines.append(line)

        # Generate relation annotations from predictions
        relation_lines = []
        relation_counter = 1

        for pred in file_predictions:
            relation = pred['predicted_relation']

            # Skip 'Na' relations if requested
            if skip_na and relation == 'Na':
                stats['skipped_na'] += 1
                continue

            # Check confidence if available
            if 'confidence' in pred and pred['confidence'] < min_confidence:
                stats['skipped_low_confidence'] += 1
                continue

            # Find sentence start position
            sent_start = find_sentence_start(text_content, pred['text'])
            if sent_start == -1:
                print(f"Warning: Could not find sentence in {os.path.basename(txt_path)}")
                continue

            # Find entity IDs
            head_id = find_entity_in_sentence(pred['head'], sent_start, entity_map)
            tail_id = find_entity_in_sentence(pred['tail'], sent_start, entity_map)

            if not head_id or not tail_id:
                stats['entities_not_found'] += 1
                continue

            # Create relation annotation
            rel_line = f"R{relation_counter}\t{relation} Arg1:{head_id} Arg2:{tail_id}\t\n"
            relation_lines.append(rel_line)
            relation_counter += 1
            stats['relations_added'] += 1

        # Write new .ann file
        with open(output_ann_path, 'w', encoding='utf-8') as f:
            # Write entities first
            for line in entity_lines:
                f.write(line)

            # Write relations
            for line in relation_lines:
                f.write(line)

    # Print statistics
    print("\n" + "=" * 80)
    print("CONVERSION STATISTICS")
    print("=" * 80)
    print(f"Files processed: {stats['files_processed']}")
    print(f"Relations added: {stats['relations_added']}")
    print(f"Skipped 'Na' relations: {stats['skipped_na']}")
    print(f"Skipped low confidence: {stats['skipped_low_confidence']}")
    print(f"Entities not found: {stats['entities_not_found']}")
    print("=" * 80)
    print(f"\n✓ BRAT files saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert OpenNRE predictions back to BRAT format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert predictions to BRAT format
  python predictions2brat.py \\
      --predictions predictions.json \\
      --original_brat_dir /path/to/original/brat \\
      --output_dir /path/to/output/brat

  # Filter by confidence
  python predictions2brat.py \\
      --predictions predictions.json \\
      --original_brat_dir /path/to/original/brat \\
      --output_dir /path/to/output/brat \\
      --min_confidence 0.7

  # Include 'Na' predictions
  python predictions2brat.py \\
      --predictions predictions.json \\
      --original_brat_dir /path/to/original/brat \\
      --output_dir /path/to/output/brat \\
      --include_na
        """
    )

    parser.add_argument(
        '--predictions',
        required=True,
        help='JSON file with predictions from predict.py'
    )

    parser.add_argument(
        '--original_brat_dir',
        required=True,
        help='Directory containing original BRAT .ann and .txt files'
    )

    parser.add_argument(
        '--output_dir',
        required=True,
        help='Output directory for BRAT files with predicted relations'
    )

    parser.add_argument(
        '--min_confidence',
        type=float,
        default=0.0,
        help='Minimum confidence threshold (default: 0.0, use all predictions)'
    )

    parser.add_argument(
        '--include_na',
        action='store_true',
        help='Include predictions with relation="Na" (default: skip them)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.predictions):
        print(f"ERROR: Predictions file not found: {args.predictions}")
        exit(1)

    if not os.path.exists(args.original_brat_dir):
        print(f"ERROR: Original BRAT directory not found: {args.original_brat_dir}")
        exit(1)

    # Convert
    convert_predictions_to_brat(
        predictions_file=args.predictions,
        original_brat_dir=args.original_brat_dir,
        output_dir=args.output_dir,
        min_confidence=args.min_confidence,
        skip_na=not args.include_na
    )

    print("\n✅ Conversion complete!")


if __name__ == '__main__':
    main()
