# OpenNRE-ru (Russian adaptation)
 
This is a fork for OpenNRE package for Russian language. Some other functions and scripts were created too like BRAT to TACRED conversion and separate prediction script for trained OpenNRE module.

For main functions and preliminary installation, refer to original repository.

## BRAT to TACRED Converter

**Script**: `brat2tacred.py` - Convert BRAT format to TACRED format for OpenNRE

**Dependencies**:
```bash
pip install razdel tqdm
```

**Basic Usage**:
```bash
python brat2tacred.py \
    --input_dir /path/to/brat/data \
    --output_file output.txt \
    --rel2id rel2id.json
```

### What it Does

Converts BRAT standoff format (`.ann` + `.txt`) to TACRED JSON format (one JSON per line).

#### Input (BRAT)
```
doc.txt:  Иван Петров работает в Google.
doc.ann:  T1  PERSON 0 11 Иван Петров
          T2  ORGANIZATION 23 29  Google
          R1  WORKPLACE Arg1:T1 Arg2:T2
```

#### Output (TACRED)
```json
{
  "token": ["Иван", "Петров", "работает", "в", "Google", "."],
  "h": {"name": "Иван Петров", "pos": [0, 2], "type": "PERSON"},
  "t": {"name": "Google", "pos": [4, 5], "type": "ORGANIZATION"},
  "relation": "WORKPLACE"
}
```

### Command-Line Options

| Option | Required | Description |
|--------|----------|-------------|
| `--input_dir` | Yes | Directory with BRAT .ann and .txt files |
| `--output_file` | Yes | Output file (one JSON per line) |
| `--rel2id` | No | Save relation→ID mapping (JSON) |
| `--ner2id` | No | Save entity type→ID mapping (JSON) |
| `--include_no_relations` | No | Include sentences without relations |

---

## BRAT Document-to-Sentence TACRED Converter

**Script**: `brat2tacred-sent.py` - Convert document-level BRAT to sentence-level TACRED

**Use when**: Your BRAT data contains **full documents** (multiple sentences per file)

**What it does**:
1. Splits documents into sentences
2. Filters relations to sentence-level only
3. Adjusts entity positions relative to sentences
4. Outputs TACRED format (one JSON per line)

---

**Basic Usage**

```bash
python brat2tacred-sent.py \
    --input_dir /path/to/document-level/brat \
    --output_file output.txt \
    --rel2id rel2id.json
```
---

#### Input Format (Document-level BRAT) (multiple sentences per file):

```
doc001.txt:
Иван Петров работает в Google. Он учился в MIT. Google основана в 1998.

doc001.ann:
T1  PERSON 0 11 Иван Петров
T2  ORGANIZATION 23 29  Google
T3  PERSON 31 33  Он
T4  ORGANIZATION 43 46  MIT
T5  ORGANIZATION 48 54  Google
T6  DATE 65 69  1998
R1  WORKPLACE Arg1:T1 Arg2:T2
R2  EDUCATION Arg1:T3 Arg2:T4
R3  DATE_FOUNDED Arg1:T5 Arg2:T6
```

**Note**: This is a multi-sentence document. The script will:
1. Split into 3 sentences
2. Create separate TACRED entries for each sentence's relations
3. Filter out any cross-sentence relations
---

#### Output Format (TACRED)

One JSON per line:

```json
{"token": ["Иван", "Петров", "работает", "в", "Google", "."], "h": {"name": "Иван Петров", "pos": [0, 2], "type": "PERSON"}, "t": {"name": "Google", "pos": [4, 5], "type": "ORGANIZATION"}, "relation": "WORKPLACE"}
{"token": ["Он", "учился", "в", "MIT", "."], "h": {"name": "Он", "pos": [0, 1], "type": "PERSON"}, "t": {"name": "MIT", "pos": [3, 4], "type": "ORGANIZATION"}, "relation": "EDUCATION"}
{"token": ["Google", "основана", "в", "1998", "."], "h": {"name": "Google", "pos": [0, 1], "type": "ORGANIZATION"}, "t": {"name": "1998", "pos": [3, 4], "type": "DATE"}, "relation": "DATE_FOUNDED"}
```

**Key**: Entity positions (`.pos`) are **relative to the sentence**, not the document.

---

### Command-Line Options

| Option | Required | Description |
|--------|----------|-------------|
| `--input_dir` | Yes | Directory with document-level BRAT files |
| `--output_file` | Yes | Output file (one JSON per line) |
| `--rel2id` | No | Save relation→ID mapping (JSON) |
| `--include_no_relations` | No | Include sentences without relations |
| `--min_entities` | No | Minimum entities per sentence (default: 1) |

---

### Dependencies

```bash
pip install nltk tqdm

# Download NLTK data
python -c 'import nltk; nltk.download("punkt")'
```
