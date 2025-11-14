# Patient-Builder

**Patient-Builder** is a Python toolkit for quickly generating realistic, high-fidelity therapy/patient transcripts and avatars from short character vignettes, powered by a fine-tuned model.

## Features

- Generates expert-level, detailed dialogue transcripts based on simple prompts.
- Supports fully custom vignettes for varied patient profiles.
- Easily extensible for research, teaching, or product demos.

## Contents

- `vignette_generator.py` — Core script for creating clinical vignette transcripts.
- `Ex.py` — Example usage and interface code.

## Installation

Clone the repo:
```bash
git clone https://github.com/aryanb03/Patient-Builder.git
cd Patient-Builder
```

Install dependencies (if any):
```bash
pip install -r requirements.txt
```

## Usage

Run directly from command line:
```bash
python vignette_generator.py --vignette "Short patient vignette here"
```

Or use in Python:
```python
from vignette_generator import generate_transcript

transcript = generate_transcript("Brief patient vignette text")
print(transcript)
```

## Author

[aryanb03](https://github.com/aryanb03)
