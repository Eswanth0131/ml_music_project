# Scaling Laws for Symbolic Music Generation | ML PROJECT

## README.md

This repository along with a google drive (link: https://drive.google.com/drive/folders/1m3_EeTz2uSYvga_28-SJ3BGKcA8a0pUh?usp=share_link) contains the code and experimental results for the ML Project, exploring scaling laws for Transformer and RNN models applied to symbolic music data represented in ABC notation.

### 1\. Project Overview and Goals

The primary goal of this project was to empirically investigate how model capacity ($N$) affects generalization performance (Validation Loss, $L$) in the structured, non-linguistic domain of symbolic music, adhering to the power law $L \propto N^{-\alpha}$.

The project was executed in four main parts:

1.  **Data Preparation:** Building a character-level tokenization pipeline for a large corpus of ABC notation.
2.  **Transformer Scaling:** Training five different Transformer (GPT) model sizes to derive the scaling exponent $\alpha$.
3.  **Architectural Comparison:** Training four RNN (LSTM) models and comparing their scaling behavior against the Transformers.
4.  **Final Generation:** Training the largest feasible model (GPT-XL) to full convergence and generating music samples.

### 2\. Setup and Dependencies

This project is built upon the `nanoGPT` framework.

**Hardware:**

  * **Scaling Studies:** CUDA GPU (e.g., NVIDIA V100/P100) or Apple MPS.
  * **Final Training:** High-memory CUDA GPU (e.g., NVIDIA A100/A6000) was required for the largest GPT-XL model.

**Prerequisites:**
You need the `midi2abc` command-line tool installed on your system to run the conversion scripts.

**Python Dependencies:**
Create and activate a virtual environment, then install dependencies:

```bash
pip install torch torchvision torchaudio numpy tqdm scikit-learn scipy matplotlib
# For data conversion:
pip install music21
```

### 3\. Data Preparation Pipeline (Part 1)

The pipeline converts the **Lakh MIDI Dataset** into character-level tokens suitable for training.

**A. Data Acquisition and Conversion:**

  * **Tool Used:** `midi2abc`
  * **Input:** Raw MIDI files (Lakh MIDI Dataset)
  * **Output:** ABC notation files saved to `data/abc/`
  * **Script:** `convert_midi_to_abc.py`

**B. Cleaning and Filtering:**

  * **Purpose:** Removes comments, empty lines, and files missing required headers (specifically `K:` - Key Signature) to ensure basic syntactic validity. No filtering was done for short length or duplication.
  * **Script:** `clean_abc.py`

**C. Vocabulary Generation and Tokenization:**

  * **Tokenization:** Pure **Character-Level Tokenization**.
  * **Vocabulary Size:** $\mathbf{97}$ unique characters.
  * **Output:** `data/vocab.json` (containing `token_to_id` and `id_to_token` mappings) and tokenized `train.jsonl`, `val.jsonl`, `test.jsonl` files.
  * **Scripts:** `build_vocab.py` and `tokenize_dataset.py`

### 4\. Model Training and Configurations

All models were trained using the **AdamW** optimizer with standard cross-entropy loss.

| Architecture | Scaling Study Constraint | Context Window | Optimizer | Precision |
| :--- | :--- | :--- | :--- | :--- |
| **GPT (Transformer)** | About 100M toketns | 256 tokens | AdamW | Float16 / Mixed |
| **RNN (LSTM)** | About 100M tokens | 256 tokens | AdamW | Float32 (Full) |

**Scaling Configurations Tested (Example):**

| Model Size | Type | Layers / Heads | Parameters ($N$) |
| :--- | :--- | :--- | :--- |
| Tiny | GPT | 3 / 3 | $\sim$1.4M |
| Large | GPT | 8 / 12 | $\sim$50M |
| XL (Best) | GPT | 12 / 12 | $\sim$107M |
| RNN-Large | LSTM | 5 Layers | $\sim$42.2M |

### 5\. Running the Final Generation (Part 4)

To replicate the final results from the best-performing GPT-XL model, ensure the following file structure is maintained and run the generation script:

```
nanoGPT/
├── generate.py
├── model.py
├── out_best/
│   └── ckpt.pt
└── data/
    └── music_project/
        └── meta.pkl
```

**Execution:**

```bash
python generate.py
```

  * **Output:** Generates 12 diverse samples (6 unconditional, 6 conditional) and saves them to `generated_samples.txt`.
  * **Final Result:** The GPT-XL model achieved a final test loss corresponding to a low perplexity, with approximately $\mathbf{75\%}$ of generated samples being syntactically valid (playable).
