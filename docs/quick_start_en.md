# Quick Start Guide

## Environment Setup

### Requirements
- Python 3.8 or above
- PyTorch 1.7 or above
- CUDA-capable GPU recommended (but can run without)

### Setup
```bash
# Clone the repository
git clone https://github.com/username/art-captioning-project.git
cd art-captioning-project

# Install dependencies
pip install -r requirements.txt

# Download NLTK resources
python download_nltk.py
```

## Dataset Preparation
1. Download the [SemArt dataset](https://research.aston.ac.uk/en/datasets/semart-dataset)
2. Extract to `data/semart/` directory
3. Verify the folder structure:
   ```
   data/
   ├── Images/           # Image files
   ├── semart_train.csv  # Training data
   ├── semart_val.csv    # Validation data
   └── semart_test.csv   # Test data
   ```

## Usage

### Building the Vocabulary
```bash
python create_vocab.py
```

### Training the Model
```bash
python train.py
```

### Generating Captions
```bash
python infer.py --model_path checkpoints/checkpoint_epoch_15.pth.tar --image_path data/Images/00000-allegory.jpg --vocab_path data/vocabulary.pkl
```

### With Attention Visualization
```bash
python infer.py --model_path checkpoints/checkpoint_epoch_15.pth.tar --image_path data/Images/00000-allegory.jpg --vocab_path data/vocabulary.pkl --visualize
```

### Evaluating a Trained Model
```bash
python evaluate.py --model_path checkpoints/checkpoint_epoch_15.pth.tar --data_path data/ --vocab_path data/vocabulary.pkl
```

### Generating Caption Examples
```bash
python examine_captions.py --model_path checkpoints/checkpoint_epoch_15.pth.tar --data_path data/ --vocab_path data/vocabulary.pkl --num_examples 5
```