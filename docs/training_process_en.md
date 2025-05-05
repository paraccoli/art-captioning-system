# Training Process

## Dataset

We use the SemArt dataset:
- 21,384 artworks with corresponding captions
- Split ratio: Training 80%, Validation 10%, Testing 10%

## Preprocessing

### Images
- Resize: 256×256
- Center crop: 224×224
- Normalize: using ImageNet mean and standard deviation

### Text
- Tokenization: split by spaces and periods
- Convert to lowercase
- Special tokens: `<start>`, `<end>`, `<unk>`, `<pad>`
- Minimum frequency: 5 occurrences (words appearing less frequently are replaced with `<unk>`)

## Training Settings

### Hyperparameters
- Batch size: 32
- Epochs: 15-30
- Optimizer: Adam
- Learning rates: 1e-4 (encoder), 4e-4 (decoder)
- Weight decay: 1e-5
- Gradient clipping: 5.0
- Dropout rate: 0.5 (decoder)

### Loss Function
- Cross-entropy loss
- Teacher forcing: using actual words as input for the next step

### Training Strategy
- First few layers of the encoder are frozen (not trained)
- After validation loss plateaus, fine-tune the encoder as well
- Early stopping if validation loss doesn't improve for 5 epochs

### Checkpoints
Models are saved as checkpoints at the end of each epoch