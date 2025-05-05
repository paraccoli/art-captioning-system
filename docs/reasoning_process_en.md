# Inference Process

## Overview
This document explains the process of generating captions for new artwork images using the trained model.

## Preprocessing
1. Load and convert the image to RGB format
2. Apply the same resizing and normalization as during training
3. Pass through the encoder to extract features

## Caption Generation Algorithms

### Beam Search
By default, captions are generated using the beam search algorithm:

1. Start with a `<start>` token as the first word
2. At each step, calculate the probability of the next word for each of the current candidate sequences
3. Keep the top k sequences (beam width)
4. Repeat until reaching the `<end>` token or maximum length
5. Select the sequence with the highest probability

### Greedy Generation
As a simpler alternative, we also implement greedy generation that selects the most probable word at each step.

### Parameters
- Beam width: 3 (default)
- Temperature: 1.0 (higher values increase diversity)
- Repetition penalty: 2.0 (reduces repetition of the same words)

## Attention Visualization
Using the `--visualize` option, you can visualize the attention weights during the generation process. This shows which regions of the image the model is focusing on for each generated word.

The attention maps help interpret the model's decision-making process and verify that it's attending to relevant image regions when generating specific words.