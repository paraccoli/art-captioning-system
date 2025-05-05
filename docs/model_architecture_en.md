# Model Architecture

This project uses a deep learning model that combines an encoder-decoder structure with an attention mechanism.

## Overall Structure

```
+---------------+    +----------------+    +----------------+
|               |    |                |    |                |
|  Image Encoder |--->|  Attention    |--->|    Decoder    |---> Caption
|   (ResNet-50)  |    |  Mechanism    |    |    (LSTM)     |
|               |    |                |    |                |
+---------------+    +----------------+    +----------------+
```

## Encoder

The encoder uses a pre-trained ResNet-50:

- Removes the final classification layer to extract feature maps
- Uses adaptive average pooling to adjust the size of feature maps
- Output: 14×14×2048 feature map (196 vectors of 512 dimensions)

## Attention Mechanism

The soft attention mechanism allows the decoder to focus on relevant image regions:

- Calculates alignment scores between the decoder's hidden state and each image region feature
- Converts alignment scores to attention weights using softmax function
- Aggregates features into a context vector using attention weights

## Decoder

The LSTM-based decoder generates words sequentially:

- Word embedding layer converts words to vector representations
- LSTM cell updates hidden states
- Attention gate adjusts the importance of the context vector
- Fully connected layer and softmax function predict the next word