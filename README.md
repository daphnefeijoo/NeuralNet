This repository features a custom-built neural network framework developed entirely without external libraries, specifically designed for various machine learning tasks, including handwriting recognition. The framework employs a node-based architecture that allows users to construct and train models intuitively. It includes implementations for several tasks:

Binary Perceptron: Designed for binary classification with output labels of either 1 or -1, utilizing a parameter node for weights without needing a separate bias.

Regression Model: Trains a neural network to approximate 
sin
⁡
(
𝑥
)
sin(x) over the interval 
[
−
2
𝜋
,
2
𝜋
]
[−2π,2π], using nn.SquareLoss as the loss function.

Digit Classification: Classifies handwritten digits from the MNIST dataset, outputting a 10-dimensional vector for each 28x28 pixel input, with nn.SoftmaxLoss as the loss function. The model aims for at least 97% accuracy on the test set.

Language Identification: Identifies the language of individual words from a dataset comprising five languages.

By focusing on a tailored approach, this project provides a unique platform for exploring and implementing machine learning concepts, particularly in the context of handwriting recognition and classification tasks.






