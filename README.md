# Road to Data Scientist III (Deep Learning CNN for Image Classification)

In this project, 4 CNN models for sport predicting has been developed. Transfer learning was used for 3 of the models, incorporating the pretrained weights of "ResNet50", "EfficientNetB0" and "MobileNetV2", respectively. One of the models was developed from scratch with convolutional/pool/batch-normalization/dense layers using only the training data for weights calculation. 

Transfer learning proved to be essential for predicting 100 different types of sports-based images. The model that was built from scratch without using pretrained models, only reach nearly 50 % of accuracy in the test dataset, while the other 3 models that integrates transfer learning, achieved over 90 % accuracy in few epochs.

# Transfer Learning using Pre-trained model. What is transfer learning?

Transfer learning is a machine learning technique that allows a pre-trained model, which has been trained on a large dataset, to be reused as a starting point for a different but related task or dataset. Instead of training a model from scratch, transfer learning leverages the knowledge and learned representations of the pre-trained model to accelerate and improve the training process for a new task.

In transfer learning, the pre-trained model, typically a deep neural network, has already learned to extract relevant features and patterns from the original dataset it was trained on, which is often a large-scale, general-purpose dataset such as ImageNet. These learned features capture generic image characteristics like edges, textures, shapes, and higher-level object representations.

To apply transfer learning, the pre-trained model is taken as a base network, and the final layers of the network, responsible for the specific task or output, are replaced or fine-tuned. The idea is to retrain the pre-trained network's knowledge of low-level and intermediate features while adapting the higher-level layers to the new task or dataset.

This can help with :

Reduced Training Time: reduces the time and computational resources required to train a model from scratch. Since the base network has already learned a rich set of features, only the final layers need to be trained, resulting in faster convergence.

Improved Performance: Pre-trained models often exhibit strong generalization capabilities due to their exposure to a large and diverse dataset. By leveraging these learned representations, transfer learning can enhance the model's performance on the new task, especially when the new dataset is limited or lacks sufficient training samples.

Handling Data Scarcity: Transfer learning is particularly useful when the target task has limited labeled data. The pre-trained model's knowledge serves as a form of regularization, preventing overfitting and improving generalization on the new dataset.

The dataset used was provided by Gerry, accessible at the following URL: https://www.kaggle.com/datasets/gpiosenka/sports-classification
