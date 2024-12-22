# Denoising Images Using Autoencoder  

This project focuses on using a convolutional autoencoder to remove noise from images. The goal is to develop a deep learning model capable of restoring noisy images to their clean, original state.  

## Features  
- Denoising images affected by various types of noise (e.g., blur, Gaussian noise).  
- RGB image support with shape `(180, 180, 3)`.  
- Customizable architecture with Conv2D, Conv2DTranspose, BatchNormalization, and Dropout layers.  

## Dataset  
- The dataset consists of 2,000 RGB images of size `(180, 180, 3)`.  
- Noisy images are generated using a custom noise function.  

## Architecture  
- **Encoder**: Sequential convolutional layers to capture the essential features of the image.  
- **Latent Space**: Compressed representation of the image.  
- **Decoder**: Transposed convolutional layers to reconstruct the denoised image.  

## Tools and Libraries  
- **Python**: Programming language.  
- **Keras/TensorFlow**: For building and training the autoencoder model.  
- **NumPy**: For data manipulation and adding noise to the images.   

## Training Details  
- **Input**: Noisy images.  
- **Target (Output)**: Clean images.  
- **Loss Function**: Mean Squared Error (MSE).  
- **Optimizer**: Adam.  
