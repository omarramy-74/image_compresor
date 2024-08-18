import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans  # Using sklearn for better K-Means implementation

st.title("Image Compressor")
st.write("Made by Omar Ramy")
UploadedFile = st.file_uploader('Upload your file here')

if UploadedFile:
    original_img = Image.open(UploadedFile).convert('RGB')
    original_img = np.array(original_img)

    # Flatten the image
    X_img = original_img.reshape((-1, 3))

    # Number of colors (clusters)
    K = 16  # Increase this number to improve quality

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=K, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_img)

    # Replace each pixel value with its centroid value
    X_compressed = kmeans.cluster_centers_[kmeans.labels_]
    X_compressed = np.clip(X_compressed.astype('uint8'), 0, 255)

    # Reshape back to the original image shape
    X_compressed = X_compressed.reshape(original_img.shape)

    # Display original and compressed images
    fig, ax = plt.subplots(1, 2, figsize=(16, 16))
    ax[0].imshow(original_img)
    ax[0].set_title('Original')
    ax[0].axis('off')

    ax[1].imshow(X_compressed)
    ax[1].set_title(f'Compressed with {K} colors')
    ax[1].axis('off')

    st.pyplot(fig)
