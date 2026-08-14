import numpy as np

def normalize_image(image, mean, std):
    """
    Returns: 3D list of shape (H, W, C), each value rounded to 4 decimals
    """
    image = np.array(image)
    mean = np.array(mean)
    std = np.array(std)

    mean = mean[np.newaxis, np.newaxis, :]
    std = std[np.newaxis, np.newaxis, :]

    image = (image - mean) / std
    return image.round(4)

    
