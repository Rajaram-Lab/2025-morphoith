# %%

import numpy as np
import scipy

# %%

def SmoothResponseAndClasses(inResponse,smoothSize=0):
    """Process raw output from tissue classifier.

    Args:
        inResponse (array): tissue lassifier output.
        smoothSize (int, optional): Smoothing of response. Defaults to 0.

    Returns:
        smoothClasses (array): smoothed  tissue mask.
        smoothResponse (array): smoothet tisue model response.
    """
    numChannels = inResponse.shape[-1] if len(inResponse.shape) == 3 else 1
    smoothResponse=np.zeros((inResponse.shape[0], inResponse.shape[1], numChannels))
    for channelCounter in range(numChannels):
        smoothResponse[:,:,channelCounter] = scipy.ndimage.gaussian_filter(
            inResponse[:,:,channelCounter], smoothSize)
    smoothClasses = np.argmax(smoothResponse,axis=-1)
    return smoothClasses, smoothResponse