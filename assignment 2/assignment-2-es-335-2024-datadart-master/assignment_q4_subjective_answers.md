# Upscaling: Qualitative comparison

Perform superresolution on the image shown in the notebook to enhance its resolution by a factor of 2. Show a qualitative comparison of the original and reconstructed images.

## Upscaling: Qualitative Comparison

The code performs superresolution on the image to enhance its resolution by a factor of 2. It first uses bilinear interpolation to upscale the image and then compares the original and reconstructed images qualitatively. Below is the code snippet for this part:


## Upscaling: Quantitative comparison

Let us now do a quantitative comparison.

- Start with a 400x400 image (ground truth high resolution).
- Resize it to a 200x200 image (input image).
- Use RFF + Linear regression to increase the resolution to 400x400 (predicted high resolution image).
- Compute the following metrics:
  - RMSE on predicted vs. ground truth high-resolution image.
  - Peak SNR

## Upscaling: Quantitative Comparison

For the quantitative comparison, the code follows these steps:

It normalizes the original image and crops it to a specified size. Then, it downscales the cropped image to create a low-resolution input image. Next, it creates a coordinate map and applies Random Fourier Features (RFF) transformation to the input data. After that, it uses a linear regression model to upscale the low-resolution image to the original resolution. Finally, it computes the Root Mean Square Error (RMSE) and Signal-to-Noise Ratio (SNR) between the predicted and ground truth high-resolution images. Below is the code snippet for this part:


## Theoretical Observations

### Superresolution (Qualitative Comparison):

The superresolution technique effectively enhances the resolution of the image by a factor of 2. The qualitative comparison between the original and super-resolution images visually demonstrates the improvement in image quality and sharpness achieved through the superresolution process.

### Superresolution (Quantitative Comparison):

For quantitative comparison, the RMSE (Root Mean Squared Error) between the predicted high-resolution image and the ground truth high-resolution image provides a measure of the reconstruction accuracy. Additionally, the Peak Signal-to-Noise Ratio (SNR) quantifies the quality of the reconstructed image relative to the original. Higher RMSE values indicate higher reconstruction errors, while higher SNR values indicate better fidelity of the reconstructed image to the original.

---

## Completing Image with Random Missing Data

Apply RFF to complete the image with 10%, 20%, and so on up to 90% of its data missing randomly. Randomly remove portions of the data, train the model on the remaining data, and predict on the entire image. Display the reconstructed images for each missing data percentage and show the metrics calculated above. What do you conclude?

This code demonstrates the application of Random Fourier Features (RFF) to complete an image with random missing data. It follows the steps outlined below:

- **Generate Mask:** A random mask is generated using `torch.rand_like` to represent the missing data in the image.
- **Iterate Over Missing Data Percentages:** The code iterates over percentages of missing data from 10% to 90% in increments of 10%.
- **Create Incomplete Image:** For each percentage, a version of the original image with missing data is created by element-wise multiplication with the mask.
- **Apply RFF Transformation:** The incomplete image is transformed using RFF to prepare it for model training.
- **Model Training:** A linear regression model is trained using the transformed incomplete image and ground truth image data.
- **Reconstruct Image:** The trained model is then used to predict the missing data and reconstruct the complete image.
- **Display Results:** The original image, the image with missing data, and the reconstructed image are displayed for visual comparison.

## Theoretical Observation:

The reconstructed images show varying degrees of fidelity to the original image, depending on the percentage of missing data. As the percentage of missing data increases, the quality of reconstruction generally decreases. This is expected, as higher percentages of missing data provide less information for the model to learn from, resulting in less accurate predictions. Additionally, metrics such as RMSE and SNR can be computed to quantitatively assess the quality of reconstruction at different missing data percentages. Overall, this experiment demonstrates the capability of RFF-based image completion and highlights the trade-off between missing data percentage and reconstruction accuracy.
