
# Obervations
Observations:

1. The use of low-rank decomposition through gradient descent consistently produces high-quality reconstructions for the images.

2. A clear trend emerges that as the rank increases, the mean RMSE tends to rise while the mean PSNR decreases. This suggests that images with missing patches containing fewer distinct colors yield superior reconstruction quality, while those with more colors exhibit lower quality.
----------------------------------------------------------------

We varied the low-rank value (\(r\)) across \([5, 10, 25, 50]\) for each case, reconstructing the patches using Gradient Descent and comparing them to the original image. Here are our observations:

1. **Single Color Patch**:
   - Single-color patches inherently possess low-rank characteristics.
   - Even with small \(r\) values, the reconstructed patches closely resemble the originals.
   - Minimal differences in reconstruction quality are observed across different \(r\) values.

2. **Multicolor Patch (2-3 Colors)**:
   - Patches with 2-3 colors exhibit increased complexity.
   - Higher \(r\) values capture more color variations in the reconstructed patches.
   - Enhanced reconstruction quality is evident with larger \(r\) values.

3. **Multicolor Patch (At Least 5 Colors)**:
   - Patches with at least 5 colors demonstrate high diversity.
   - Lower \(r\) values struggle to represent the complexity, resulting in noticeable artifacts.
   - Substantial improvements in reconstruction quality are achieved with larger \(r\) values.

**Overall Trends**:
- Increasing \(r\) consistently improves reconstruction quality.
- Single-color patches require only small \(r\) values for satisfactory reconstruction.
- Multicolor patches benefit significantly from higher \(r\) values, particularly when multiple colors are present.