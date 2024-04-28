Q3)
Ans:

Sklearn's LinearRegression uses a more advanced algorithm called the Ordinary Least Squares (OLS) method, similar to the normal equation, but with additional enhancements. It internally handles the issue of multicollinearity by employing Singular Value Decomposition (SVD).
The goal of linear regression is to find the coefficients θ that minimize the sum of squared differences between the predicted values and the actual values. This is typically formulated as the least squares problem.
The ordinary least squares solution involves solving the system of equations: $$ X^TX\theta = X^Ty $$ for θ, where X is the feature matrix and y is the target variable.

The SVD method decomposes the feature matrix X into three matrices U, Σ, and V, such that X = UΣV. The coefficients $\theta$ are then calculated as $\theta = V \cdot \text{diag}\left(\frac{1}{\sigma_i}\right) \cdot U^T \cdot y$, where $\sigma_i$ are the singular values obtained from the diagonal of $\Sigma$.
The SVD approach is numerically stable, hence works even for ill-conditioned matrices, i.e. matrices whose ratio of largest to smallest singular value in the matrix is very high.
The implementation includes checks for edge-cases, such as singular or nearly singular matrices which may lead to issues in the matrix inversion process.
