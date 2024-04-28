Q2) 
Ans: 
The normal equation for linear regression exists as:
$$
\theta = (X^TX)^{-1}X^TY
$$

In the original notebook, we used the 'np.linalg.inv' function calculate the inverse which might be computationally expensive and less stable for large matrices. Thus, we may also use 'np.linalg.solve' method directly to solve the above system of linear equations.
If we were to compare both the functions and their utilities, 'np.linalg.inv' method involves the calculation of matrix inverse and thus has a time complexity of the order of O(n^3), where n is the size of the matrix. This is computationally expensive for large matrices, as it requires more operations.
Whereas for 'np.linalg.solve' method, the system of linear equations is solved directly using the specialized algorithm, LU Decomposition, which lowers the time complexity than matrix inversion. 

Also, the process of inverting a matrix may lead to numerical instability in cases of a singular matrix, i.e., when the determinant of the matrix is zero or approaches zero. Whereas there is no such case in the 'np.linalg.solve' method unlike the 'np.linalg.inv' method.
