import numpy as np
import pandas as pd
import xgboost as xgb
import random
from sklearn.preprocessing import LabelEncoder

class ThreadPredictor:
    def __init__(self, model_path="thread_count_predictor.bst"):
        # Load the trained XGBoost model from .bst file
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        # Initialize encoders for categorical features
        self.op_encoder = LabelEncoder()
        self.op_encoder.classes_ = np.array(['Multiplication', 'Addition', 'Transposition', 
                                           'Determinant', 'Inverse', 'Eigenvalues', 
                                           'LU Decomposition', 'Cholesky Decomposition', 
                                           'QR Decomposition', 'Singular Value Decomposition (SVD)'])
        
        self.var_type_encoder = LabelEncoder()
        self.var_type_encoder.classes_ = np.array(['int', 'float', 'double', 'complex'])
        
        self.matrix_type_encoder = LabelEncoder()
        self.matrix_type_encoder.classes_ = np.array(['Dense', 'Sparse', 'Diagonal', 'Symmetric', 'Triangular'])
    
    def estimate_execution_time(self, type_op, matrix_size, var_type, matrix_type):
        # Simplified estimations based on operation type and size
        base_times = {
            'Multiplication': 1e-6 * matrix_size**2.5,
            'Addition': 1e-7 * matrix_size**2,
            'Transposition': 1e-7 * matrix_size**2,
            'Determinant': 1e-6 * matrix_size**3,
            'Inverse': 1e-6 * matrix_size**3,
            'Eigenvalues': 1e-6 * matrix_size**3,
            'LU Decomposition': 1e-6 * matrix_size**3,
            'Cholesky Decomposition': 1e-6 * matrix_size**3,
            'QR Decomposition': 1e-6 * matrix_size**3,
            'Singular Value Decomposition (SVD)': 1e-6 * matrix_size**3
        }
        
        # Apply type modifiers
        type_modifier = {
            'int': 0.8,
            'float': 1.0,
            'double': 1.2,
            'complex': 2.5
        }
        
        # Apply matrix type modifiers
        matrix_modifier = {
            'Dense': 1.0,
            'Sparse': 0.5,  # Sparse operations are generally faster
            'Diagonal': 0.3,
            'Symmetric': 0.8,
            'Triangular': 0.7
        }
        
        estimated_time = base_times[type_op] * type_modifier[var_type] * matrix_modifier[matrix_type]
        
        # Ensure the result is within the dataset's observed range
        return min(max(estimated_time, 0.001), 0.699)
    
    def estimate_num_operations(self, type_op, matrix_size):
        # Approximate operation counts based on algorithm complexity
        op_complexity = {
            'Multiplication': matrix_size**3,
            'Addition': matrix_size**2,
            'Transposition': matrix_size**2,
            'Determinant': (2/3) * matrix_size**3,  # ~ O(n³) for LU decomposition
            'Inverse': matrix_size**3,
            'Eigenvalues': 10 * matrix_size**3,  # Iterative methods
            'LU Decomposition': (2/3) * matrix_size**3,
            'Cholesky Decomposition': (1/3) * matrix_size**3,
            'QR Decomposition': 4 * matrix_size**3,
            'Singular Value Decomposition (SVD)': 12 * matrix_size**3
        }
        
        # Scale to dataset range (1 to 100 million)
        operations = op_complexity[type_op]
        scaling_factor = 100
        
        return min(max(int(operations * scaling_factor), 1), 100_000_000)
    
    def estimate_num_variables(self, matrix_size):
        # A simple heuristic: number of variables is proportional to matrix size
        estimated_vars = matrix_size * 1.2
        
        # Ensure the result is within the dataset's observed range
        return min(max(int(estimated_vars), 1), 1000)
    
    def estimate_complexity(self, type_op):
        # Theoretical complexity values based on algorithm asymptotic behavior
        complexity_map = {
            'Multiplication': 3.0,  # O(n³) 
            'Addition': 2.0,        # O(n²)
            'Transposition': 2.0,   # O(n²)
            'Determinant': 3.0,     # O(n³)
            'Inverse': 3.0,         # O(n³)
            'Eigenvalues': 3.0,     # O(n³)
            'LU Decomposition': 3.0,
            'Cholesky Decomposition': 3.0,
            'QR Decomposition': 3.0,
            'Singular Value Decomposition (SVD)': 3.0
        }
        
        # Add some randomness to match dataset distribution
        variation = random.uniform(-0.3, 0.0)
        
        return complexity_map[type_op] + variation
    
    def determine_is_iterative(self, type_op):
        # Some operations are inherently iterative
        iterative_ops = ['Eigenvalues', 'Singular Value Decomposition (SVD)']
        mostly_iterative = ['Inverse', 'QR Decomposition']
        
        if type_op in iterative_ops:
            return 1
        elif type_op in mostly_iterative:
            return random.choice([0, 1, 1])  # 2/3 chance of being iterative
        else:
            return random.choice([0, 0, 1])  # 1/3 chance of being iterative
    
    def determine_memory_pattern(self, type_op, matrix_type):
        # Memory-intensive operations tend to create temporary matrices
        high_memory_ops = ['Multiplication', 'Inverse', 'Eigenvalues', 'QR Decomposition', 'Singular Value Decomposition (SVD)']
        
        # Sparse and diagonal matrices are more memory efficient
        low_memory_types = ['Sparse', 'Diagonal']
        
        if type_op in high_memory_ops and matrix_type not in low_memory_types:
            return random.choice([0, 1, 1, 1])  # 75% chance of high memory
        else:
            return random.choice([0, 0, 0, 1])  # 25% chance of high memory
    
    def predict(self, type_op, matrix_size, var_type, matrix_type, is_iterative=None, memory_pattern=None):
        """
        Predicts optimal thread count for matrix operations based on user inputs.
        
        Parameters:
        type_op (str): Type of matrix operation
        matrix_size (int): Size of the matrix (n for n×n)
        var_type (str): Variable type ('int', 'float', 'double', 'complex')
        matrix_type (str): Type of matrix ('Dense', 'Sparse', 'Diagonal', 'Symmetric', 'Triangular')
        is_iterative (int, optional): Whether operation is iterative (0 or 1)
        memory_pattern (int, optional): Memory usage pattern (0 or 1)
        
        Returns:
        int: Predicted optimal thread count (1-8)
        dict: The estimated features used for prediction
        """
        # Estimate all required features
        exe_time = self.estimate_execution_time(type_op, matrix_size, var_type, matrix_type)
        num_op = self.estimate_num_operations(type_op, matrix_size)
        num_var = self.estimate_num_variables(matrix_size)
        complexity = self.estimate_complexity(type_op)
        
        # Determine categorical features if not provided
        if is_iterative is None:
            is_iterative = self.determine_is_iterative(type_op)
        
        if memory_pattern is None:
            memory_pattern = self.determine_memory_pattern(type_op, matrix_type)
        
        # Encode categorical features
        encoded_type_op = self.op_encoder.transform([type_op])[0]
        encoded_var_type = self.var_type_encoder.transform([var_type])[0]
        encoded_matrix_type = self.matrix_type_encoder.transform([matrix_type])[0]
        
        # Create feature array for prediction
        features = pd.DataFrame({
            'typeOp': [encoded_type_op],
            'matrixSize': [matrix_size],
            'numOp': [num_op],
            'numVar': [num_var],
            'varType': [encoded_var_type],
            'exeTime': [exe_time],
            'isIterative': [is_iterative],
            'memoryPattern': [memory_pattern],
            'complexity': [complexity],
            'matrixType': [encoded_matrix_type]
        })
        
        # Convert DataFrame to DMatrix for XGBoost prediction
        dmatrix = xgb.DMatrix(features)
        
        # Make prediction
        prediction = self.model.predict(dmatrix)
        optimal_threads = prediction[0]
        
        # Return the prediction and estimated features
        estimated_features = {
            'exeTime': exe_time,
            'numOp': num_op,
            'numVar': num_var,
            'complexity': complexity,
            'isIterative': is_iterative,
            'memoryPattern': memory_pattern
        }
        
        # Round to nearest integer within valid range (1-8)
        return max(1, min(8, round(optimal_threads))), estimated_features

# Example usage
if __name__ == "__main__":
    # Create an instance of the predictor
    thread_predictor = ThreadPredictor()
    
    # Make a prediction
    optimal_threads, features = thread_predictor.predict(
        type_op='Multiplication',
        matrix_size=500,
        var_type='double',
        matrix_type='Dense'
    )
    
    print(f"Predicted optimal thread count: {optimal_threads}")
    print("Estimated features:")
    for feature, value in features.items():
        print(f"  {feature}: {value}")