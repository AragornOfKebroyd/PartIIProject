import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
"""
naming convention:
lower/UPPER - constant
lowercase_ - vector
CAPTIAL_ - matrix
"""


def generate_sparse_vector(N, s, distribution, support): # N dimensionality of x_, s is sparsity |x_|_0
    values = distribution(size=s)
    x_ = np.zeros(N)
    x_[support] = values
    return x_


def add_noise(x_, distribution):
    x_ += distribution(size=x_.shape)
    return x_


def generate_matrix(N, M, distribution):
    A_ = distribution(size=(M, N))
    return A_


def OMP_CV(A_, y_, A_cv_, y_cv_, max_iter=100):
    """
    Orthogonal Matching Pursuit with Cross Validation
    A_ : measurement matrix
    y_ : measurements
    A_cv_ : cross validation measurement matrix
    y_cv_ : cross validation measurements
    max_iter : maximum number of iterations
    """
    N = A_.shape[1]
    residual = y_.copy()
    support = []
    x_hat_ = np.zeros(N)

    best_error = np.inf
    best_x = None
    best_support = None
    
    for _ in range(max_iter):
        # Step 1: Find the index of the column that is most correlated with the residual
        correlations = A_.T @ residual
        new_index = np.argmax(np.abs(correlations))
        support.append(new_index)
        
        # Step 2: Solve the least squares problem to find the new estimate of x_
        A_support_ = A_[:, support]
        x_support_, _, _, _ = np.linalg.lstsq(A_support_, y_, rcond=None)
        
        # Step 3: Update the estimate of x_
        x_hat_[support] = x_support_
        
        # Step 4: Update the residual
        residual = y_ - A_support_ @ x_support_
        
        # Step 5: Cross-validation check
        cv_residual = y_cv_ - A_cv_[:, support] @ x_support_
        cv_error = np.linalg.norm(cv_residual)

        if cv_error < best_error:
            best_error = cv_error
            best_x = x_hat_.copy()
            best_support = support.copy()
            
    return best_x, best_support


def column_normalize(A_):
    """
    Normalize the columns of a matrix A_
    """
    for i in range(A_.shape[1]):
        if np.all(A_[:, i] == 0):
            continue
        A_[:, i] = A_[:, i] / np.linalg.norm(A_[:, i])
    return A_

# matrix distribution should have a 1/3 chance of being 0, and 1/3 chance of being 1 and 1/3 chance of being -1
def sparse_matrix_distribution(size, p):
    choices = [0,1.0,-1.0]
    probabilities = [p, (1-p)/2, (1-p)/2]
    return np.random.choice(choices, size=size, p=probabilities)


def gaussian_matrix_distribution(size):
    return np.random.normal(loc=0, scale=1, size=size)

def recover_sparse_vector(x_, N, s, m, p):
    sparse_districution = lambda size: sparse_matrix_distribution(size=size, p=p)
    noise_distribution = lambda size : np.random.normal(loc=0, scale=0.1, size=size)

    Phi = sparse_districution
    Psi = sparse_districution
    

    A_ = generate_matrix(N, m, Phi)
    A_ = column_normalize(A_)

    m_cv = 20

    A_cv_ = generate_matrix(N, m_cv, Psi)
    A_cv_ = column_normalize(A_cv_)

    y_ = add_noise(A_ @ x_, noise_distribution)
    y_cv_ = add_noise(A_cv_ @ x_, noise_distribution)

    x_hat_, support = OMP_CV(A_, y_, A_cv_, y_cv_)    
    
    # find difference between x_hat_ and x_
    error = np.linalg.norm(x_hat_ - x_)
    return error, x_hat_, support

def support_check(true_support, estimated_support):
    overlap = len(set(estimated_support).intersection(set(true_support)))
    # find set difference between support and true support
    false_positives = len(set(estimated_support) - set(true_support))
    # find set difference between true support and support
    false_negatives = len(set(true_support) - set(estimated_support))

    # print sizes
    print(f"True support size: {len(true_support)}")
    print(f"Estimated support size: {len(estimated_support)}")
    print(f"Overlap: {overlap}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}") 


def main():
    N=1000
    s=10
    m=200

    distribution = lambda size : np.random.normal(loc=20, scale=6, size=size)

    # Generate a random support set
    support = np.random.choice(list(range(N)), size=s, replace=False)
    x_ = generate_sparse_vector(N, s, distribution, support)

    errors = []
    # range p from 0.1 to 0.99 in increments of 0.01
    for p in np.arange(0.9, 1.0, 0.01):
        p = min(round(p, 2),1.0)
        print(p)
        current_errors = []
        for i in range(20):
            error, x_hat_, estimated_support = recover_sparse_vector(x_, N, s, m, p)
            if i == 0: support_check(support, estimated_support)
            current_errors.append(error)
        error = np.mean(current_errors)
        print("error for " + str(p) + ": " + str(error))
        std_error = np.std(current_errors)
        errors.append({
            "p": p,
            "error": error,
            "std_error": std_error
        })

    df = pd.DataFrame(errors)
    # plot error vs p with error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(df["p"], df["error"], yerr=df["std_error"], fmt='o', alpha=0.5)
    plt.xlabel("p")
    plt.ylabel("Reconstruction Error")
    plt.title("Reconstruction Error vs p")
    plt.grid()
    plt.savefig("reconstruction_error_vs_p.png")
    
    



if __name__ == "__main__":
    main()