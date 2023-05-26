import numpy as np
import random
from scipy.stats import norm

def GGM_instance(n=100, p=100, max_edges=10):
    def generate_vertices(p):
        vertices = np.random.uniform(size=(p,2))
        return vertices
    def connecting_prob(v1,v2,p):
        # Euclidean distance of v1, v2
        d = np.linalg.norm(v1-v2)
        # calculating connecting probability
        prob = norm.pdf(d/np.sqrt(p))
        return prob
    def remove_edges(p, adj, max_edges):
        idx = list(range(p))
        np.random.shuffle(idx)

        for i in idx:
            if np.all(np.sum(adj, axis=1) <= (max_edges+1)):
                break
            # Indices of nodes connected to v_i
            nonzero_i = list(np.nonzero(adj[i])[0])
            n_edges = len(nonzero_i)

            # Delete some edges if there are redundancies
            if n_edges > (max_edges+1):
                nonzero_i.remove(i)
                removed_idx_i = random.sample(nonzero_i,n_edges-max_edges)
                # Remove other edges
                adj[i,removed_idx_i] = 0
                adj[removed_idx_i,i] = 0

        return adj

    vertices = generate_vertices(p)

    adj_mat = np.eye(p)

    for i in range(p):
        for j in range(i+1,p):
            v_i = vertices[i]
            v_j = vertices[j]
            adj_mat[i,j] = np.random.binomial(n=1,
                                              p=connecting_prob(v1=v_i,
                                                                v2=v_j,
                                                                p=p))

    # symmetrize
    adj_mat = adj_mat + adj_mat.T - np.eye(p)

    # remove redundant edges
    adj_mat = remove_edges(p, adj_mat, max_edges)

    # maximal off-diag value to guarantee diagonal dominance
    max_off_diag = 1/max_edges
    max_off_diag = max_off_diag*0.9

    # generate a PD precision
    precision = np.random.uniform(low=-max_off_diag,high=max_off_diag,
                                  size=(p,p))
    # precision = max_off_diag * (np.random.binomial(n=1,p=0.5,size=(p, p)) * 2 - 1)
    # symmetrize precision
    precision = np.tril(precision)
    precision = precision + precision.T
    # sparsify precision based on adjacency matrix
    precision = precision * adj_mat
    np.fill_diagonal(precision, 1)
    cov = np.linalg.inv(precision)

    # standardize the covariance
    cov = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
    precision = np.linalg.inv(cov)

    X = np.random.multivariate_normal(mean=np.zeros(p),
                                      cov=cov, size=n)

    return precision*n, cov/n, X/np.sqrt(n)