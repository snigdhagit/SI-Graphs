import numpy as np
import random
from scipy.stats import norm
def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

def GGM_instance(n=100, p=100, max_edges=10, signal=1.):
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

    # generate a PD precision
    precision = np.random.uniform(size=(p,p), low=0, high=max_off_diag*signal)

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

    return precision, cov, X

def GGM_random_instances(n=200, p=50, theta=-0.2):

    # Guarantee same sparsity level as in Friedman et al.:
    # https://www.asc.ohio-state.edu/statistics/statgen/joul_aut2015/2010-Friedman-Hastie-Tibshirani.pdf
    prob = 0.4 / (np.abs(theta)*p)

    invertible = False

    # Generate invertible precision
    while not invertible:
        prec = np.eye(p)

        # Randomly selecting edges
        for i in range(p):
            for j in range(i + 1, p):
                prec[i, j] = theta * np.random.binomial(n=1, p=prob)

        # symmetrize
        prec = prec + prec.T - np.eye(p)

        invertible = is_invertible(prec)

    cov = np.linalg.inv(prec)
    # standardize the covariance
    cov = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
    prec = np.linalg.inv(cov)

    X = np.random.multivariate_normal(mean=np.zeros(p),
                                      cov=cov, size=n)

    return prec, cov, X

def GGM_hub_instances(n=200, p=50, K=10, theta=-0.175):
    group_size = int(p / K)

    invertible = False
    while not invertible:
        prec = np.eye(p)
        for k in range(K):
            group_k = range(k * group_size, (k + 1) * group_size)
            hub = random.sample(list(group_k), 1)[0]
            for i in group_k:
                # fix column at hub, iterate over all rows in the group
                if i != hub:
                    prec[i, hub] = theta

        # symmetrize
        prec = prec + prec.T - np.eye(p)

        invertible = is_invertible(prec)

    cov = np.linalg.inv(prec)
    # standardize the covariance
    cov = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
    prec = np.linalg.inv(cov)

    X = np.random.multivariate_normal(mean=np.zeros(p),
                                      cov=cov, size=n)

    return prec, cov, X

def GGM_clique_instances(n=200, p=400, K=20, group_size=7, theta=-0.175):
    # Partition [p] into p/K (big_group_size) disjoint sets,
    # then choose a fixed-size subset of each disjoint set

    assert K * group_size < p
    big_group_size = int(p/K)

    invertible = False
    while not invertible:
        prec = np.eye(p)
        for k in range(K):
            group_k = range(k * big_group_size, (k + 1) * big_group_size)
            variables_k = np.random.choice(group_k,
                                           size=group_size, replace=False)
            for i in variables_k:
                for j in variables_k:
                    # Set theta_ij = theta
                    if i != j:
                        prec[i, j] = theta

        invertible = is_invertible(prec)

    cov = np.linalg.inv(prec)
    # standardize the covariance
    cov = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
    prec = np.linalg.inv(cov)

    X = np.random.multivariate_normal(mean=np.zeros(p),
                                      cov=cov, size=n)

    return prec, cov, X
