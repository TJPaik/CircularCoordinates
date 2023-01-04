import numpy as np
from ripser import ripser
from scipy import sparse, spatial


def ripsr(data, prime):
    """Convenience function to compute Vietoris–Rips persistence barcodes using the ripser library

    Parameters
    ----------
    data : ndarray or pandas dataframe
         Input data.
     prime : int
         Prime basis to compute homology.

    Returns
    -------
    dict
        result of ripser.

    """
    if isinstance(data, np.ndarray):
        data1 = data
    else:
        data1 = data.to_numpy()
    rips = ripser(data1, coeff=prime, do_cocycles=True)
    return rips


def errs(str_, err_, errs_):
    if err_ not in errs_:
        raise ValueError(str_ + ' can only be ' + str(errs_)[1:-1])


class circular_coordinate():
    """This is the  main class for circular-coordinates library used to create and plot circular coordinates from persistent cohomology.


    Parameters
    ----------
    prime : int
        The prime basis over which to compute homology

    Attributes
    ----------
    prime : int
        The prime basis over which to compute homology
    ripsr : dict
        Result of ripser on input data
    vertex_values : list
        List of circular coordinates

    """

    def __init__(self, prime):
        self.prime = prime

    def get_epsilon(self, dgms):
        """Finds index of epsilon from the homology persistence diagram

        Parameters
        ----------
        dgms:ndarray or list
            List of birth and death of persistence barcodes.

        
        Returns
        -------
        int
           The index of epsilon.

        """
        arg_eps = max(enumerate(dgms), key=lambda pt: pt[1][1] - pt[1][0])[0]
        return arg_eps

    def boundary_cocycles(self, rips, epsilon, prime, spec=None):

        """Convenience function that combines 'delta' and 'cocycles'
        
        Parameters
        ----------
        rips:dict
            The result from ripser(computing the vitoris rips complex) on the input data.
        prime:int
            The prime basis over which to compute homology.
        epsilon:float
            Epsilon used to truncate the vitoris rips
        spec:int
            The index of the specific cocycle to extract. If spec is 'None' all cocycles will be retuned

        
        Returns
        -------
        Delta:ndarray
            boundary (𝛿⁰)
        cocycles:ndarray
            specific cocycle or all cocycles

        """

        distances = rips["dperm2all"]
        edges = np.array((distances <= epsilon).nonzero()).T
        new_dist = distances[edges[:, 0], edges[:, 1]]

        # Construct 𝛿⁰
        Delta = self.delta(distances, edges)

        # Extract the cocycles
        dshape = Delta.shape[0]
        cocycles = self.cocycles(rips, distances, edges, dshape, prime, spec)
        return Delta, cocycles, new_dist

    def delta(self, distances, edges):

        """Constructs the boundary (𝛿⁰)
        
        Parameters
        ----------

        distances:ndarray
            Distances from points in the ripser greedy permutation to points in the original data point set
        edges:ndarray
            Distances truncated to epsilon

        Returns
        -------
        ndarray
            boundary (𝛿⁰)
        """
        I = np.c_[np.arange(edges.shape[0]), np.arange(edges.shape[0])]
        I = I.flatten()
        J = edges.flatten()
        V = np.c_[-1 * np.ones(edges.shape[0]), np.ones(edges.shape[0])]
        V = V.flatten()
        Delta = sparse.coo_matrix((V, (I, J)), shape=(edges.shape[0], distances.shape[0]))
        return Delta

    def cocycles(self, rips, distances, edges, dshape, prime, spec=None):

        """Extracts cocycles
        
        Parameters
        ----------

        rips :dict
            The result from ripser on the input data.
        distances: ndarray
            Distances from points in the greedy permutation to points in the original point set
        edges: ndarray
            Distances truncated to epsilon
        dshape: int
            Number of rows in boundary (𝛿⁰)
        prime : int
            The prime basis over which to compute homology.
        spec: int
            The index of the specific cocycle to extract. If spec is 'None' all cocycles will be retuned


        Returns
        -------
        ndarray
            specific cocycle or all cocycles
        """

        cocycles = []
        if spec is not None:

            cocycle = rips["cocycles"][1][spec]
            val = cocycle[:, 2]
            val[val > (prime - 1) / 2] -= prime
            Y = sparse.coo_matrix((val, (cocycle[:, 0], cocycle[:, 1])), shape=(distances.shape[0], distances.shape[0]))
            Y = Y - Y.T
            Z = np.zeros((dshape,))
            Z = Y[edges[:, 0], edges[:, 1]]
            return [Z]
        else:
            for cocycle in rips["cocycles"][1]:
                val = cocycle[:, 2]
                val[val > (prime - 1) / 2] -= prime
                Y = sparse.coo_matrix((val, (cocycle[:, 0], cocycle[:, 1])),
                                      shape=(distances.shape[0], distances.shape[0]))
                Y = Y - Y.T
                Z = np.zeros((dshape,))
                Z = Y[edges[:, 0], edges[:, 1]]
                cocycles.append(Z)
                return cocycles

    def minimize(self, Delta, cocycle, new_dist, weight=None):

        """Minimizes ∥ζ-𝛿1α∥2 and computes vertex values map to [0,1]
        
        Parameters
        ----------

        Delta : ndarray
            Boundary (𝛿⁰)
        cocycle: ndarray
            Corresponding cocycle (ζ)

        Returns
        -------
        ndarray
            circular coordinates mapped to [0,1] interval
        """
        if weight is not None:
            edge_weight = weight(Delta, cocycle, new_dist)
            new_delta = Delta.multiply(edge_weight[:, None])
            mini = sparse.linalg.lsqr(new_delta, edge_weight * np.asarray(cocycle).reshape(-1))
            return np.array(mini[0]), cocycle
        else:
            mini = sparse.linalg.lsqr(Delta, np.array(cocycle).squeeze())
            # vertex_values = np.mod(np.array(mini[0]), 1.0)
            # return vertex_values, cocycle
            return np.array(mini[0]), cocycle

    def circular_coordinate(self, rips, prime, vertex_values=None, arg_eps=None, check=None, intr=10, weight=None,
                            eps=None):

        """Computes and plots the circular_coordinates
        
        Parameters
        ----------

            rips : dict
                The result from ripser(computing the vitoris rips complex) on the input data.
            prime : int
                The prime basis over which to compute homology.
            vertex_values : ndarray
                Circular coordinates of the longest barcode
            arg_eps: int
                Index of epsilon
            check: string
                Can be 'All' or 'Max' or None:
                    All: compute circular coordinates for all persistence barcodes
                    Max: compute circular coordinates over the largest persistence barcode
                    None:  compute circular coordinates for the largest persistence barcode or epsilon
            intr: int
                Only required if check is 'Max', specifies as to how many points to compute coordinates over the max persistence barcode
            
       
        Returns
        -------
            vertex_values : ndarray
                circular coordinates
            max_/all_ : ndarray
                list of circular coordinates of all persistence barcodes or binned from the longest barcode based on the 'check' input
            all_list/max_list: ndarray
                list of epsilon for which the circular coordinates are computed 
        """
        checks = ['All', 'Max', None]
        errs('check', check, checks)

        if check == 'All':

            if vertex_values is None:
                vertex_values = self.circular_coordinate(rips, prime)

            all_, _ = self.all_verts(rips, prime, vertex_values)
            return all_, [x[1] for x in rips['dgms'][1]]

        elif check == 'Max':

            if vertex_values is None:
                vertex_values = self.circular_coordinate(rips, prime)

            max_, max_list, _ = self.max_verts(rips, prime, vertex_values, intr=intr)

            return max_, max_list
        else:
            if arg_eps is None:
                arg_eps = self.get_epsilon(rips['dgms'][1])
            if eps is None:
                eps = rips["dgms"][1][arg_eps][1]
            delta, cocycles, new_dist = self.boundary_cocycles(rips, eps, prime, arg_eps)
            vertex_values, cocycle_mat = self.minimize(delta, cocycles, new_dist, weight=weight)
            return delta, vertex_values, cocycle_mat

    def all_verts(self, rips, prime, init_verts, dist=None):

        """Function to find circular coordinates for all persistence barcodes
        
        Parameters
        ----------
        rips : dict
            The result from ripser(computing the vitoris rips complex) on the input data.
        prime : int
            The prime basis over which to compute homology.
        init_verts: ndarray
            Circular coordinates of the longest persistence barcode 
        dist : string
            Distance calculation metric betweeen the circular coordinates: 'l1','l2', 'cosine' or 'None'


        Returns
        -------
        all_ : list
            Circular coordinates for all persistence barcodes
        dist_ : list
            Distances between the the circular coordinates
        """
        dists = ['l1', 'l2', 'cosine', None]
        errs('dist', dist, dists)
        all_ = []
        dist_ = []
        for indi, eps in enumerate(rips["dgms"][1]):
            eps = eps[1]
            delta, cocycles = self.boundary_cocycles(rips, eps, prime, indi)
            vertex_values, cocycle_mat = self.minimize(delta, cocycles)
            all_.append(vertex_values)

            if dist is not None:
                dist_.append(self.get_dist(init_verts, vertex_values, dist))

        return all_, dist_

    def max_verts(self, rips, prime, init_verts, intr=10, dist=None):

        """Function to find circular coordinates over the largest persistence barcode
        
        Parameters
        ----------
        rips : dict
            The result from ripser(computing the vitoris rips complex) on the input data.
        prime : int
            The prime basis over which to compute homology.
        init_verts: ndarray
            Circular coordinates of the longest persistence barcode 
        intr : int
            Specifies as to how many points to compute the circular coordinates of over the largest persistence barcode
        dist : string
            Distance calculation metric betweeen the circular coordinates: 'l1','l2', 'cosine' or 'None'

        Returns
        -------
        max_ : list
            Circular coordinates over the largest persistence barcode

        arr: ndarray
            List of points used between the birth and death of the largest barcode

        dist_ : list
            Distances between the the circular coordinates
        

        """

        dists = ['l1', 'l2', 'cosine', None]
        errs('dist', dist, dists)
        max_ = []
        dist_ = []
        arg_eps = self.get_epsilon(rips['dgms'][1])
        eps_ = rips["dgms"][1][arg_eps]
        arr = np.linspace(eps_[0], eps_[1], intr)
        for ep in arr:
            delta, cocycles = self.boundary_cocycles(rips, ep, prime, arg_eps)
            vertex_values, cocycle_mat = self.minimize(delta, cocycles)
            max_.append(vertex_values)

            if dist is not None:
                dist_.append(self.get_dist(init_verts, vertex_values, dist))

        return max_, arr, dist_

    def get_dist(self, init_verts, vert, dist='l1'):

        """Convenience function to find distance between two sets of circular coordinates
        
        Parameters
        ----------
        init_verts : ndarray
            The first array of circular coordinates
        vert : ndarray
            The second array of circular coordinates
        dist : string
            Distance calculation metric betweeen the circular coordinates: 'l1','l2' or 'cosine'

        Returns
        -------
        float
            distance

        

        """
        if dist == 'l1':
            d = np.linalg.norm(vert - init_verts, ord=1)
        elif dist == 'l2':
            d = np.linalg.norm(vert - init_verts)
        elif dist == "cosine":
            d = spatial.distance.cosine(vert, init_verts)
        return d

    def fit_transform(self, data, weight=None):
        """Function to find circular coordinates from persistent cohomology of input data

        Parameters
        ----------

         data :(ndarray or pandas dataframe)
            Input data

        Returns
        -------
        list
            List of circular coordinates


        """

        self.rips = ripsr(data, self.prime)
        _, self.vertex_values, _ = self.circular_coordinate(self.rips, self.prime, weight=weight)
        return self.vertex_values
