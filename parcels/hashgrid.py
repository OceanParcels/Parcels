

class HashGrid:
    """Class for creating a hash grid for fast particle lookup.
    
    Parameters
    ----------
    x0 : float
        x-coordinate of the lower left corner of the grid.
    y0 : float
        y-coordinate of the lower left corner of the grid.
    dx : float
        Grid spacing in x-direction.
    dy : float
        Grid spacing in y-direction.
    nx : int
        Number of grid points in x-direction.
    ny : int
        Number of grid points in y-direction  
    ugrid_elements : list
        List of lists of unstructured grid indices in each hash cell.
    """

    def __init__(self,x0,y0,dx,dy,nx,ny):
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.nx = nx
        self.ny = ny
        ugrid_elements = [[] for i in range(self.nx*self.ny)]



    def get_hashindex_for_xy(self,x,y):
        """Get the grid indices for a given x and y coordinate."""
        i = int((x-self.x0)/self.dx)
        j = int((y-self.y0)/self.dy)
        return i+self.nx*j
    
    
    def populate_ugrid_elements(self,vertices, elements):
        """
        Efficiently find the list of triangles whose bounding box overlaps with the specified hash cells.
        
        Parameters:
        - vertices (np.ndarray): Array of vertex coordinates of shape (n_vertices, 2).
        - elements (np.ndarray): Array of element-to-vertex connectivity, where each row contains 3 indices into the vertices array.
        
        Returns:
        - overlapping_triangles (dict): A dictionary where keys are the hash cell index and values are lists of triangle indices.
        """
        import numpy as np
        
        overlapping_triangles = [[] for i in range(self.nx*self.ny)]
        
        # Loop over each triangle element
        for triangle_idx, triangle in enumerate(elements):
            # Get the coordinates of the triangle's vertices
            triangle_vertices = vertices[triangle]
            
            # Calculate the bounding box of the triangle
            x_min, y_min = np.min(triangle_vertices, axis=0)
            x_max, y_max = np.max(triangle_vertices, axis=0)
            
            # Find the hash cell range that overlaps with the triangle's bounding box
            i_min = int(np.floor((x_min-self.x0) / self.dx))
            i_max = int(np.floor((x_max-self.x0) / self.dx))
            j_min = int(np.floor((y_min-self.y0) / self.dy))
            j_max = int(np.floor((y_max-self.y0) / self.dy))
            
            # Iterate over all hash cells that intersect the bounding box
            for j in range(j_min, j_max + 1):
                for i in range(i_min, i_max + 1):
                    overlapping_triangles[i+self.nx*j].append(triangle_idx)

        self.ugrid_elements = overlapping_triangles