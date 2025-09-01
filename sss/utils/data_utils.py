import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import Voronoi, cKDTree
from scipy.ndimage import zoom

def random_2d_gaussian(shape, zoom_factor, sigma, clip=3, norm_min=0, norm_max=1):
    # generate a random 3d gaussian with the given shape and sigma
    # zoom is the zoom factor
    # return a 3d gaussian with the given shape and sigma
    small_shape = tuple(int(s / zoom_factor)+1 for s in shape)
    x = np.random.randn(*small_shape).astype(np.float32)
    x = zoom(x, zoom_factor, order=0)[:shape[0], :shape[1]]

    if sigma > 0:
        x = gaussian_filter(x, sigma, order=0)

    x = np.clip(x, -clip, clip)

    # convert to (0, 1)
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    x = x * (norm_max - norm_min) + norm_min
    return x

def generate_voronoi_map(shape, num_points, norm_min=0.0, norm_max=1.0, seed=None):
    height, width = shape
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random control points
    points = np.random.rand(num_points, 2) * [width, height]
    # Create Voronoi diagram
    vor = Voronoi(points)
    
    # Generate image coordinate grid
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    coords = np.stack((xx, yy), axis=-1).reshape(-1, 2)

    # Find nearest Voronoi point for each pixel
    tree = cKDTree(points)
    _, regions = tree.query(coords)
    
    # Assign random value per region
    values = np.random.uniform(norm_min, norm_max, size=num_points).astype(np.float32)
    voronoi_map = values[regions].reshape((height, width))
    
    return voronoi_map

def random_line_src(sh, wavelength, dL, n_angles=6, pml_x=40, pml_y=40, direction='h', source_PML_spacing=0):
    assert direction in ['h', 'v']

    Nx, Ny = sh

    source_amp = 1/dL/dL
    k0 = 2 * np.pi / wavelength

    source_len = np.random.randint(1, (Ny-2*pml_y)//2 if direction=='h' else (Nx-2*pml_x)//2)
    source_vec = np.zeros(source_len, dtype=complex)

    for i in range(n_angles):
        angle_deg = np.random.randint(-90,90)
        angle_rad = angle_deg * np.pi / 180
        # Compute the wave vector
        kx = k0 * np.cos(angle_rad)
        ky = k0 * np.sin(angle_rad)

        # Get an array of the y positions across the simulation domain
        L = source_len * dL
        vec = np.linspace(0, L, source_len)
        
        # Make a new source where source[x] ~ exp(i * kx * x) to simulate an angle
        phase = 2*np.pi*np.random.random()
        source_vec += np.exp(1j * ky * vec + 1j*phase) if direction=='h' else np.exp(1j * kx * vec + 1j*phase)
    
    source_vec *= source_amp*1/n_angles

    rand_source = np.zeros(sh, dtype=np.complex64)

    start_x = np.random.randint(pml_x+1+source_PML_spacing, Nx-pml_x-2-(source_len if direction=="v" else 0)-source_PML_spacing)
    start_y = np.random.randint(pml_y+1+source_PML_spacing, Ny-pml_y-2-(source_len if direction=="h" else 0)-source_PML_spacing)
    if direction=='h':
        rand_source[start_x,start_y:start_y+source_len] = source_vec
    else:
        rand_source[start_x:start_x+source_len,start_y] = source_vec

    return rand_source