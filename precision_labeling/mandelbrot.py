import numpy as np
from PIL import Image

def generate_mandelbrot_boundary(width=256):
    # 1. Define Coordinates and Ranges
    # Range: [-2 - 1.2i, 0.5 + 1.2i]
    real_min, real_max = -1.9, 0.5
    imag_min, imag_max = -1.2, 1.2
    
    # Calculate height based on aspect ratio to keep pixels square
    real_range = real_max - real_min
    imag_range = imag_max - imag_min
    pixel_size = real_range / width
    height = int(width * (imag_range / real_range))
    
    print(f"Generating image {width}x{height}...")

    # Create coordinate grid
    x = np.linspace(real_min, real_max, width)
    y = np.linspace(imag_min, imag_max, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    # 2. Initialization based on Pseudo-Code
    # c = p.affix
    # z = c
    Z = C.copy()
    
    # der_c = 1+0j
    Der_C = np.ones_like(C, dtype=np.complex128)
    
    # Track which pixels have escaped and their final states
    escaped_mask = np.zeros_like(C, dtype=bool)
    final_z = np.zeros_like(C, dtype=np.complex128)
    final_der = np.zeros_like(C, dtype=np.complex128)
    
    # Parameters
    N = 10000        # Max iterations
    R = 100.0      # Escape radius (Squared > R*R). Higher R gives better DEM precision.
    thickness_factor = 1.0
    
    # 3. Iteration Loop
    # We use a mask 'active' to only calculate for points that haven't escaped yet
    active = np.ones_like(C, dtype=bool)

    for n in range(N):
        if not np.any(active):
            break
            
        # Extract active values to save computation
        z_curr = Z[active]
        c_curr = C[active]
        der_curr = Der_C[active]
        
        # Check escape condition: squared_modulus(z) > R*R
        # Note: abs(z)**2 is squared modulus
        rsq = z_curr.real**2 + z_curr.imag**2
        escaped_now_indices = rsq > (R * R)
        
        # Determine full-grid indices of points that just escaped
        # Subset of active points that escaped
        active_indices = np.nonzero(active)
        # Of those active, which ones escaped?
        just_escaped_flat = np.where(escaped_now_indices)[0]
        
        # Map back to 2D coordinates
        rows = active_indices[0][just_escaped_flat]
        cols = active_indices[1][just_escaped_flat]
        
        # Store final states for those that escaped
        escaped_mask[rows, cols] = True
        final_z[rows, cols] = z_curr[escaped_now_indices]
        final_der[rows, cols] = der_curr[escaped_now_indices]
        
        # Remove escaped points from active processing
        # We need to map boolean 'escaped_now_indices' (relative to active subset)
        # back to the full 'active' mask.
        # An easier way in numpy is to just update active mask generally:
        active[rows, cols] = False
        
        # Update Z and Der for those still active (not escaped_now)
        # Filter the current batch
        keep_indices = ~escaped_now_indices
        
        z_next = z_curr[keep_indices]
        c_next = c_curr[keep_indices]
        der_next = der_curr[keep_indices]
        
        # new_der_c = der_c*2*z + 1
        # Note: Using old z (z_curr) as per pseudo-code structure implies
        # der update happens before z becomes z^2+c? 
        # Pseudo code:
        #   new_z = z*z+c
        #   new_der_c = der_c*2*z + 1 (z here is old z)
        #   z = new_z
        der_next = der_next * 2 * z_next + 1
        
        # new_z = z*z+c
        z_next = z_next * z_next + c_next
        
        # Update main arrays
        # (This relies on the fact that active points are updated sequentially in memory usually,
        # but explicit indexing is safer)
        # We find indices of points that remain active
        remain_rows = active_indices[0][keep_indices]
        remain_cols = active_indices[1][keep_indices]
        
        Z[remain_rows, remain_cols] = z_next
        Der_C[remain_rows, remain_cols] = der_next

    # 4. Coloring Logic
    # Default color: White (255)
    # The pseudo-code says:
    # if reason == NOT_ENOUGH_ITERATES (Inside): not_enough_iterates_color
    # else (Outside): Check distance.
    
    # Initialize image buffer with White (Background)
    img_data = np.full(C.shape, 255, dtype=np.uint8)
    
    # Process only escaped points (Outside)
    # Those that didn't escape (escaped_mask == False) remain White
    
    z_out = final_z[escaped_mask]
    der_out = final_der[escaped_mask]
    
    rsq = z_out.real**2 + z_out.imag**2
    # Avoid log(0)
    rsq = np.maximum(rsq, 1e-10) 
    
    # Condition:
    # if rsq*(log(rsq)**2) < squared_modulus(thickness_factor*pixel_size*der_c)
    
    lhs = rsq * (np.log(rsq)**2)
    
    # squared_modulus(complex) = real^2 + imag^2 = abs()^2
    rhs_complex = thickness_factor * pixel_size * der_out
    rhs = rhs_complex.real**2 + rhs_complex.imag**2
    
    # Identify boundary pixels
    boundary_condition = lhs < rhs
    
    # Map boundary pixels back to image
    # Get coordinates of all escaped points
    escaped_coords = np.nonzero(escaped_mask)
    rows = escaped_coords[0]
    cols = escaped_coords[1]
    
    # Filter for those satisfying boundary condition
    boundary_rows = rows[boundary_condition]
    boundary_cols = cols[boundary_condition]
    
    # Set boundary color to Black (0)
    img_data[boundary_rows, boundary_cols] = 0
    
    # 5. Save Image
    print("Saving image...")
    img = Image.fromarray(img_data, mode='L') # 'L' mode is 8-bit grayscale
    img.save('mandelbrot_boundary.png')
    print("Done. Saved as mandelbrot_boundary.png")

if __name__ == "__main__":
    generate_mandelbrot_boundary()
