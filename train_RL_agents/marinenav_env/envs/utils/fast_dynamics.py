# fast_dynamics.py
# Numba-accelerated numerical computation functions for robot dynamics and ocean currents
import numpy as np
import numba as nb


@nb.njit
def compute_motion_step_numba(
    mass_matrix_inv,
    m,
    xDotU, yDotV, yDotR, nDotV, nDotR,
    xU, xUU, yV, yVV, yRV, yVR, yRR,
    nV, nVV, nRV, nVR, nR, nRR,
    u_r, v_r, u, v, r,
    left_thrust, right_thrust,
    left_pos, right_pos,
    length, width,
    dt
):
    """
    Numba-accelerated computation of robot motion dynamics.
    
    All inputs should be float64 for consistency.
    
    Args:
        mass_matrix_inv: Inverse of the mass matrix (3x3)
        m: Robot mass
        xDotU, yDotV, etc.: Hydrodynamic derivative coefficients
        u_r, v_r: Relative velocity components in robot frame
        u, v: Absolute velocity components in robot frame
        r: Angular velocity
        left_thrust, right_thrust: Thruster forces
        left_pos, right_pos: Thruster angles
        length, width: Robot dimensions
        dt: Time step
        
    Returns:
        u_r_new, v_r_new, r_new: Updated velocity components
    """
    
    # Convert scalars to float for consistency
    m = float(m)
    u_r = float(u_r)
    v_r = float(v_r)
    u = float(u)
    v = float(v)
    r = float(r)
    
    # Coriolis matrix (rigid body)
    C_RB = np.array([
        [0.0,  -m * r, 0.0],
        [m*r,   0.0,   0.0],
        [0.0,   0.0,   0.0]
    ], dtype=np.float64)
    
    # Coriolis matrix (added mass)
    C_A = np.array([
        [0.0, 0.0, yDotV * v_r + yDotR * r],
        [0.0, 0.0, -xDotU * u_r],
        [-yDotV * v_r - yDotR * r, xDotU * u_r, 0.0]
    ], dtype=np.float64)
    
    # Linear damping matrix
    D = -np.array([
        [xU,  0.0, 0.0],
        [0.0, yV,  0.0],
        [0.0, nV,  nR]
    ], dtype=np.float64)
    
    # Nonlinear damping matrix
    D_n = -np.array([
        [xUU * abs(u_r), 0.0, 0.0],
        [0.0, yVV * abs(v_r) + yRV * abs(r), yVR * abs(v_r) + yRR * abs(r)],
        [0.0, nVV * abs(v_r) + nRV * abs(r), nVR * abs(v_r) + nRR * abs(r)]
    ], dtype=np.float64)
    
    # Total non-inertial/damping terms
    N = C_A + D + D_n
    
    # Thruster forces and moments
    F_x_left = left_thrust * np.cos(left_pos)
    F_y_left = left_thrust * np.sin(left_pos)
    M_x_left = F_x_left * width / 2.0
    M_y_left = -F_y_left * length / 2.0
    
    F_x_right = right_thrust * np.cos(right_pos)
    F_y_right = right_thrust * np.sin(right_pos)
    M_x_right = -F_x_right * width / 2.0
    M_y_right = -F_y_right * length / 2.0
    
    F_x = F_x_left + F_x_right
    F_y = F_y_left + F_y_right
    M_n = M_x_left + M_y_left + M_x_right + M_y_right
    
    tau_p = np.array([F_x, F_y, M_n], dtype=np.float64)
    
    # Compute acceleration: mass_matrix * acc = -C_RB V - N V_r + tau_p
    V = np.array([u, v, r], dtype=np.float64)
    V_r = np.array([u_r, v_r, r], dtype=np.float64)
    
    # Ensure mass_matrix_inv is float64
    mass_matrix_inv_f64 = mass_matrix_inv.astype(np.float64)
    
    b = -C_RB @ V - N @ V_r + tau_p
    
    # Apply precomputed inverse
    acc = mass_matrix_inv_f64 @ b
    
    # Update relative velocity
    V_r_new = V_r + acc * dt
    
    return V_r_new[0], V_r_new[1], V_r_new[2]


@nb.njit
def compute_velocity_from_cores_numba(
    x, y,
    core_x, core_y, core_gamma, core_clockwise,
    r, two_pi_r_sq
):
    """
    Numba-accelerated computation of ocean velocity from vortex cores.
    
    Args:
        x, y: Query position
        core_x, core_y: Arrays of vortex core positions
        core_gamma: Array of circulation strengths
        core_clockwise: Array of rotation directions (1.0 for clockwise, -1.0 for counter-clockwise)
        r: Vortex core radius
        two_pi_r_sq: Precomputed 2 * pi * r^2
        
    Returns:
        vx, vy: Velocity components at query position
    """
    n_cores = len(core_x)
    
    # Arrays to track which cores have been processed and their radial vectors
    # We'll use a simpler approach: just accumulate velocity from all cores
    # that aren't shadowed by closer cores
    
    # Compute distances to all cores
    distances = np.empty(n_cores)
    for i in range(n_cores):
        dx = core_x[i] - x
        dy = core_y[i] - y
        distances[i] = np.sqrt(dx*dx + dy*dy)
    
    # Sort indices by distance
    sorted_indices = np.argsort(distances)
    
    # Initialize velocity
    vx = 0.0
    vy = 0.0
    
    # Store radial vectors of processed cores
    radial_x = np.empty(n_cores)
    radial_y = np.empty(n_cores)
    num_processed = 0
    
    for idx in sorted_indices:
        i = idx
        
        # Compute radial vector from query to core
        dx = core_x[i] - x
        dy = core_y[i] - y
        dis = distances[i]
        
        # Check if this core is in the outer area of any processed core
        skip = False
        for j in range(num_processed):
            # Dot product: if positive, core i is in outer area of core j
            dot = radial_x[j] * dx + radial_y[j] * dy
            if dot > 0:
                skip = True
                break
        
        if skip:
            continue
        
        # Store radial vector for this core
        radial_x[num_processed] = dx
        radial_y[num_processed] = dy
        num_processed += 1
        
        # Normalize radial vector
        dx_norm = dx / dis
        dy_norm = dy / dis
        
        # Compute tangent vector (perpendicular to radial)
        # For clockwise: rotate radial by -90 degrees
        # For counter-clockwise: rotate radial by +90 degrees
        if core_clockwise[i] > 0:
            # Clockwise: (x, y) -> (y, -x)
            tx = dy_norm
            ty = -dx_norm
        else:
            # Counter-clockwise: (x, y) -> (-y, x)
            tx = -dy_norm
            ty = dx_norm
        
        # Compute speed at this distance
        if dis <= r:
            speed = core_gamma[i] / two_pi_r_sq * dis
        else:
            speed = core_gamma[i] / (2.0 * np.pi * dis)
        
        # Add velocity contribution
        vx += tx * speed
        vy += ty * speed
    
    return vx, vy
