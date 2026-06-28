import numpy as np
# Use a single complex dtype for numpy everywhere.
DTYPE = np.complex128

INV_SQRT2 = 1.0 / np.sqrt(2.0)
H = INV_SQRT2 * np.array([[1, 1], [1, -1]], dtype=DTYPE)

# LAMBDA_PI is the base rotation angle realized by the H/T building blocks:
# cos(LAMBDA_PI) = cos^2(pi/8) = (1 + 1/sqrt2)/2. Because LAMBDA_PI / (2 pi) is
# irrational, the multiples {k * LAMBDA_PI mod 2 pi} densely fill [0, 2 pi).
LAMBDA_PI = np.arccos((1.0 + INV_SQRT2) / 2.0)
TWO_PI = 2.0 * np.pi


class Bloch:
    """Axis-angle (Bloch) form of a 2x2 unitary G:

        G = e^{i alpha} (cos(theta/2) I - i sin(theta/2) (n . sigma))

    i.e. a global phase e^{i alpha} times a rotation by angle `theta` about the
    Bloch-sphere axis `n`. Here (n . sigma) = n_x X + n_y Y + n_z Z.
    """

    alpha: float  # global phase
    n: np.ndarray  # unit rotation axis, shape (3,): [n_x, n_y, n_z]
    theta: float  # rotation angle

    # for testing of code
    # def __init__(self,alpha,n,theta):
    # self.alpha =  np.real_if_close(alpha)
    # self.n = np.real_if_close(n)
    # self.theta = np.real_if_close(theta)
    # def display(self):
    # print("alpha(phase Angle) is {},\n the unit vector is nx={}  ny={}  nz={},\n the angle of rotation is{}".format(self.alpha,self.n[0],self.n[1],self.n[2],self.theta))


def to_bloch(g: np.ndarray) -> Bloch:
    """Recover the Bloch form (alpha, n, theta) of a 2x2 unitary `g`."""
    x = np.linalg.det(g)
    alpha = np.arctan2(x.imag, x.real)*(1/2)
    g_thilda = np.exp(-1j * alpha)*g
    theta = 2*np.arccos(np.trace(g_thilda)/2)
    n = []
    sigma = [np.array([[0, 1], [1, 0]]), np.array(
        [[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
    for sigmas in sigma:
        matrix = sigmas @ g_thilda
        trace = (np.trace(matrix)*(1j))/2
        nj = trace/np.sin(theta/2)
        n.append(nj)
    n_array = np.array(n)
    ans = Bloch(alpha, n_array, theta)
    return ans
# testing of to_bloch function
# Bloch_ = to_bloch(H)
# Bloch_.display()


# n1, n2 are two orthogonal Bloch-sphere axes (n1 . n2 == 0)
cot = np.cos(np.pi/8.0)/np.sin(np.pi/8)
n1 = np.array([-cot/np.sqrt(1+2*cot*cot), 1 /
              np.sqrt(1+2*cot*cot), cot/np.sqrt(1+2*cot*cot)])
n2 = np.array([1/(np.sqrt(2)*np.sqrt(1+2*cot*cot)), (np.sqrt(2)*cot)/(np.sqrt(1+2*cot*cot)),
               (-1*1)/(np.sqrt(2)*np.sqrt(1+2*cot*cot))])

# frame derived from the axes (given)
# take the dot product of the Bloch axis with these
# the minus sign arises from the double cover issue
a1 = -n1
a2 = -n2
n3 = np.cross(n1, n2)


# this is the part where i made most of the mistakes
def n1n2n1_angles(b: Bloch) -> tuple[float, float, float, float]:
    phi_half = b.theta / 2.0
    phase = b.alpha
    n = np.array(b.n, dtype=float)

    # 2. Define the known vectors and scalars using phi_half!
    n3 = np.cross(n1, n2)
    V = n * np.sin(phi_half)
    D = np.cos(phi_half)

    # 3. Build the 3x3 basis matrix and solve for A, B, C
    basis_matrix = np.column_stack((n1, n2, n3))
    A, B, C = np.linalg.solve(basis_matrix, V)

    # 4. Extract the half-angle combinations
    gamma_plus_alpha = np.arctan2(A, D)
    gamma_minus_alpha = np.arctan2(C, B)

    # 5. Extract beta_half using vector magnitudes
    cos_beta_half = np.linalg.norm([A, D])
    sin_beta_half = np.linalg.norm([B, C])
    beta_half = np.arctan2(sin_beta_half, cos_beta_half)

    # 6. Algebraically isolate alpha_half and gamma_half
    gamma_half = (gamma_plus_alpha + gamma_minus_alpha) / 2.0
    alpha_half = (gamma_plus_alpha - gamma_minus_alpha) / 2.0

    # 7. Scale the half-angles back up to the full hardware angles!
    alpha = alpha_half * 2.0
    beta = beta_half * 2.0
    gamma = gamma_half * 2.0

    # 8. Clean up angles to strictly sit in the [0, 2*pi] circle
    alpha, beta, gamma = np.mod([alpha, beta, gamma], 2 * np.pi)

    return (phase, alpha, beta, gamma)


def approx_angle_with_tolerance(angle: float, tolerance: float) -> int:
    multiple_angle = LAMBDA_PI
    i = 2
    while (min(abs(multiple_angle-angle), TWO_PI-abs(multiple_angle-angle)) > tolerance):
        multiple_angle = i*multiple_angle
        multiple_angle = np.mod(multiple_angle, 2 * np.pi)
        i = i+1

    return i

# checking the function approx_angle_with_tolerance
# approx_angle_with_tolerance(float(np.pi),0.000001)


def decompose_2x2(u: np.ndarray, tolerance: float) -> tuple[int, int, int]:

    u_bloch = to_bloch(u)
    phase, alpha, beta, gamma = n1n2n1_angles(u_bloch)
    print(alpha, beta, gamma)
    k = approx_angle_with_tolerance(alpha, tolerance)
    l = approx_angle_with_tolerance(beta, tolerance)
    m = approx_angle_with_tolerance(gamma, tolerance)

    return (k, l, m)
