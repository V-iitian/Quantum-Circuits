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

# sigma that is used in multiple functions below 
sigma = [np.array([[0, 1], [1, 0]]), np.array(
        [[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
# Identity Matrix 
I = np.array([[1,0],[0,1]])
             
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
    def __init__(self,alpha,n,theta):
        self.alpha =  np.real_if_close(alpha)
        self.n = np.real_if_close(n)
        self.theta = np.real_if_close(theta)
    def display(self):
        print("alpha(phase Angle) is {},\n the unit vector is nx={}  ny={}  nz={},\n the angle of rotation is{}".format(self.alpha,self.n[0],self.n[1],self.n[2],self.theta))


def to_bloch(g: np.ndarray) -> Bloch:
    """Recover the Bloch form (alpha, n, theta) of a 2x2 unitary `g`."""
    x = np.linalg.det(g)
    alpha = np.arctan2(x.imag, x.real)*(1/2)
    g_thilda = np.exp(-1j * alpha)*g
    theta = 2*np.arccos(np.trace(g_thilda)/2)
    n = []
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


# ---------------------------------------------------------------------------
# Single-qubit rotation helpers (see cpp/src/Unitary2_Bloch.h).
#
# These are the inverse/companion operations to to_bloch and are reused by the
# multi-qubit decomposition pipeline in decompose.py.
# ---------------------------------------------------------------------------


def from_axis_angle(b: Bloch) -> np.ndarray:
    phase = b.alpha
    axis = b.n
    theta = b.theta
    I = np.array([[1,0],[0,1]])
    G_without_phase = (np.cos(theta/2)*I) -1j*np.sin(theta/2)*(axis[0]*sigma[0]+axis[1]*sigma[1]+axis[2]*sigma[2])
    G = np.exp(1j*phase)*G_without_phase
    G = np.round(G,decimals=10)
    return G 


def Rz(theta: float) -> np.ndarray:
    """Rotation about the z axis (no global phase):

    Rz(theta) = diag(e^{-i theta/2}, e^{i theta/2}).
    """
    # TODO: implement (hint: from_axis_angle about axis [0, 0, 1]).
    n = np.array([0,0,1])
    Rz_theta = (np.cos(theta/2)*I) -1j*np.sin(theta/2)*(n[0]*sigma[0]+n[1]*sigma[1]+n[2]*sigma[2])
    Rz_theta = np.round(Rz_theta,decimals=10)
    return Rz_theta



def Ry(theta: float) -> np.ndarray:
    """Rotation about the y axis (no global phase):

    Ry(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]].
    """
    # TODO: implement (hint: from_axis_angle about axis [0, 1, 0]).
    n = np.array([0,1,0])
    Ry_theta = (np.cos(theta/2)*I) -1j*np.sin(theta/2)*(n[0]*sigma[0]+n[1]*sigma[1]+n[2]*sigma[2])
    Ry_theta = np.round(Ry_theta,decimals=10)
    return Ry_theta


def euler_angles_zyz(u: np.ndarray) -> tuple[float, float, float, float]:
    x = np.linalg.det(u)
    alpha = np.arctan2(x.imag, x.real)*(1/2)
    S = np.exp(-1j*alpha)*u
    gamma = 2*np.arccos(np.abs(S[0,0]))
    if np.isclose(np.sin(gamma/2), 0.0):
        Beta = 0.0
        Delta = np.real(-1j * 2 * np.log(S[1,1]))

    elif np.isclose(np.cos(gamma/2), 0.0):
        Beta = 0.0
        Delta = np.real(1j * 2 * np.log(S[1,0]))

    else:
        Beta = np.real(-1j * np.log( S[1,0] * S[1,1] / (np.sin(gamma/2) * np.cos(gamma/2)) ))
        Delta = np.real(-1j * np.log( -1 * (S[0,1] * S[1,1]) / (np.sin(gamma/2) * np.cos(gamma/2)) ))
    alpha = float(np.round(np.real(alpha), decimals=10))
    Beta = float(np.round(Beta, decimals=10))
    gamma = float(np.round(np.real(gamma), decimals=10))
    Delta = float(np.round(Delta, decimals=10))
    angles = tuple([alpha,Beta,gamma,Delta])
    return angles


def unitary2_sqrt(u: np.ndarray) -> np.ndarray:
    b = to_bloch(u)
    alpha_half = b.alpha/2
    theta_half = b.theta/2
    n = b.n 
    b_sqrt = Bloch(alpha_half,n,theta_half)
    b_sqrt_matrix = from_axis_angle(b_sqrt)
    return b_sqrt_matrix

# ---------------------------------------------------------------------------
# H/T word machinery for approximating a 2x2 unitary in {H, T} (see cpp/src/HT.h).
#
# M1, M2 are short H/T words that realize rotations by THETA_M = 2*LAMBDA_PI about
# the axes a1, a2. A word is a flat string of 'H'/'T' characters, read left-to-right
# as a matrix product (leftmost char = leftmost/outermost factor).
# ---------------------------------------------------------------------------

# alternating (T-power, H-power, ...) exponents, starting with T
M1_WORD = [7, 1, 1, 1]
M2_WORD = [2, 1, 1, 1, 6, 1, 7, 1, 5, 1, 1, 1, 2, 1, 1, 1, 2, 1, 7, 1, 6]


def expand_word(word: list[int]) -> str:
    expanded = ""
    for count,words in enumerate(word):
        if count % 2 == 0 :
            while words!=0 : 
                expanded += 'T'
                words-=1
        else :
            while words!=0:
                expanded+='H'
                words-=1
    return expanded

# flat H/T strings for the two building-block words (computed once expand_word works)
M1_STR = expand_word(M1_WORD)
M2_STR = expand_word(M2_WORD)

# T- Matrix 
Z = np.array([[1,0],[0,-1]])
T = unitary2_sqrt(unitary2_sqrt(Z))
def gates_to_unitary(gates: str) -> np.ndarray:
    """The 2x2 unitary of a flat H/T gate string (left-to-right product)."""
    M = np.identity(2,dtype='complex128')
    for char in gates:
        if char == 'H':
            M = M @ H
        else: 
            M = M @ T
    
    return M


def invert_gates(gates: str) -> str:
    """Inverse of a flat H/T word: reverse the gate order and invert each gate.
    H^-1 = H; the {H, T} basis has no T-dagger, so T^-1 must be spelled as T^7.
    """
    inverted_str = ""
    i = len(gates)-1
    while i>=0 :
        if gates[i]=='H':
            inverted_str+='H'
        else:
            inverted_str+='TTTTTTT'
        i-=1
    return inverted_str


def power_gates(base: str, k: int) -> str:
    """The k-th power of a flat H/T word: base repeated k times. Negative k uses the
    inverse word (invert_gates).
    """
    power_str = ""
    if k>0 :
        while k>=0 :
            power_str += base
            k-=1
    else:
        k = -1*k
        base_inverted = base[::-1]
        while k>=0 :
            power_str += base_inverted
            k-=1
    
    return power_str



def approximate_in_ht(u: np.ndarray, error: float) -> str:
    """Approximate a 2x2 unitary `u` by a flat H/T word (up to global phase) to the
    angular tolerance `error` (smaller -> longer, more accurate).

    Use decompose_2x2 to get the powers (k, l, m) with u ~= M1^k M2^l M1^m, then
    assemble the word:

        power_gates(M1_STR, k) + power_gates(M2_STR, l) + power_gates(M1_STR, m).
    """
    k,l,m = decompose_2x2(u,error)
    final_word = power_gates(M1_STR,k)+power_gates(M2_STR,l)+power_gates(M1_STR,m)
    return final_word
