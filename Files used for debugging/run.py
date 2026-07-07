"""Decompose an arbitrary unitary into the universal basis {H, T, CNOT}.

A unitary is
lowered in stages:

    1. twolevel_decomposition:  Unitary     -> list[TwoLevel]
    2. decompose_twolevel:      TwoLevel    -> SingleQubitGate + ControlledU
    3. decompose_controlledU:   ControlledU -> CU + CNOT
    4. decompose_cu:            CU          -> SingleQubitGate + CNOT
    5. decompose_to_ht:         SingleQubitGate -> H / T words (using rotation.py)

Numpy types: a `Unitary` (N x N) and a 2x2 gate block are
both np.ndarray (complex128); a `ComplexVec` is a 1-D np.ndarray. A `Circuit` is a
Python list of gate objects, each exposing `to_unitary()`; gates are stored in
order of application (the first gate is applied first, i.e. the rightmost matrix
factor).

Every function/method below is a stub for you to implement; See "03 - Completing the Decomposition.pdf" for the recommended order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

import rotation as rt 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def num_qubits(N: int) -> int:
    """Number of qubits n such that N == 2^n (N is the unitary / two-level size)."""
    # TODO: implement.
    i=0
    while 2**i != N:
        i+=1
    return i

# ---------------------------------------------------------------------------
# Gate representations
#
# Each is a sparse description of an operation with a `to_unitary()` returning the
# full N x N matrix. As the decomposition progresses, gates get rewritten into
# simpler ones. The 2x2 block `unitary` is a (2, 2) complex ndarray.
# ---------------------------------------------------------------------------


@dataclass
class TwoLevel:
    """A two-level unitary: acts as the 2x2 `unitary` on the two basis states
    `level0`, `level1` of a size-`size` register, and as identity everywhere else.
    """

    size: int
    level0: int
    level1: int
    unitary: np.ndarray  # (2, 2)

    def to_unitary(self) -> np.ndarray:
        one = self.level0
        two = self.level1
        """Expand to the full `size` x `size` matrix: identity except the 2x2 block
        placed at rows/cols (level0, level1).
        """
        M = np.identity(self.size,dtype="complex128")
        M[one,one]=self.unitary[0,0]
        M[one,two]=self.unitary[0,1]
        M[two,one]=self.unitary[1,0]
        M[two,two]=self.unitary[1,1]
        return M

@dataclass
class SingleQubitGate:
    """A single-qubit gate acting as the 2x2 `unitary` on `qubit` of an n-qubit
    register (N = 2^n), identity on the other qubits.
    """

    n: int
    qubit: int
    unitary: np.ndarray  # (2, 2)

    def to_unitary(self) -> np.ndarray:
        """i was already aware of kronecker product so i used that directly """   
        """ Method 1 """
        # M = np.identity(2**self.n,dtype=int)
        # for i in range(0,self.n):
        #     if i==0:
        #         if i == self.qubit :
        #             temp = self.unitary
        #         else:
        #             temp = np.identity(2,dtype=int)
        #     else:
        #         if i == self.qubit:
        #             temp = np.kron(temp,self.unitary)
        #         else:
        #             temp = np.kron(temp,np.identity(2,dtype=int))
        #     if(i+1==self.n):
        #         return temp
        """Method 2 (Although its same under view of mathematics)"""


        """I wrote a very lengthy solution for this since i didn't have much experience of using bitmask so used ai for this part for a optimal solution"""

        size = 2 ** self.n
        target_qubit = self.qubit
        final_matrix = np.zeros((size, size), dtype=complex)
        gate_2x2 = self.unitary
        
        num_qubits = self.n
        shift = num_qubits - 1 - target_qubit
        mask = 1 << shift
        for i in range(size):
            if (i & mask) == 0:
                
                partner = i | mask
                
                final_matrix[i, i]             = gate_2x2[0, 0]
                final_matrix[i, partner]       = gate_2x2[0, 1]
                final_matrix[partner, i]       = gate_2x2[1, 0]
                final_matrix[partner, partner] = gate_2x2[1, 1]
                
        return final_matrix



@dataclass
class ControlledU:
    """
    A fully-controlled single-qubit gate C^k(U): apply the 2x2 `unitary` to
    `target` iff every other qubit is 1. Controls are always conditioned on 1, so
    their positions need not be stored.
    """

    n: int
    target: int
    unitary: np.ndarray  # (2, 2)

    def to_unitary(self) -> np.ndarray:
        """Identity everywhere except the single controlled block: the pair (all
        ones except the target bit, all ones).
        """
        index1 = 0
        for i in range(0,self.n):
            index1+=2**i
        
        Two_lev = TwoLevel()
        Two_lev.size = 2 ** self.n
        Two_lev.level0 = index1 - (2 **(self.n-self.target-1))
        Two_lev.level1 = index1
        Two_lev.unitary=self.unitary 
        return Two_lev.to_unitary()
        
    

@dataclass
class CU:
    """A singly-controlled single-qubit gate C(U): apply the 2x2 `unitary` to
    `target` iff `control` is 1. The full U(2) (global phase kept) is stored, since
    under a control the global phase becomes a physical relative phase. This is the
    recursion leaf of decompose_controlled.
    """

    n: int
    control: int
    target: int
    unitary: np.ndarray 

    def to_unitary(self) -> np.ndarray:
        """Identity except the control=1 blocks, where `unitary` acts on `target`."""

        # TODO: implement.
        shift = self.n - 1 - self.target
        shift2 = self.n-1-self.control
        mask = 1 << shift
        mask2 = 1 << shift2
        gate_2x2 = self.unitary
        final_matrix = np.identity(2**self.n,dtype="complex128")
        for i in range(0,2**self.n):
            # check if the control bit is zero or one :
            if(i & mask2) != 0 :
                if(i & mask) == 0:
                    partner = i|mask
                    final_matrix[i, i]             = gate_2x2[0, 0]
                    final_matrix[i, partner]       = gate_2x2[0, 1]
                    final_matrix[partner, i]       = gate_2x2[1, 0]
                    final_matrix[partner, partner] = gate_2x2[1, 1]
        return final_matrix

@dataclass
class CNOT:
    """A controlled-NOT: flip `target` iff `control` is 1. Its 2x2 is fixed to
    Pauli-X, so (unlike CU) it stores no unitary.
    """

    n: int
    control: int
    target: int

    def to_unitary(self) -> np.ndarray:
        
        """
        Identity except the control=1 blocks, where X swaps the target's 0/1
        amplitudes.
        """
        Cnot = CU()
        Cnot.n = self.n
        Cnot.control = self.control
        Cnot.target = self.target
        Cnot.unitary = np.array([[0,1],[1,0]])
        return Cnot.to_unitary() 
    
@dataclass
class Swap:
    """

    A multi-controlled NOT (generalized Toffoli): flip `target` iff every other
    qubit equals its entry in `control_vals`. `control_vals` has size n and is
    indexed by qubit; control_vals[target] is unused.

    """

    target: int
    control_vals: list[bool]


# A gate is any of the sparse representations above; a circuit is a list of gates.
Gate = Union[TwoLevel, SingleQubitGate, ControlledU, CU, CNOT]
Circuit = list  # list[Gate]
TwoLevels = list  # list[TwoLevel]


def circuit_to_unitary(circuit: Circuit) -> np.ndarray:
    """Full N x N unitary of a whole circuit. Gates are stored in order of
    application, so the product premultiplies (first gate is the rightmost factor):
    result = g_last @ ... @ g_1. Assumes the circuit is non-empty.
    """
    # TODO: implement.
    final_matrix = np.identity(circuit[0].shape[0],dtype="complex128")
    for i in range(len(circuit)-1,-1,-1):
        final_matrix*=circuit[i].to_unitary()
    
    return final_matrix

def to_circuit(two_levels: TwoLevels) -> Circuit:
    """Wrap a two-level sequence as a circuit, so decompose_unitary /
    twolevel_decomposition output flows straight into a Circuit.
    """
    myCircuit : Circuit = []
    for two_lev in two_levels:
        myCircuit.append(two_lev)
    return myCircuit


def error_up_to_phase(a: np.ndarray, b: np.ndarray) -> float:
    """Elementwise difference between two same-size unitaries, ignoring an overall
    global phase: align b to a by the phase of their Hilbert-Schmidt overlap
    <b, a> = sum conj(b_ij) a_ij, then compare. ~0 means equal up to global phase.
    """
    # TODO: implement.
    b_a = np.sum(np.conj(b)*a)
    b = b * b_a
    
    diff = b - a
    norm = float(np.linalg.norm(diff))
    return norm
# ---------------------------------------------------------------------------
# Stage 1: Unitary -> two-level unitaries (see cpp/src/TwoLevel.h)
# ---------------------------------------------------------------------------


def align(x: complex, y: complex, norm: float) -> np.ndarray:
    """The 2x2 unitary [[conj(x), conj(y)], [-y, x]] / norm. Premultiplying it onto
    a column with entries (x, y) at two levels rotates the amplitude at the second
    level onto the first, leaving the real `norm` there and 0 below.
    """
    # TODO: implement.
    Matrix = np.array([[np.conj(x),np.conj(y)],[-1*y , x]])/norm
    return Matrix

def decompose_vector(vec: np.ndarray) -> TwoLevels:
    """Given the first column of a unitary, return a sequence of two-levels which,
    when premultiplied onto the unitary, make its first column be (1, 0, 0, ...).
    Walk from the bottom to up, using `align` at each pivot to zero out one entry; the
    running pivot holds the accumulated real norm after the first rotation.
    """
    size = len(vec)
    sequence : TwoLevels = []
    for i in range(size-2,-1,-1):
       A = TwoLevel()
       A.size = size 
       A.level0 = i 
       A.level1 = i+1
       norm = vec[i]*np.conj(vec[i])+vec[i+1]*np.conj(vec[i+1])
       A.unitary = align(vec[i],vec[i+1],norm)
       vec = A.to_unitary() @ vec
       if np.isclose(vec[i],1.0):
           vec[i] = 1.0 + 0j
       if np.isclose(vec[i+1],0.0):
           vec[i+1] = 0.0 + 0j
       sequence.append(A)

    return sequence


def expand_twolevels(input: TwoLevels, n: int) -> TwoLevels:
    """Expand each TwoLevel to n dimensions by shifting size, level0, level1 up by
    the offset (n - tl.size). Used to lift a sub-block decomposition back to full n.
    """
    # TODO: implement.
    for tl in input :
        offset = n - tl.size
        tl.size = n
        tl.level0 +=offset
        tl.level1 +=offset
    return input 


def two_levels_to_unitary(two_levels: TwoLevels) -> np.ndarray:
    """Full matrix of a two-level sequence: premultiply each two-level's matrix in
    order (result = tl.to_unitary() @ result), reproducing the application order.
    """
    result = np.identity(two_levels[0].size,dtype="complex128")
    for tl in two_levels:
        result = tl.to_unitary() @ result

    return result


def adjoint_twolevel(tl: TwoLevel) -> TwoLevel:
    """Adjoint of a single two-level: same levels, adjoint (conjugate transpose) of
    the 2x2 block.
    """
    # TODO: implement.
    """since other entries of final unitary is 1 and its updated idesntity matrix. 
    Therefore, just updating the unitary matrix will suffice """
    tl.unitary = np.transpose(np.conj(tl.unitary))

    return tl


def adjoint_twolevels(two_levels: TwoLevels) -> TwoLevels:
    """Adjoint of a sequence: reverse the order and take the adjoint of each, since
    (A_k ... A_1)^dagger = A_1^dagger ... A_k^dagger.
    """
    new_sequence : TwoLevels = []
    for i in range(len(two_levels)-1,-1,-1):
        two_levels[i] = adjoint_twolevel(two_levels[i])
        new_sequence.append(two_levels[i])
    
    return new_sequence



def decompose_unitary(u: np.ndarray) -> TwoLevels:
    """Repeat decompose_vector on successive sub-columns to reduce u to identity.
    At step k, columns/rows 0..k-1 are already reduced, so work on the lower-right
    (n-k) block: clear column k below the diagonal. Finally append a phase two-level
    on the last two levels to cancel the residual phase, so the product is identity.
    Returns the sequence S with prod(S) @ u == I (i.e. prod(S) = u^dagger).
    """
    # TODO: implement.
    


def twolevel_decomposition(u: np.ndarray) -> TwoLevels:
    """The two-level decomposition of u itself: decompose_unitary returns the
    sequence S that reduces u to identity (prod(S) = u^dagger), so its adjoint is
    the sequence whose product is u.
    """
    # TODO: implement (hint: adjoint_twolevels(decompose_unitary(u))).
    sequence = decompose_unitary(u)
    new_sequence = adjoint_twolevels(sequence)
    return new_sequence


# ---------------------------------------------------------------------------
# ABC decomposition of a single-qubit gate (see cpp/src/ABC.h)
# ---------------------------------------------------------------------------


@dataclass
class ABC:
    """Nielsen & Chuang Corollary 4.2: every single-qubit U factors as
    U = e^{i alpha} A X B X C with A B C = I (X is Pauli-X). Building block for a
    single-controlled C(U).
    """

    alpha: float  # global phase
    A: np.ndarray  # (2, 2)
    B: np.ndarray  # (2, 2)
    C: np.ndarray  # (2, 2)


def abc_decompose(u: np.ndarray) -> ABC:
    """Build the ABC decomposition of u (Corollary 4.2). Take the ZYZ Euler angles
    (alpha, beta, gamma, delta) of u, then set
        A = Rz(beta) Ry(gamma/2)
        B = Ry(-gamma/2) Rz(-(delta+beta)/2)
        C = Rz((delta-beta)/2)
    Using X Ry(t) X = Ry(-t) and X Rz(t) X = Rz(-t), these satisfy A B C = I and
    e^{i alpha} A X B X C = u.
    """
    alpha , beta ,gamma , delta = rt.euler_angles_zyz(u)

    A = rt.Rz(beta)*rt.Ry(gamma/2)
    B = rt.Ry(-1*gamma/2)*rt.Rz(-(delta+beta)/2)
    C = rt.Rz((delta-beta)/2)
    ABC_rep = ABC()
    ABC_rep.alpha = alpha
    ABC_rep.A = A
    ABC_rep.B = B
    ABC_rep.C = C
    return ABC_rep



def abc_reconstruct(d: ABC) -> np.ndarray:
    """Reassemble e^{i alpha} A X B X C from an ABC (inverse of abc_decompose)."""
    X = np.array([[0,1],[1,0]])
    u = np.exp(1j*d.alpha)*(d.A @ X @ d.B @ X @ d.C)

    return u

# ---------------------------------------------------------------------------
# Gray code and controlled circuits (see cpp/src/Swap.h, cpp/src/Circuit.h)
# ---------------------------------------------------------------------------


def gray_code(tl: TwoLevel) -> list[Swap]:
    """Gray code connecting level0 and level1 of a two-level, as the sequence of
    single-qubit flips walking from level0 to level1 (one Swap per differing bit).
    At each step the Swap records which qubit flips and the current code's values on
    the other qubits (the control pattern).
    """
    # TODO: implement.
    level0 = tl.level0
    level1 = tl.level1
    n = num_qubits(tl.size)
    list1 = np.zeros(2**n,dtype=bool)
    list2 = np.zeros(2**n,dtype=bool)
    list3 = []
    i = 0
    while level0!=0 :
        if level0 % 2 != 0:
            list1[i] = True
        level0 = level0//2
        i+=1
    i = 0
    while level1!=0 :
        if level1 % 2 != 0:
            list2[i] = True
        level1 = level1//2
        i+=1
    
    list1 = list1[::-1]
    list2 = list2[::-1]
    for i in range(0,len(list1)):
        if list1[i] == list2[i]:
            pass
        else:
            list1[i]=list2[i]
            Swap1 = Swap()
            Swap1.target = i 
            Swap.control_vals = list1
            list3.append(Swap1)
    return list3

def decompose_swap(swap: Swap) -> Circuit:
    """Decompose a Swap (multi-controlled NOT) into a Circuit: a controlled-X with
    the swap's arbitrary control values.
    """
    # TODO: implement (hint: controlled_circuit with Pauli-X).
    

def controlled_circuit(
    n: int, target: int, control_vals: list[bool], unitary: np.ndarray
) -> Circuit:
    """Circuit applying the 2x2 `unitary` to `target` iff every non-target qubit
    equals control_vals[q]. Realized as a fully-controlled-U core (ControlledU,
    controls = all 1) sandwiched by X gates on the qubits conditioned on 0, so those
    become 1-controls. The sandwich is symmetric (X is its own inverse).
    """
    # TODO: implement.
    raise NotImplementedError("controlled_circuit is not implemented yet")


# ---------------------------------------------------------------------------
# Stage 2-5: the decomposition pipeline (see cpp/src/Circuit.h)
# ---------------------------------------------------------------------------


def decompose_twolevel(tl: TwoLevel) -> Circuit:
    """Lower a TwoLevel to a Circuit (Nielsen-Chuang 4.5.2): walk a gray code so
    level0 becomes adjacent to level1, apply the controlled-U on that last
    transition, then undo the walk. Orient the 2x2 so a00 (level0's corner) sits on
    the target value the second-to-last code has.
    """
    # TODO: implement using gray_code, decompose_swap, controlled_circuit.
    


def decompose_controlled(
    n: int, controls: list[int], target: int, u: np.ndarray
) -> Circuit:
    """Decompose C^k(U) (k = len(controls)) into singly-controlled gates C(U) and
    CNOTs (Nielsen-Chuang fig 4.8). Base cases: no control -> a plain SingleQubitGate;
    one control -> a CNOT if U == X else a CU. Otherwise, with V = sqrt(U):
        a. C(V) on target
        b. C^{k-1}(X) onto the pivot control
        c. C(V dagger) on target
        d. repeat b
        e. C^{k-1}(V) on target
    Phases are kept throughout.
    """
    # TODO: implement (recursive; use rotation.unitary2_sqrt for V).
    raise NotImplementedError("decompose_controlled is not implemented yet")


def decompose_controlledU(g: ControlledU) -> Circuit:
    """Lower a ControlledU (controlled on all other qubits) into CNOTs + C(U): build
    the list of all non-target qubits as controls and call decompose_controlled.
    """
    # TODO: implement.
    raise NotImplementedError("decompose_controlledU is not implemented yet")


def decompose_cu(g: CU) -> Circuit:
    """Lower a singly-controlled C(U) into single-qubit gates + 2 CNOTs
    (Nielsen-Chuang Corollary 4.2 / fig 4.6). With U = e^{i alpha} A X B X C and
    A B C = I, emit: C, CNOT, B, CNOT, A on the target, plus a diag(1, e^{i alpha})
    phase on the control line. control=0: CNOTs vanish, target sees A B C = I;
    control=1: CNOTs act as X, target sees A X B X C = U with phase e^{i alpha}.
    """
    # TODO: implement using abc_decompose.
    raise NotImplementedError("decompose_cu is not implemented yet")


def decompose_to_basis(u: np.ndarray) -> Circuit:
    """Fully lower a Unitary to a Circuit of only SingleQubitGate and CNOT, running
    the four stages in sequence:
        1. twolevel_decomposition: Unitary     -> TwoLevels
        2. decompose_twolevel:     TwoLevel    -> SingleQubitGate + ControlledU
        3. decompose_controlledU:  ControlledU -> CU + CNOT
        4. decompose_cu:           CU          -> SingleQubitGate + CNOT
    Each stage rewrites only its own gate type and passes the rest through unchanged.
    """
    # TODO: implement (run each rewrite pass over the circuit).
    raise NotImplementedError("decompose_to_basis is not implemented yet")


def ht_gates(n: int, qubit: int, word: str) -> Circuit:
    """Expand a flat H/T word into a Circuit of SingleQubitGate H/T gates on `qubit`.
    The word (leftmost char = leftmost matrix factor) is pushed in reverse so the
    circuit's application order (first gate first = rightmost factor) reproduces
    rotation.gates_to_unitary(word).
    """
    # TODO: implement.
    raise NotImplementedError("ht_gates is not implemented yet")


def decompose_to_ht(u: np.ndarray, error: float) -> Circuit:
    """Fully lower a Unitary to a Circuit of only H, T, and CNOT gates (the discrete
    fault-tolerant basis): run decompose_to_basis, then replace each arbitrary
    SingleQubitGate with its {H, T} word from rotation.approximate_in_ht (CNOTs pass
    through). `error` is the per-gate angular tolerance (smaller -> longer, more
    accurate). Each word matches its gate up to a global phase; those per-gate phases
    factor out into one overall global phase, so the result reconstructs u up to
    global phase (compare with error_up_to_phase).
    """
    # TODO: implement using decompose_to_basis, ht_gates, and rotation.approximate_in_ht.
    raise NotImplementedError("decompose_to_ht is not implemented yet")
