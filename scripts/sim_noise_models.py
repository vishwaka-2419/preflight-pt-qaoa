"""Simulation study of the edge-transfer estimator under noise models that are not Pauli channels.

For every source circuit of the reference job, the executed Clifford body C and the executed sign-balanced product
inputs are simulated exactly (density matrix, no shot noise) under a noise model applied to the gates of the body and
to the readout. The fitted slope t_e of every readout-pair observable, obtained from the simulated correlators of the
executed inputs by the registered estimator (ordinary least squares with intercept), is compared with the exact
transfer coefficient c_e = Tr[P_e Lambda^dagger(O_e)] / 2^n, where Lambda is the noisy body and O_e the measured
observable including readout errors. Preparations are ideal, so that the comparison isolates the estimator.

Noise models (per gate, applied after SX and X gates on the gate's qubit and after CZ on both qubits):
  pauli   depolarizing p1 = 1e-3 (one-qubit), p2 = 1e-2 (two-qubit); symmetric readout flip 2%
  damp    amplitude damping gamma1 = 2e-3, gamma2 = 4e-3; symmetric readout flip 2%
  coh     coherent over-rotation RX(0.05) after SX, ZZ phase exp(-i 0.03 ZZ) after CZ; ideal readout
  asym    as pauli, with asymmetric readout P(1|0) = 0.01, P(0|1) = 0.05
  hw      p1 = 3e-4, p2 = 2.5e-3, gamma1 = 5e-4, gamma2 = 1e-3, RX(0.01) after SX, exp(-i 0.01 ZZ) after CZ,
          readout P(1|0) = 0.01, P(0|1) = 0.03  (magnitudes comparable to the reported hardware errors)

usage: python3 -I sim_noise_models.py <results_v3 dir> <probe_pairing_check_record.json> <out dir> <model> [max_n]
"""
import json, math, os, sys, time
import numpy as np, pandas as pd
from qiskit import qpy, QuantumCircuit
from qiskit.quantum_info import Clifford, Pauli

R, REC, OUT, MODEL = sys.argv[1:5]
MAXN = int(sys.argv[5]) if len(sys.argv) > 5 else 10
os.makedirs(OUT, exist_ok=True)
rec = json.load(open(REC))
DT = np.complex64
MODELS = dict(
    pauli=dict(p1=1e-3, p2=1e-2, g1=0, g2=0, sx=0, czp=0, p01=0.02, p10=0.02),
    damp=dict(p1=0, p2=0, g1=2e-3, g2=4e-3, sx=0, czp=0, p01=0.02, p10=0.02),
    coh=dict(p1=0, p2=0, g1=0, g2=0, sx=0.05, czp=0.03, p01=0, p10=0),
    asym=dict(p1=1e-3, p2=1e-2, g1=0, g2=0, sx=0, czp=0, p01=0.01, p10=0.05),
    hw=dict(p1=3e-4, p2=2.5e-3, g1=5e-4, g2=1e-3, sx=0.01, czp=0.01, p01=0.01, p10=0.03),
)
M = MODELS[MODEL]
SX = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], DT) / 2
X = np.array([[0, 1], [1, 0]], DT)
def RZ(th): return np.array([[np.exp(-1j * th / 2), 0], [0, np.exp(1j * th / 2)]], DT)
def RX(th): return np.array([[np.cos(th / 2), -1j * np.sin(th / 2)], [-1j * np.sin(th / 2), np.cos(th / 2)]], DT)
PAULI = {"I": np.eye(2, dtype=DT), "X": X, "Y": np.array([[0, -1j], [1j, 0]], DT), "Z": np.array([[1, 0], [0, -1]], DT)}

class DM:
    """Density matrix on n qubits, big-endian (qubit 0 = most significant bit)."""
    def __init__(self, n, rho):
        self.n, self.N, self.rho = n, 2 ** n, rho.astype(DT)
    def u1(self, U, q):
        A = 2 ** q
        r = self.rho.reshape(A, 2, -1); r = np.matmul(U[None, :, :], r); rho = r.reshape(self.N, self.N)
        r = rho.reshape(self.N * A, 2, -1); r = np.matmul(U.conj()[None, :, :], r); self.rho = r.reshape(self.N, self.N)
    def kraus1(self, Ks, q):
        acc = None
        for K in Ks:
            d = DM(self.n, self.rho); d.u1(K, q); acc = d.rho if acc is None else acc + d.rho
        self.rho = acc
    def damp(self, q, g):
        """Amplitude damping with probability g on qubit q (Kraus K0 = diag(1, sqrt(1-g)), K1 = sqrt(g)|0><1|)."""
        A = 2 ** q; B = self.N // (2 * A); r = self.rho.reshape(A, 2, B, A, 2, B); out = np.empty_like(r); sq = math.sqrt(1 - g)
        out[:, 0, :, :, 0, :] = r[:, 0, :, :, 0, :] + g * r[:, 1, :, :, 1, :]
        out[:, 0, :, :, 1, :] = sq * r[:, 0, :, :, 1, :]
        out[:, 1, :, :, 0, :] = sq * r[:, 1, :, :, 0, :]
        out[:, 1, :, :, 1, :] = (1 - g) * r[:, 1, :, :, 1, :]
        self.rho = out.reshape(self.N, self.N)
    def diag_phase(self, phases):
        self.rho = (phases[:, None] * self.rho) * phases.conj()[None, :]
    def depol(self, qs, p):
        """Depolarizing on the qubits qs: rho -> (1-f) rho + f Tr_qs(rho) (x) I/2^k with f = 4^k p/(4^k-1)."""
        k = len(qs); f = (4 ** k) * p / (4 ** k - 1)
        r = self.rho.reshape([2] * (2 * self.n))
        tr = r
        for j, q in enumerate(sorted(qs, reverse=True)):   # trace highest axes first so indices stay valid
            tr = np.trace(tr, axis1=q, axis2=q + (self.n - j))
        # tr has shape [2]*(2*(n-k)); expand back with identity/2^k on qs
        m = self.n - k; keep = [q for q in range(self.n) if q not in qs]
        full = np.zeros([2] * (2 * self.n), DT)
        tr2 = tr.reshape([2] * (2 * m))
        # insert axes for traced qubits: identity means equal row/col index
        import itertools
        for bits in itertools.product([0, 1], repeat=k):
            idx = [slice(None)] * (2 * self.n)
            for q, b in zip(sorted(qs), bits): idx[q] = b; idx[q + self.n] = b
            full[tuple(idx)] += tr2 / (2 ** k)
        self.rho = (1 - f) * self.rho + f * full.reshape(self.N, self.N)
    def expect_diag(self):
        return np.real(np.diag(self.rho)).astype(np.float64)

def bit(n, q, i): return (i >> (n - 1 - q)) & 1
def z_vec(n, q): return np.array([1 - 2 * bit(n, q, i) for i in range(2 ** n)], np.float64)
_CZ = {}
def cz_phase(n, a, b, extra=0.0):
    if (n, a, b, extra) in _CZ: return _CZ[(n, a, b, extra)]
    ph = np.ones(2 ** n, DT)
    for i in range(2 ** n):
        za, zb = 1 - 2 * bit(n, a, i), 1 - 2 * bit(n, b, i)
        ph[i] = (-1 if (za < 0 and zb < 0) else 1) * np.exp(-1j * extra * za * zb)
    _CZ[(n, a, b, extra)] = ph
    return ph

def small_ops(circ, idx):
    """Gate list of a circuit restricted to the active qubits; returns (prep ops, body ops) split at the first barrier."""
    prep, body, seen = [], [], False
    for inst in circ.data:
        nm = inst.operation.name
        if nm == "barrier": seen = True; continue
        if nm in ("measure", "delay"): continue
        qs = [idx[circ.find_bit(q).index] for q in inst.qubits]
        op = (nm, qs, [float(p) for p in inst.operation.params])
        (body if seen else prep).append(op)
    return prep, body

def prep_state(n, ops):
    psi = np.zeros(2 ** n, DT); psi[0] = 1
    for nm, qs, pr in ops:
        U = {"rz": lambda: RZ(pr[0]), "sx": lambda: SX, "x": lambda: X}[nm]()
        q = qs[0]; A = 2 ** q
        psi = np.matmul(U[None, :, :], psi.reshape(A, 2, -1)).reshape(-1)
    return np.outer(psi, psi.conj())

def run_body(dm, body):
    """Apply the body with noise. Virtual RZ gates are merged into the next physical one-qubit gate on the same qubit
    (they are error-free); RZ gates left over at the end commute with CZ and with Z-basis measurement and are dropped."""
    I2 = np.eye(2, dtype=DT); pending = {}
    for nm, qs, pr in body:
        if nm == "rz":
            pending[qs[0]] = RZ(pr[0]) @ pending.get(qs[0], I2)
        elif nm in ("sx", "x"):
            q = qs[0]; U = (SX if nm == "sx" else X) @ pending.pop(q, I2)
            if nm == "sx" and M["sx"]: U = RX(M["sx"]) @ U
            dm.u1(U, q)
            if M["p1"]: dm.depol([q], M["p1"])
            if M["g1"]: dm.damp(q, M["g1"])
        elif nm == "cz":
            dm.diag_phase(cz_phase(dm.n, qs[0], qs[1], M["czp"]))
            if M["p2"]: dm.depol(list(qs), M["p2"])
            if M["g2"]:
                for q in qs: dm.damp(q, M["g2"])
        else:
            raise ValueError(nm)

def measured_zz(diag, n, a, b):
    """Expectation of the measured Z_a Z_b with readout errors P(1|0) = p01, P(0|1) = p10 on every qubit."""
    s, t = 1 - M["p01"] - M["p10"], M["p10"] - M["p01"]
    za, zb = z_vec(n, a), z_vec(n, b)
    return s * s * float(diag @ (za * zb)) + s * t * float(diag @ za) + t * s * float(diag @ zb) + t * t * float(diag.sum())

rows = []
t0 = time.time()
with_qpy = {}
ONLY = os.environ.get("SIM_ONLY")
for key, r in rec.items():
    be, src = key.split(":"); n = r["n"]
    if n > MAXN or n < int(os.environ.get("SIM_MINN", 0)): continue
    if ONLY and key != ONLY: continue
    KEYS = os.environ.get("SIM_KEYS")
    if KEYS and key not in KEYS.split(","): continue
    if be not in with_qpy:
        m = json.load(open(f"{R}/window4_v2/{be}_window4_v2_manifest.json"))
        with open(f"{R}/window4_v2/{be}_window4_v2.qpy", "rb") as f: with_qpy[be] = (m["entries"], qpy.load(f))
    ents, circs = with_qpy[be]
    active = r["active"]; idx = {q: k for k, q in enumerate(active)}
    epairs = [tuple(sorted((r["readout"][str(u)], r["readout"][str(v)]))) for u, v, _ in r["edges"]]
    wts = [wt for _, _, wt in r["edges"]]
    # executed inputs
    sims = []
    body = None
    for k in r["circuits"]:
        prep, bd = small_ops(circs[ents[k]["circuit_index"]], idx)
        if body is None: body = bd
        dm = DM(n, prep_state(n, prep)); run_body(dm, body); dg = dm.expect_diag()
        sims.append((k, dg))
    # fitted slopes from simulated correlators of the executed inputs
    for e, ((a, b), wt) in enumerate(zip(epairs, wts)):
        S, Mv = [], []
        for k, dg in sims:
            s_ = r["ideal"][f"{k},{a},{b}"]
            if abs(s_) > 0.5: S.append(s_); Mv.append(measured_zz(dg, n, a, b))
        if not (S and min(S) < 0 < max(S)):
            continue
        t_hat, b_hat = np.polyfit(S, Mv, 1)
        # exact transfer coefficient: input P_e / 2^n, P_e = C^dagger Z_a Z_b C on the input of C (Heisenberg) -> use the
        # Schroedinger-picture identity Tr[P_e Lambda^dagger(O)] / 2^n = Tr[O Lambda(P_e / 2^n)]
        # P_e is obtained by propagating Z_a Z_b backwards through the ideal body (Clifford), done here by simulating
        # the body on the dual basis: we instead compute c_e directly as the response to the ideal input P_e/2^n.
        qc = QuantumCircuit(n)
        for nm, qs, pr in body:
            if nm == "rz":
                kq = int(round(pr[0] / (math.pi / 2))) % 4
                for _ in range(kq): qc.s(qs[0])
            elif nm == "sx": qc.sx(qs[0])
            elif nm == "x": qc.x(qs[0])
            elif nm == "cz": qc.cz(*qs)
        lab = ["I"] * n; lab[a] = "Z"; lab[b] = "Z"
        pe = Pauli("".join(reversed(lab))).evolve(Clifford(qc), frame="h")
        lab_in = pe.to_label().lstrip("-+i")[::-1]          # qubit-0-first string
        sign = -1 if pe.to_label().startswith("-") else 1
        mat = np.array([[1]], DT)
        for ch in lab_in: mat = np.kron(mat, PAULI[ch])
        dm = DM(n, sign * mat / (2 ** n)); run_body(dm, body)
        c_e = measured_zz(dm.expect_diag(), n, a, b)
        rows.append(dict(backend=be, source=int(src), n=n, u=r["edges"][e][0], v=r["edges"][e][1], weight=wt, t_hat=float(t_hat), b_hat=float(b_hat), c_e=float(c_e), n_inputs=len(S)))
    print(f"{MODEL} {key} n={n} done, {time.time()-t0:.0f} s", flush=True)
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/sim_{MODEL}.csv", index=False)
d = df.t_hat - df.c_e
print(f"{MODEL}: edges {len(df)}, mean(t_hat - c_e) {d.mean():+.4f}, RMS {np.sqrt((d**2).mean()):.4f}, max |dev| {d.abs().max():.4f}, Spearman(t_hat, c_e) {df.t_hat.corr(df.c_e, method='spearman'):.4f}")
