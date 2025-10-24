# utils/pz_tools.py
import numpy as np
import sympy as sp
import math
from sympy.abc import s, w
from numpy import polynomial
from numpy.polynomial import Polynomial as P
from numpy import asarray
from fractions import Fraction as F
from scipy import signal


# Opcional: alta precisión con mpmath (si está instalada)
try:
    from mpmath import mp
    _mpmath_available = True
except Exception:
    mp = None
    _mpmath_available = False

try:
    from scipy import signal
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

import matplotlib.pyplot as plt

# -------------------------
#   Zeros & Poles (SymPy)
# -------------------------
def get_zeros_and_poles(F, var=sp.symbols('s'), params=None, use_scipy=False):
    """
    Calcula ceros y polos de una función transferencia F(var).

    F        : expresión SymPy racional en 'var'
    var      : símbolo de variable (por ej., s)
    params   : dict para sustituir parámetros simbólicos por valores numéricos, ej {R: 0.5}
    use_scipy: si True usa scipy.signal.tf2zpk (requiere coeficientes numéricos)

    return: (zeros, poles) como arrays de complex
    """
    # Normalizar a fracción y sustituir parámetros si los hay
    F = sp.together(F)
    num, den = sp.fraction(F)  # Expr, Expr

    # Detectar símbolos “no var”
    other_syms = (num.free_symbols | den.free_symbols) - {var}
    if other_syms:
        if not params:
            raise ValueError(
                f"La expresión contiene símbolos {other_syms}. "
                f"Pasá 'params={{...}}' con valores numéricos (p.ej. {{R: 0.5}})."
            )
        num = sp.N(num.subs(params))
        den = sp.N(den.subs(params))

    # Convertir a Poly en la variable
    num = sp.expand(num)
    den = sp.expand(den)
    try:
        num_poly = sp.Poly(num, var)
    except sp.PolynomialError:
        raise ValueError(f"El numerador no es polinómico en {var}: {num}")
    try:
        den_poly = sp.Poly(den, var)
    except sp.PolynomialError:
        raise ValueError(f"El denominador no es polinómico en {var}: {den}")

    deg_num = num_poly.degree()
    deg_den = den_poly.degree()

    if not use_scipy:
        # Usamos SymPy (nroots) → no requiere SciPy
        zeros = []
        if deg_num > 0:
            zeros = [complex(r) for r in sp.nroots(num_poly)]
        poles = []
        if deg_den > 0:
            poles = [complex(r) for r in sp.nroots(den_poly)]
        return np.array(zeros, dtype=complex), np.array(poles, dtype=complex)

    # SciPy: tf2zpk con coeficientes numéricos
    if not HAVE_SCIPY:
        raise RuntimeError("SciPy no disponible. Usá use_scipy=False o instalá scipy.")

    def coeffs_to_np(poly):
        return np.array([complex(sp.N(c)) for c in poly.all_coeffs()], dtype=complex)

    b = coeffs_to_np(num_poly) if deg_num >= 0 else np.array([0.0], dtype=complex)
    a = coeffs_to_np(den_poly)
    z, p, k = signal.tf2zpk(b, a)
    return np.array(z, dtype=complex), np.array(p, dtype=complex)


# -------------------------
#   PZ map con semicírculo
# -------------------------
def plot_pz_map(zeros, poles, r=1.0, fill=False):
    """
    Grafica mapa de polos y ceros con semicírculo de radio r en el semiplano izquierdo.

    zeros, poles: arrays de números complejos
    r           : radio del semicírculo (default 1.0)
    fill        : si True, rellena el semicírculo con transparencia
    """
    plt.figure(figsize=(8, 6))

    # Puntos
    plt.scatter(np.real(zeros), np.imag(zeros), marker='o', color='b', label='Ceros')
    plt.scatter(np.real(poles), np.imag(poles), marker='x', color='r', label='Polos')

    # Ejes
    plt.axhline(0, color='k', linewidth=0.8)
    plt.axvline(0, color='k', linewidth=0.8)

    # Semicírculo: semiplano izquierdo (theta de +pi/2 a +3pi/2)
    theta = np.linspace(np.pi/2, 3*np.pi/2, 400)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    plt.plot(x, y, 'g--', linewidth=1.5, label=f'Semicírculo r={r}')
    if fill:
        plt.fill_betweenx(y, x, 0, where=(x <= 0), color='g', alpha=0.12)

    # Límites y aspecto
    # Auto-límites cómodos según datos y r
    all_x = np.concatenate([np.real(zeros), np.real(poles), x])
    all_y = np.concatenate([np.imag(zeros), np.imag(poles), y])
    if all_x.size and all_y.size:
        dx = max(1.2 * np.max(np.abs(all_x)), r * 1.2, 1.6)
        dy = max(1.2 * np.max(np.abs(all_y)), r * 1.2, 1.6)
    else:
        dx = dy = max(r * 1.2, 1.6)
    plt.xlim(-dx, dx)
    plt.ylim(-dy, dy)

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    plt.xlabel(r'$\sigma$')
    plt.ylabel(r'$j\omega$')
    plt.grid(True)
    plt.legend(loc='best')
    plt.title('Mapa de Polos y Ceros')
    plt.show()







# --- Bessel (Thomson) ---


# --- helper: retardo de grupo analógico (numérico) ---
def gd_analog_ba(ba, w):
    """
    Retardo de grupo τ_g(w) = -dφ/dω para filtro analógico definido por (b, a) en s.
    Derivada numérica sobre H(jω). Devuelve escalar float.
    """
    b, a = ba

    def Hjw(w_):
        s = 1j * w_
        num = np.polyval(b, s)
        den = np.polyval(a, s)
        return num / den

    # Paso de derivación: relativo si w>0, absoluto si w≈0
    dw = (1e-6 * w) if (w > 0) else 1e-6
    H1 = Hjw(w)
    H2 = Hjw(w + dw)

    # fase local y diferencia desenrollada
    ph = np.unwrap([np.angle(H1), np.angle(H2)])
    dphi = ph[1] - ph[0]
    tau_g = - dphi / dw
    return float(tau_g)

















def reverse_bessel_poly(n: int, var):
    """
    Polinomio inverso de Bessel (Thomson) normalizado:
      Θ̂_n(s) = Θ_n(s) / Θ_n(0), con Θ̂_n(0)=1
    Coeficientes: a_k = (2n - k)! / [ 2^{(n-k)} (n-k)! k! ]
    Devuelve: (Θ̂_n(s) como sympy expr, lista de coeficientes normalizados [a0..an])
              (coeficientes en potencias ascendentes de s)
    Uso: Gs = 1 / Θ̂_n(s)
    """
    if n < 1:
        raise ValueError("n debe ser >= 1")

    # a_k sin normalizar (enteros exactos con big-int de Python)
    coeffs = []
    for k in range(n + 1):
        num = math.factorial(2*n - k)
        den = (2**(n - k)) * math.factorial(n - k) * math.factorial(k)
        coeffs.append(sp.Rational(num, den))

    # normalizo para que Θ̂_n(0) = 1
    a0 = coeffs[0]
    coeffs_hat = [sp.simplify(c / a0) for c in coeffs]

    # Θ̂_n(s) = sum_{k=0..n} a_k s^k  (potencias ASCENDENTES)
    Theta_hat = sum(coeffs_hat[k] * (var**k) for k in range(n + 1))
    return Theta_hat, coeffs_hat




def _optimum_poly(N):
    """
    Devuelve coeficientes enteros del polinomio 'óptimo' L_N(ω) tal que
    |H(jω)|^2 = 1 / (1 + ε^2 * L_N(ω^2)).
    Los coeficientes están en potencias de ω decrecientes (solo potencias pares).
    """
    if N == 0:
        return np.array([0])

    if N % 2:  # N impar
        k = (N - 1) // 2
        a = np.arange(1, 2*(k + 1) + 1, 2)  # 1,3,5,...
        # denominador sqrt(2)*(k+1) sale fuera del cuadrado
    else:      # N par
        k = (N - 2) // 2
        a = np.arange(1, 2*(k + 1) + 1, 2)  # 1,3,5,...
        # denominador sqrt((k+1)*(k+2)) sale fuera del cuadrado
        if k % 2:   # k impar -> anulos los pares
            a[::2] = 0
        else:       # k par   -> anulos los impares
            a[1::2] = 0

    a = [F(int(i)) for i in a]
    domain = [F(-1), F(1)]

    # v(x) = sum a_i P_i(x), luego lo paso a serie de potencias
    v = polynomial.Legendre(a, domain).convert(domain, polynomial.Polynomial)

    # Íntegrando según Papoulis/Fukada (ramas N impar/par)
    if N % 2:
        integrand = v**2 / (2*(k + 1)**2)
    else:
        integrand = P([F(1), F(1)]) * v**2 / ((k + 1) * (k + 2))

    # Integro y evalúo entre x = -1 y x = 2*ω^2 - 1  (sale función de ω^2)
    indef = P(polynomial.polynomial.polyint(integrand.coef), domain)
    defi = indef(P([F(-1), F(0), F(2)])) - indef(F(-1))

    # Devuelvo como enteros, orden decreciente en ω
    return np.array([int(x) for x in defi.coef[::-1]], dtype=int)

def _roots_legendre_L(a):
    """Raíces (polos) del denominador; usa mpmath si está disponible para alta N."""
    N = (len(a) - 1)//2
    if _mpmath_available:
        mp.dps = 150
        p, err = mp.polyroots(list(map(mp.mpf, a)), maxsteps=1000, error=True)
        if err > mp.mpf('1e-32'):
            raise ValueError(f"No se pudo calcular con precisión la orden {N} (error {err})")
        p = asarray(p, dtype=complex)
    else:
        p = np.roots(a)
        if N > 25:
            # Numéricamente inestable con solo dobles >~25
            raise ValueError("Óptimo-L puede fallar numéricamente para N > 25 sin mpmath.")
    return p

def legendre_papoulis_prototype(N):
    """
    Prototipo analógico de orden N: devuelve (z, p, k) con |H(0)| = 1.
    """
    # Magnitud cuadrada: 1 / (1 + L_N(ω^2))
    a = _optimum_poly(N).astype(float)
    a[-1] = 1.0
    # Sustitución s = jω -> −s^2 = ω^2: cambiar signo en potencias 2,6,10,...
    a[-3::-4] = -a[-3::-4]

    z = np.array([], dtype=complex)
    p = _roots_legendre_L(a)
    p = p[p.real < 0]  # Polinomio de Hurwitz (solo semiplano izquierdo)

    # Normalizo para |H(0)| = 1  => k = ∏|p_i|
    k = float(np.prod(np.abs(p)))
    return z, p, k

def _L_at(N, w):
    """Evalúa L_N(ω^2) usando el polinomio resultante (que ya está en ω con solo potencias pares)."""
    coeffs = _optimum_poly(N).astype(float)
    return np.polyval(coeffs, w)

def design_legendre_papoulis_by_specs(Gp_lin, Ga_lin, Wan, n_max=30):
    """
    Busca el orden mínimo N tal que |H(jΩa)| <= Ga_lin, con
    |H(jω)| = 1 / sqrt(1 + ε^2 L_N(ω^2)),  ε^2 = 1/Gp^2 - 1.
    Retorna zeros, poles, label, N
    """
    # ε^2 desde especificación de pasabanda (Gp en magnitud lineal)
    eps2 = max(float(1.0/(Gp_lin**2) - 1.0), 0.0)

    chosen = None
    for N in range(1, min(n_max, 30) + 1):
        LN = _L_at(N, Wan)     # Esto es L_N(Wan^2)
        Hwan = 1.0 / np.sqrt(1.0 + eps2*LN)
        if Hwan <= Ga_lin + 1e-12:  # cumple stopband en Ωa
            chosen = N
            break

    if chosen is None:
        raise ValueError("No se encontró N ≤ n_max que cumpla las especificaciones.")

    z, p, k = legendre_papoulis_prototype(chosen)
    curve_label = f"Legendre-Papoulis (Óptimo-L), n = {chosen}"
    return z, p, curve_label, chosen
# --- FIN LEGENDRE–PAPOULIS ---------------------------------------------------
