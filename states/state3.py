# states/state3.py
import math
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import sympy as sp
from sympy.abc import s, w

# Helper externo (el mismo que usás en state2)
from utils.pz_tools import get_zeros_and_poles

# ------------------------- Utilidades I/O -------------------------
def _leer_float(prompt):
    while True:
        try:
            raw = input(prompt).strip().lower()
            if raw in ("q", "quit", "salir"):
                raise SystemExit(0)
            return float(raw)
        except ValueError:
            print("Error: ingresá un número válido (o 'q' para salir).")
        except (KeyboardInterrupt, EOFError):
            print("\nOperación cancelada por el usuario.")
            raise SystemExit(1)

def _leer_modo(prompt="¿Ingresás Gp/Ga en 'lineal' o 'db'? [lineal/db]: "):
    while True:
        try:
            raw = input(prompt).strip().lower()
            if raw in ("q", "quit", "salir"):
                raise SystemExit(0)
            if raw in ("lineal", "lin", "l", ""):
                return "lineal"
            if raw in ("db", "d"):
                return "db"
            print("Por favor escribí 'lineal' o 'db'.")
        except (KeyboardInterrupt, EOFError):
            print("\nOperación cancelada por el usuario.")
            raise SystemExit(1)

def _lin2db(x, eps=1e-12):
    x = float(x)
    return 20.0 * math.log10(max(x, eps))

def _db2lin(db):
    return float(10.0 ** (db / 20.0))

def _leer_tipo():
    print("\nTipo de plantilla desnormalizada:")
    print("  1) Pasabajos (LP)")
    print("  2) Pasaaltos (HP)")
    print("  3) Pasabanda (BP)")
    print("  4) Rechazabanda (BS)")
    while True:
        t = int(_leer_float("Elegí [1-4]: "))
        if t in (1, 2, 3, 4):
            return t
        print("Elegí un número entre 1 y 4.")

def _leer_aproximacion():
    opciones = {1:"Butterworth", 2:"Chebyshev I", 3:"Chebyshev II"}
    print("\nElegí una aproximación (prototipo LP):")
    for k,v in opciones.items():
        print(f"  {k}) {v}")
    while True:
        op = int(_leer_float("Opción [1-3] = "))
        if op in opciones:
            return op, opciones[op]
        print("Elegí un número entre 1 y 3.")

# ------------------------- Normalización -------------------------
@dataclass
class NormalizedSpec:
    tipo: str                 # 'LP','HP','BP','BS'
    Gp_lin: float
    Ga_lin: float
    Gp_db: float
    Ga_db: float
    Wan_norm: float           # Ωa normalizada (prototipo LP con Ωp=1)
    kappa: float              # factor de reescalado (se usa solo en BS)
    extra: dict               # parámetros útiles (wp, wa, w0, B, etc.)

def _normalize_to_LP(Gp_lin, Ga_lin, tipo, freqs):
    """
    Normaliza plantilla a prototipo LP con Ωp=1.
    Devuelve (Wan_norm, kappa, extra).
    """
    extra = {}

    if tipo == 'LP':
        wp = freqs['wp']
        wa = freqs['wa']
        if not (wa > wp > 0):
            raise ValueError("LP: se requiere 0 < wp < wa.")
        Wan = wa / wp
        kappa = 1.0
        extra.update({'wp': wp, 'wa': wa})

    elif tipo == 'HP':
        wp = freqs['wp']
        wa = freqs['wa']
        if not (wp > wa > 0):
            raise ValueError("HP: se requiere 0 < wa < wp.")
        Wan = wp / wa
        kappa = 1.0
        extra.update({'wp': wp, 'wa': wa})

    elif tipo == 'BP':
        wp1 = freqs['wp1']
        wp2 = freqs['wp2']
        wa1 = freqs['wa1']
        wa2 = freqs['wa2']
        if not (0 < wa1 < wp1 < wp2 < wa2):
            raise ValueError("BP: requiere 0 < wa1 < wp1 < wp2 < wa2.")
        w0 = math.sqrt(wp1 * wp2)
        B  = wp2 - wp1
        extra.update({'wp1': wp1, 'wp2': wp2, 'wa1': wa1, 'wa2': wa2, 'w0': w0, 'B': B})
        def Omega(omega):
            return abs(omega*omega - w0*w0) / (B * max(omega, 1e-30))
        Wan = min(Omega(wa1), Omega(wa2))
        kappa = 1.0

    elif tipo == 'BS':
        wa1 = freqs['wa1']
        wa2 = freqs['wa2']
        wp1 = freqs['wp1']
        wp2 = freqs['wp2']
        if not (0 < wp1 < wa1 < wa2 < wp2):
            raise ValueError("BS: requiere 0 < wp1 < wa1 < wa2 < wp2.")
        w0 = math.sqrt(wa1 * wa2)  # centro desde SB
        B  = wa2 - wa1
        extra.update({'wp1': wp1, 'wp2': wp2, 'wa1': wa1, 'wa2': wa2, 'w0': w0, 'B': B})
        def Omega(omega):
            return (B * omega) / max(abs(omega*omega - w0*w0), 1e-30)
        Omega_pb1 = Omega(wp1)
        Omega_pb2 = Omega(wp2)
        Omega_pb  = max(Omega_pb1, Omega_pb2)         # borde PB más exigente
        kappa     = 1.0 / max(Omega_pb, 1e-30)        # para que quede Ω'=1 en ese borde
        Wan       = kappa * 1.0                       # en SB los bordes tenían Ω=1
    else:
        raise ValueError("Tipo no reconocido.")

    return Wan, kappa, extra

# ------------------------- Helpers de prototipo -------------------------
def eval_Hjw(w_vec, zeros, poles, K=1.0):
    """ H(jw) = K Π (jw - z_i) / Π (jw - p_i) """
    jw = 1j * np.asarray(w_vec, dtype=float)
    num = np.ones_like(jw, dtype=complex)
    den = np.ones_like(jw, dtype=complex)
    for z in np.atleast_1d(zeros):
        num *= (jw - z)
    for p in np.atleast_1d(poles):
        den *= (jw - p)
    return K * (num / den)

def get_butter_poly(n, show=False):
    """ Polinomio 'simple' como en tu state2 (estructura mínima compatible) """
    x = sp.symbols('x')
    C = [0] * (n + 1)
    C[0] = 1
    for z in range(1, len(C)):
        C[z] = x**z
    return C[n], x

# ------------------------- Diseño del prototipo LP -------------------------
def design_lp_prototype(Gp_lin, Ga_lin, Wan, opcion):
    """
    Construye G(s) (simbólica) del prototipo LP normalizado según la aproximación,
    obtiene ceros/polos con get_zeros_and_poles y devuelve (zeros, useful_poles, label).
    """
    zeros = np.array([], dtype=complex)
    useful_poles = np.array([], dtype=complex)

    if opcion == 1:  # Butterworth
        xi2 = (1.0 / (Gp_lin ** 2)) - 1.0
        inner = (1.0 / xi2) * ((1 / Ga_lin) ** 2 - 1.0)
        n_real = math.log(inner) / (2.0 * math.log(Wan))
        n = max(1, int(math.ceil(n_real)))

        f_poly, x = get_butter_poly(n, show=False)
        Fsym = f_poly.subs(x, sp.I * s)
        Gs   = 1 / (xi2 * (Fsym**2) + 1)

        z_sym, p_sym = get_zeros_and_poles(Gs, var=s)

        zeros = np.array(z_sym, dtype=complex) if hasattr(z_sym, '__len__') and len(z_sym) > 0 else np.array([], dtype=complex)
        poles = np.array(p_sym, dtype=complex) if hasattr(p_sym, '__len__') and len(p_sym) > 0 else np.array([], dtype=complex)
        useful_poles = np.array([p for p in poles if np.real(p) < 0], dtype=complex)

        label = f"|H(jΩ)| Butterworth (n={n})"

    elif opcion == 2:  # Chebyshev I
        eps2 = (1.0 / (Gp_lin ** 2)) - 1.0
        if eps2 <= 0 or Wan <= 1.0:
            return zeros, useful_poles, "Chebyshev I (inválido)"
        num_arg = math.sqrt(max((1.0 / (Ga_lin ** 2)) - 1.0, 0.0))
        ratio   = num_arg / math.sqrt(eps2)
        if ratio <= 1.0:
            n_real = 1.0; n = 1
        else:
            n_real = math.acosh(ratio) / math.acosh(Wan)
            n = max(1, int(math.ceil(n_real)))

        Tn_w = sp.chebyshevt(n, w)
        Fsym = Tn_w.subs(w, s / sp.I)
        Gs   = 1 / (eps2 * (Fsym**2) + 1)
        Gs   = sp.simplify(sp.together(Gs))

        z_sym, p_sym = get_zeros_and_poles(Gs, var=s)

        zeros = np.array(z_sym, dtype=complex) if hasattr(z_sym, '__len__') and len(z_sym) > 0 else np.array([], dtype=complex)
        poles = np.array(p_sym, dtype=complex) if hasattr(p_sym, '__len__') and len(p_sym) > 0 else np.array([], dtype=complex)
        useful_poles = np.array([p for p in poles if np.real(p) < 0], dtype=complex)

        label = f"|H(jΩ)| Chebyshev I (n={n})"

    elif opcion == 3:  # Chebyshev II (inverso)
        denom  = max(1e-15, 1.0 - Ga_lin**2)
        eps_s2 = (Ga_lin**2) / denom
        eps_s  = math.sqrt(eps_s2)
        if Wan <= 1.0:
            return zeros, useful_poles, "Chebyshev II (inválido)"

        rhs = (1.0/eps_s) * math.sqrt(max(1.0/(Gp_lin**2) - 1.0, 0.0))
        if rhs <= 1.0:
            n_real = 1.0; n = 1
        else:
            n_real = math.acosh(rhs) / math.acosh(Wan)
            n = max(1, int(math.ceil(n_real)))

        Tn_is_over_s = sp.chebyshevt(n, sp.I*Wan/s)   # T_n(i*Wan/s)
        Gs_raw = (eps_s2 * (Tn_is_over_s**2)) / (1 + eps_s2 * (Tn_is_over_s**2))
        Gs     = sp.simplify(sp.together(Gs_raw))

        z_sym, p_sym = get_zeros_and_poles(Gs, var=s)

        zeros = np.array(z_sym, dtype=complex) if hasattr(z_sym, '__len__') and len(z_sym) > 0 else np.array([], dtype=complex)
        poles = np.array(p_sym, dtype=complex) if hasattr(p_sym, '__len__') and len(p_sym) > 0 else np.array([], dtype=complex)
        useful_poles = np.array([p for p in poles if np.real(p) < 0], dtype=complex)

        # Parche: si no vinieron ceros, agregamos ceros teóricos en stopband
        if zeros.size == 0:
            k    = np.arange(1, n + 1, dtype=float)
            ang  = (2.0*k - 1.0) * np.pi / (2.0*n)
            c    = np.cos(ang)
            mask = np.abs(c) > 1e-9
            w_z  = Wan / c[mask]
            zeros_theoretical = []
            for wz in w_z:
                zeros_theoretical.append(1j * wz)
                zeros_theoretical.append(-1j * wz)
            zeros = np.array(zeros_theoretical, dtype=complex)

        label = f"|H(jΩ)| Chebyshev II (n={n})"

    else:
        label = "Aprox no implementada"

    return zeros, useful_poles, label


# ------------------------- Mapas de frecuencia Ω(ω) -------------------------
def omega_map(omega, spec: NormalizedSpec):
    """ Devuelve Ω(ω) para evaluar el prototipo LP en eje real. """
    wv = np.asarray(omega, dtype=float); eps = 1e-30
    t  = spec.tipo
    e  = spec.extra
    if t == 'LP':
        return wv / max(e['wp'], eps)
    if t == 'HP':
        return e['wp'] / np.maximum(wv, eps)
    if t == 'BP':
        return np.abs(wv*wv - e['w0']**2) / (e['B'] * np.maximum(wv, eps))
    if t == 'BS':
        return (e['B'] * wv) / np.maximum(np.abs(wv*wv - e['w0']**2), eps)
    raise ValueError("tipo desconocido")

def xrange_for_plot(spec: NormalizedSpec):
    """ Rango de ω real para graficar según el tipo. """
    t = spec.tipo; e = spec.extra
    if t == 'LP':
        return (0.0, e['wa'] * 1.2)
    if t == 'HP':
        return (0.0, max(3*e['wp'], 1.2*e['wa']))
    if t == 'BP':
        return (0.0, e['wa2'] * 1.2)
    if t == 'BS':
        return (0.0, e['wp2'] * 1.2)
    return (0.0, 10.0)

# ------------------------- Estado 3: run() -------------------------
def run():
    print("=== Estado 3: Normalización → Prototipo LP → Desnormalización (plot en ω real) ===")
    print("Podés ingresar Gp/Ga en lineal o dB. Tipos admitidos: LP, HP, BP, BS.\n")

    # Unidades de amplitud
    modo = _leer_modo()

    # Leer Gp, Ga
    if modo == "lineal":
        Gp_lin = _leer_float("Gp (lineal) = ")
        Ga_lin = _leer_float("Ga (lineal) = ")
        if not (0 < Ga_lin < Gp_lin <= 2.0):
            print("Advertencia: esperábamos 0 < Ga < Gp. Verificá tus valores.")
        Gp_db = _lin2db(Gp_lin)
        Ga_db = _lin2db(Ga_lin)
    else:
        Gp_db = _leer_float("Gp (dB) = ")
        Ga_db = _leer_float("Ga (dB) = ")
        if not (Gp_db >= Ga_db):
            print("Advertencia: esperábamos Gp (dB) > Ga (dB). Verificá tus valores.")
        Gp_lin = _db2lin(Gp_db)
        Ga_lin = _db2lin(Ga_db)

    # Tipo y lectura de bordes
    t = _leer_tipo()
    if t == 1:  # LP
        tipo = 'LP'
        wp = _leer_float("ωp (rad/s) = ")
        wa = _leer_float("ωa (rad/s) = ")
        freqs = {'wp': wp, 'wa': wa}
    elif t == 2:  # HP
        tipo = 'HP'
        wp = _leer_float("ωp (rad/s) = ")
        wa = _leer_float("ωa (rad/s) = ")
        freqs = {'wp': wp, 'wa': wa}
    elif t == 3:  # BP
        tipo = 'BP'
        print("Frecuencias de PASABANDA (bordes PB):")
        wp1 = _leer_float("ωp1 (rad/s) = ")
        wp2 = _leer_float("ωp2 (rad/s) = ")
        print("Frecuencias de STOPBAND (bordes SB):")
        wa1 = _leer_float("ωa1 (rad/s) = ")
        wa2 = _leer_float("ωa2 (rad/s) = ")
        freqs = {'wp1': wp1, 'wp2': wp2, 'wa1': wa1, 'wa2': wa2}
    else:       # BS
        tipo = 'BS'
        print("Frecuencias de STOPBAND (bordes SB):")
        wa1 = _leer_float("ωa1 (rad/s) = ")
        wa2 = _leer_float("ωa2 (rad/s) = ")
        print("Frecuencias de PASABANDA (bordes PB):")
        wp1 = _leer_float("ωp1 (rad/s) = ")
        wp2 = _leer_float("ωp2 (rad/s) = ")
        freqs = {'wa1': wa1, 'wa2': wa2, 'wp1': wp1, 'wp2': wp2}

    # Normalizar a LP
    try:
        Wan_norm, kappa, extra = _normalize_to_LP(Gp_lin, Ga_lin, tipo, freqs)
    except ValueError as e:
        print(f"\nError de especificación: {e}")
        return

    spec = NormalizedSpec(
        tipo=tipo,
        Gp_lin=Gp_lin, Ga_lin=Ga_lin,
        Gp_db=_lin2db(Gp_lin), Ga_db=_lin2db(Ga_lin),
        Wan_norm=Wan_norm, kappa=kappa, extra=extra
    )

    # Resumen
    print("\n--- Plantilla normalizada a LP ---")
    print(f"Tipo original          : {spec.tipo}")
    print(f"Gp (lin / dB)          : {spec.Gp_lin:.6g}  /  {spec.Gp_db:.3f} dB")
    print(f"Ga (lin / dB)          : {spec.Ga_lin:.6g}  /  {spec.Ga_db:.3f} dB")
    print(f"Ωp (normalizada)       : 1")
    print(f"Ωa (normalizada)       : {spec.Wan_norm:.6g}")
    print(f"κ (reescalado)         : {spec.kappa:.6g}")
    for k, v in spec.extra.items():
        print(f"{k} = {v:.6g}")

    # Elegir aproximación y diseñar prototipo LP
    opcion, nombre_aprox = _leer_aproximacion()
    zeros, useful_poles, curve_label = design_lp_prototype(spec.Gp_lin, spec.Ga_lin, spec.Wan_norm, opcion)

    if useful_poles.size == 0:
        print("Aviso: no se obtuvieron polos en LHP; no se puede trazar la curva.")
        return

    # ==== Evaluación robusta y ploteo del filtro desnormalizado ====

    # ---- Rango ω y mapeo Ω(ω) con κ ----
    w_min, w_max = xrange_for_plot(spec)
    # Evitemos ω=0 en HP/BP/BS para no disparar singularidades del mapeo
    w_lo = max(w_min, 1e-3 if spec.tipo in ('HP', 'BP', 'BS') else 0.0)
    w_real = np.linspace(w_lo, w_max, 3000)

    Omega  = omega_map(w_real, spec)
    Omega_eff = spec.kappa * Omega

    # ---- Normalización robusta de ganancia (|H|≈Gp_lin en Ω~1) ----
    Omega_ref = np.linspace(0.9, 1.1, 41)            # ventanita alrededor de Ω=1
    H_ref_vec = eval_Hjw(Omega_ref, zeros, useful_poles, K=1.0)
    H_ref_mag = np.abs(H_ref_vec)
    H_ref_mag = H_ref_mag[np.isfinite(H_ref_mag) & (H_ref_mag > 0)]
    if H_ref_mag.size == 0:
        # fallback: un solo punto en Ω=1
        H_ref = eval_Hjw(np.array([1.0]), zeros, useful_poles, K=1.0)[0]
        denom = (np.abs(H_ref) if np.isfinite(H_ref) and np.abs(H_ref) > 0 else 1.0)
        K = spec.Gp_lin / denom
    else:
        K = spec.Gp_lin / np.median(H_ref_mag)

    # ---- Curva |H(jΩ(ω))|
    H_proto = eval_Hjw(Omega_eff, zeros, useful_poles, K=K)
    mag = np.abs(H_proto)
    # recorte numérico para evitar -inf y outliers absurdos
    mag = np.clip(mag, 10**(-120/20), 10**(10/20))
    mag_db = 20.0 * np.log10(mag)

    # ---- Plot en ω real con plantilla desnormalizada ----
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 5))

    # Regiones de plantilla (PB/SB)
    if spec.tipo == 'LP':
        wp, wa = spec.extra['wp'], spec.extra['wa']
        ax.add_patch(Rectangle((0.0, spec.Ga_db), wa, 0.0 - spec.Ga_db,
                               facecolor='red', alpha=0.15, label='SB'))
        ax.add_patch(Rectangle((0.0, min(spec.Gp_db, 0.0)), wp, 0.0 - min(spec.Gp_db, 0.0),
                               facecolor='green', alpha=0.10, label='PB'))
        ax.axvline(wp, color='k', ls='--', lw=0.8)
        ax.axvline(wa, color='k', ls='--', lw=0.8)

    elif spec.tipo == 'HP':
        wp, wa = spec.extra['wp'], spec.extra['wa']
        ax.add_patch(Rectangle((0.0, spec.Ga_db), wa, 0.0 - spec.Ga_db,
                               facecolor='red', alpha=0.15, label='SB'))
        ax.add_patch(Rectangle((wp, min(spec.Gp_db, 0.0)), w_max - wp, 0.0 - min(spec.Gp_db, 0.0),
                               facecolor='green', alpha=0.10, label='PB'))
        ax.axvline(wp, color='k', ls='--', lw=0.8)
        ax.axvline(wa, color='k', ls='--', lw=0.8)

    elif spec.tipo == 'BP':
        wp1, wp2, wa1, wa2 = spec.extra['wp1'], spec.extra['wp2'], spec.extra['wa1'], spec.extra['wa2']
        ax.add_patch(Rectangle((wa1, spec.Ga_db), wa2 - wa1, 0.0 - spec.Ga_db,
                               facecolor='red', alpha=0.15, label='SB'))
        ax.add_patch(Rectangle((wp1, min(spec.Gp_db, 0.0)), wp2 - wp1, 0.0 - min(spec.Gp_db, 0.0),
                               facecolor='green', alpha=0.10, label='PB'))
        for v in (wp1, wp2, wa1, wa2):
            ax.axvline(v, color='k', ls='--', lw=0.8)

    elif spec.tipo == 'BS':
        wp1, wp2, wa1, wa2 = spec.extra['wp1'], spec.extra['wp2'], spec.extra['wa1'], spec.extra['wa2']
        ax.add_patch(Rectangle((wa1, spec.Ga_db), wa2 - wa1, 0.0 - spec.Ga_db,
                               facecolor='red', alpha=0.15, label='SB'))
        ax.add_patch(Rectangle((0.0, min(spec.Gp_db, 0.0)), wp1, 0.0 - min(spec.Gp_db, 0.0),
                               facecolor='green', alpha=0.10, label='PB'))
        ax.add_patch(Rectangle((wp2, min(spec.Gp_db, 0.0)), w_max - wp2, 0.0 - min(spec.Gp_db, 0.0),
                               facecolor='green', alpha=0.10))
        for v in (wp1, wp2, wa1, wa2):
            ax.axvline(v, color='k', ls='--', lw=0.8)

    # Curva del filtro
    ax.plot(w_real, mag_db, lw=1.8, label=f'{curve_label} (desnormalizado)')

    # X logarítmico para HP/BP/BS
    usar_logx = spec.tipo in ('HP', 'BP', 'BS')
    if usar_logx:
        ax.set_xscale('log')
        ax.set_xlim(max(1e-3, w_lo), w_max * 1.1)
    else:
        ax.set_xlim(0.0, w_max * 1.1)

    # Y: autoscale robusto y “sensato”
    finite = np.isfinite(mag_db)
    if np.any(finite):
        q_lo = float(np.quantile(mag_db[finite], 0.02))
        q_hi = float(np.quantile(mag_db[finite], 0.98))
        ymin = min(spec.Ga_db - 20.0, q_lo - 3.0, float(np.min(mag_db[finite])) - 3.0)
        ymax = max(3.0, q_hi + 1.0, float(np.max(mag_db[finite])) + 1.0)
        # Límites razonables
        ymin = max(ymin, spec.Ga_db - 60.0)  # no más de 60 dB por debajo de Ga
        ymax = min(ymax, 6.0)                # techo +6 dB
        if ymax - ymin < 10.0:
            mid = 0.5 * (ymax + ymin)
            ymin = mid - 5.0
            ymax = mid + 5.0
    else:
        ymin, ymax = spec.Ga_db - 40.0, 3.0

    ax.set_ylim(ymin, ymax)

    # Estética
    ax.grid(True, which='both', linestyle=':')
    ax.set_xlabel(r'$\omega$ [rad/s]')
    ax.set_ylabel('Amplitud [dB]')
    ax.set_title(f"Plantilla {spec.tipo} desnormalizada vs. prototipo LP ({nombre_aprox})")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best')

    plt.tight_layout()
    plt.show()
