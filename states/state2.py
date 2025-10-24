# states/state2.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
from utils.pz_tools import get_zeros_and_poles, plot_pz_map, reverse_bessel_poly, design_legendre_papoulis_by_specs, gd_analog_ba
from sympy.abc import xi , w, n, s
from scipy.signal import besselap
from scipy import signal



# --- SymPy para polinomio Butterworth y pretty print ---
import sympy as sp
try:
    from IPython.display import display
    HAVE_DISPLAY = True
except Exception:
    HAVE_DISPLAY = False
sp.init_printing(use_unicode=True)

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
    """Convierte magnitud lineal a dB con piso numérico para evitar -inf."""
    x = np.asarray(x, dtype=float)
    return 20.0 * np.log10(np.clip(x, eps, None))

def _db2lin(db):
    """Convierte dB a magnitud lineal."""
    return float(10.0 ** (db / 20.0))

def _leer_aproximacion():
    """
    Muestra un menú con 6 aproximaciones y devuelve (opcion, nombre).
    """
    opciones = {
        1: "Butterworth",
        2: "Chebyshev I",
        3: "Chebyshev II",
        4: "Elíptica (Cauer)",
        5: "Bessel (Thomson)",
        6: "Legendre (Óptimo-L)"
    }
    print("\nElegí una aproximación:")
    for k, v in opciones.items():
        print(f"  {k}) {v}")
    while True:
        try:
            op = int(_leer_float("Opción [1-6] = "))
            if op in opciones:
                return op, opciones[op]
            print("Elegí un número entre 1 y 6.")
        except (KeyboardInterrupt, EOFError):
            print("\nOperación cancelada por el usuario.")
            raise SystemExit(1)


# --- Menú de aproximaciones ---
opcion, nombre_aprox = _leer_aproximacion()
print(f"\nAproximación seleccionada: {opcion}) {nombre_aprox}")

# Inicializar contenedores (evita UnboundLocalError si no hay asignación en alguna rama)
zeros = np.array([], dtype=complex)
useful_poles = np.array([], dtype=complex)



# ---------------- Tu función de polinomio Butterworth ----------------
def get_butter_poly(n, show=False):
    """
    Devuelve el 'polinomio de butterworth' de orden n (según tu construcción).
    Retorna (polinomio, símbolo)
    """
    x = sp.symbols('x')
    C = [0] * (n + 1)  # lista de tamaño n+1
    C[0] = 1           # orden 0

    for z in range(1, len(C)):
        C[z] = x**z
        if show:
            if HAVE_DISPLAY:
                display(sp.Symbol(f'Orden {z}:'))
                display(C[z])
            else:
                print(f"Orden {z}:")
                sp.pprint(C[z])
    return C[n], x
# ---------------------------------------------------------------------

def eval_Hjw(w, zeros, poles, K=1.0):
    """Evalúa H(jw) = K * Π (jw - z_i) / Π (jw - p_i) para un vector w (NumPy)."""
    jw = 1j * w.astype(float)
    num = np.ones_like(jw, dtype=complex)
    den = np.ones_like(jw, dtype=complex)
    # Ceros (podría ser array vacío)
    for z in np.atleast_1d(zeros):
        num *= (jw - z)
    # Polos (usamos los 'útiles', típicamente los del semiplano izquierdo)
    for p in np.atleast_1d(poles):
        den *= (jw - p)
    H = K * (num / den)
    return H

def run():
    print("=== Estado 2: Parámetros y regiones (Y en dB, piso dinámico) ===")
    print("Condiciones: Gp > Ga y Wan > 1.")
    print("Podés ingresar Gp/Ga en LINEAL o en dB. El eje Y siempre se grafica en dB.\n")

    # --- Modo de entrada para Gp/Ga ---
    modo = _leer_modo()

    # --- Lectura y validación ---
    while True:
        if modo == "lineal":
            Gp_lin = _leer_float("Gp (lineal) = ")
            Ga_lin = _leer_float("Ga (lineal) = ")
            Wan    = _leer_float("Wan = ")
            if not (Gp_lin > Ga_lin):
                print("Condición no cumplida: Gp > Ga (en modo lineal).\n")
                continue
            if not (Wan > 1.0):
                print("Condición no cumplida: Wan debe ser mayor que 1.\n")
                continue
            # Para graficar en dB
            Gp_db = float(_lin2db(Gp_lin))
            Ga_db = float(_lin2db(Ga_lin))
        else:  # modo == "db"
            Gp_db = _leer_float("Gp (dB) = ")
            Ga_db = _leer_float("Ga (dB) = ")
            Wan   = _leer_float("Wan = ")
            if not (Gp_db > Ga_db):
                print("Condición no cumplida: Gp > Ga (en dB).\n")
                continue
            if not (Wan > 1.0):
                print("Condición no cumplida: Wan debe ser mayor que 1.\n")
                continue
            # Para cálculos en lineal
            Gp_lin = _db2lin(Gp_db)
            Ga_lin = _db2lin(Ga_db)
        break

    # --- Cálculo solicitado: xi2 = (1 / Gp^2) - 1 (usando Gp en lineal) ---
    xi2 = (1.0 / (Gp_lin ** 2)) - 1.0
    print(f"\nxi2 = (1 / Gp^2) - 1 = {xi2:.6g}")

    # --- Menú de aproximaciones ---
    opcion, nombre_aprox = _leer_aproximacion()
    print(f"\nAproximación seleccionada: {opcion}) {nombre_aprox}")

    # Si es Butterworth: n = ceil( ln((1/xi2)*(Ga^2 - 1)) / (2*ln(Wan)) )
    if opcion == 1:  # Butterworth
        inner = (1.0 / xi2) * ((1/Ga_lin) ** 2 - 1.0)
        ok = True
        if xi2 <= 0:
            print("No se puede calcular n: xi2 debe ser > 0 (revisá Gp)."); ok = False
        if inner <= 0:
            print("No se puede calcular n: (1/xi2)*(Ga^2 - 1) debe ser > 0 (revisá Ga y Gp)."); ok = False
        if Wan <= 1.0:
            print("No se puede calcular n: Wan debe ser > 1."); ok = False

        if ok:
            n_real = math.log(inner) / (2.0 * math.log(Wan))
            n = int(math.ceil(n_real))
            if n < 1:
                n = 1  # asegurar al menos orden 1
            print(f"Orden mínimo Butterworth: n = {n} (valor continuo: {n_real:.6f})")

            # --- Construir el polinomio Butterworth de orden n y mostrarlo ---
            f_poly, x = get_butter_poly(n, show=False)
            print("\nPolinomio Butterworth (según get_butter_poly) de orden n:")
            if HAVE_DISPLAY:
                display(sp.simplify(f_poly))
            else:
                sp.pprint(sp.simplify(f_poly))

            # Construir G(s) simbólica: sustituimos x -> j*ω con ω≡s (porque s=jω en eje jω)
            Fsym = f_poly.subs(x, sp.I*s)      # F(jω) con ω=s
            Gs   = 1 / (xi2 * (Fsym**2) + 1)   # G(s) simbólica

            zeros, poles = get_zeros_and_poles(Gs, var=s)  # ya que xi2 es numérico
            useful_poles = [p for p in poles if np.real(p) < 0]



    elif opcion == 2:  # Chebyshev I
    # ε^2 desde Gp lineal
        if xi2 <= 0:
            print("No se puede calcular n: xi2 (ε^2) debe ser > 0 (revisá Gp).")
        elif Wan <= 1.0:
            print("No se puede calcular n: Wan debe ser > 1.")
        else:
            eps = math.sqrt(xi2)

            # Orden mínimo Chebyshev I:
            # n = ceil( acosh( sqrt((1/Ga^2) - 1) / ε ) / acosh(Wan) )
            num_arg = math.sqrt(max((1.0 / (Ga_lin**2)) - 1.0, 0.0))
            ratio = num_arg / eps
            if ratio <= 1.0:
                n = 1
                n_real = 1.0
                print("Advertencia: sqrt((1/Ga^2)-1)/ε ≤ 1 → se toma n=1.")
            else:
                try:
                    n_real = math.acosh(ratio) / math.acosh(Wan)
                except ValueError as e:
                    print(f"No se puede calcular n (acosh dominio inválido): {e}")
                    n_real = 1.0
                n = max(1, int(math.ceil(n_real)))

            print(f"Orden mínimo Chebyshev I: n = {n} (valor continuo: {n_real:.6f})")

            # Construcción simbólica de G(s):
            # |H(jω)|^2 = 1 / (1 + ε^2 T_n(ω)^2)  ⇒ G(s) = 1 / (1 + ε^2 T_n(s/i)^2)
            Tn_w = sp.chebyshevt(n, w)           # T_n(w)
            Fsym = Tn_w.subs(w, s / sp.I)        # T_n(s/i)
            Gs   = 1 / (1 + (eps**2) * (Fsym**2))

            # Ceros y polos con tu helper
            z_, p_ = get_zeros_and_poles(Gs, var=s)

            # Convertir a arrays de numpy (pueden venir como listas de SymPy)
            zeros = np.array(z_, dtype=complex) if len(z_) else np.array([], dtype=complex)
            poles = np.array(p_, dtype=complex) if len(p_) else np.array([], dtype=complex)

            # Polos útiles: semiplano izquierdo
            useful_poles = np.array([p for p in poles if np.real(p) < 0], dtype=complex)

            if useful_poles.size == 0:
                print("Aviso: no se obtuvieron polos en LHP con la construcción simbólica de Chebyshev I.")



    elif opcion == 3:  # Chebyshev II (inverse Chebyshev) — prototipo LP
    # Parámetro de rizado en STOPBAND desde Ga (lineal):
    #   Ga^2 = eps_s^2 / (1 + eps_s^2)  =>  eps_s^2 = Ga^2 / (1 - Ga^2)
        denom  = max(1e-15, 1.0 - Ga_lin**2)
        eps_s2 = (Ga_lin**2) / denom
        eps_s  = math.sqrt(eps_s2)

        if Wan <= 1.0:
            print("No se puede calcular n: Wan debe ser > 1.")
        else:
            # Orden mínimo:
            #   cosh(n acosh(Wan)) >= (1/eps_s) * sqrt(1/Gp^2 - 1)
            #   => n = ceil( acosh( (1/eps_s)*sqrt(1/Gp^2-1) ) / acosh(Wan) )
            rhs = (1.0/eps_s) * math.sqrt(max(1.0/(Gp_lin**2) - 1.0, 0.0))
            if rhs <= 1.0:
                n = 1
                n_real = 1.0
                print("Advertencia: (1/ε_s)*sqrt(1/Gp^2-1) ≤ 1 → se toma n=1.")
            else:
                try:
                    n_real = math.acosh(rhs) / math.acosh(Wan)
                except ValueError as e:
                    print(f"No se puede calcular n (acosh dominio inválido): {e}")
                    n_real = 1.0
                n = max(1, int(math.ceil(n_real)))

            print(f"Orden mínimo Chebyshev II: n = {n} (valor continuo: {n_real:.6f})")

            # ---------- Construcción simbólica robusta ----------
            # Usamos directamente T_n(i*Wan/s)
            Tn_is_over_s = sp.chebyshevt(n, sp.I*Wan/s)  # T_n(i*Wan/s)
            Gs_raw = (eps_s2 * (Tn_is_over_s**2)) / (1 + eps_s2 * (Tn_is_over_s**2))

            # Forzar fracción racional en s para que el helper pueda factorizar
            Gs = sp.simplify(sp.together(Gs_raw))

            # Ceros y polos por tu helper
            z_sym, p_sym = get_zeros_and_poles(Gs, var=s)

            zeros = np.array(z_sym, dtype=complex) if len(z_sym) else np.array([], dtype=complex)
            poles = np.array(p_sym, dtype=complex) if len(p_sym) else np.array([], dtype=complex)

            # Polos útiles en LHP
            useful_poles = np.array([p for p in poles if np.real(p) < 0], dtype=complex)

            # --- Parche: si no vinieron ceros, agregamos los ceros teóricos en jω (stopband) ---
            if zeros.size == 0:
                k    = np.arange(1, n + 1, dtype=float)
                ang  = (2.0*k - 1.0) * np.pi / (2.0*n)
                c    = np.cos(ang)
                mask = np.abs(c) > 1e-9  # evita cos≈0 (cero en ∞)
                w_z  = Wan / c[mask]
                zeros_theoretical = []
                for wz in w_z:
                    zeros_theoretical.append(1j * wz)
                    zeros_theoretical.append(-1j * wz)
                zeros = np.array(zeros_theoretical, dtype=complex)

            curve_label = f"|H(jω)| Chebyshev II (n={n})"

            # Debug opcional:
            # print(f"[cheb2] zeros={len(zeros)}, poles={len(useful_poles)}; eps_s2={eps_s2:.3e}")






    elif opcion == 5:  # Bessel (Thomson) — SOLO plantilla pasabajos (SciPy bessel)
        N_MAX = 20
        elegido = None

        zeros = np.array([], dtype=complex)
        useful_poles = np.array([], dtype=complex)
        curve_label = "Bessel (Thomson)"

        for n_try in range(1, N_MAX + 1):
            try:
                # Prototipo Bessel (Thomson) normalización 'delay'
                # Tomo ZPK para evaluar |H(jw)|
                z_sc, p_sc, k_sc = signal.bessel(n_try, 1.0, analog=True, output='zpk', norm='delay')

                z_arr = np.array(z_sc, dtype=complex) if z_sc is not None else np.empty(0, dtype=complex)
                p_arr = np.array(p_sc, dtype=complex)

                # Polos en LHP
                poles_lhp = p_arr[np.real(p_arr) < 0]
                if poles_lhp.size == 0:
                    continue

                # === FIX: Forzar H(0)=1 con K complejo (no lo conviertas a float) ===
                denom0 = np.prod(-poles_lhp, dtype=complex) if poles_lhp.size else (1+0j)
                numer0 = np.prod(-z_arr,     dtype=complex) if z_arr.size     else (1+0j)
                K_dc   = denom0 / numer0     # H(0) = K_dc / ∏(-p_i) = 1

                # Plantilla LP: PB=[0,1], SB=[Wan, 1.5*Wan]
                w_pass = np.linspace(0.0, 1.0, 400)       # incluye w=0 → aten. arranca en 0 dB
                w_stop = np.linspace(Wan, Wan * 1.5, 400)

                H_pass = eval_Hjw(w_pass, z_arr, poles_lhp, K=K_dc)
                H_stop = eval_Hjw(w_stop, z_arr, poles_lhp, K=K_dc)

                if H_pass.size == 0 or H_stop.size == 0:
                    continue

                min_pass = float(np.min(np.abs(H_pass)))
                max_stop = float(np.max(np.abs(H_stop)))

                # Condiciones de plantilla
                if (min_pass >= Gp_lin) and (max_stop <= Ga_lin):
                    elegido = n_try
                    zeros = z_arr
                    useful_poles = poles_lhp
                    curve_label = f"Bessel (Thomson), n = {elegido}"
                    print(f"Orden Bessel elegido por plantilla: n = {elegido}")
                    break

            except Exception as e:
                print(f"[Bessel n={n_try}] aviso: {e}")

        # Fallback si nada cumple
        if elegido is None:
            n_fallback = min(N_MAX, 15)
            print(f"Ningún orden 1..{N_MAX} cumple la plantilla. Se grafica con n={n_fallback}.")
            z_sc, p_sc, k_sc = signal.bessel(n_fallback, 1.0, analog=True, output='zpk', norm='delay')

            z_arr = np.array(z_sc, dtype=complex) if z_sc is not None else np.empty(0, dtype=complex)
            p_arr = np.array(p_sc, dtype=complex)
            poles_lhp = p_arr[np.real(p_arr) < 0]

            # === FIX aplicado también en el fallback ===
            denom0 = np.prod(-poles_lhp, dtype=complex) if poles_lhp.size else (1+0j)
            numer0 = np.prod(-z_arr,     dtype=complex) if z_arr.size     else (1+0j)
            K_dc   = denom0 / numer0

            zeros = z_arr
            useful_poles = poles_lhp
            curve_label = f"Bessel (Thomson), n = {n_fallback} (fallback)"







    # --- Configuración del gráfico (en dB, piso dinámico) ---
    x_max = Wan * 1.5                      # X: 0 .. Wan + 0.5*Wan
    y_top = 0.0                            # 0 dB ≡ magnitud 1
    y_floor = Ga_db - 20.0                 # piso dinámico: 20 dB por debajo de Ga

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 5))

    # Ejes
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(y_floor, y_top)
    ax.set_xlabel(r'$\omega$  [rad/s]')
    ax.set_ylabel('Amplitud [dB]')
    ax.set_title('Regiones requeridas (Estado 2) — Eje Y en dB (piso = Ga−20 dB)')
    ax.grid(True, which='both', linestyle=':', linewidth=0.7)

    # --- Rectángulo 1: x ∈ [0, 1], y ∈ [y_floor, Gp_db] ---
    x0_1, w1 = 0.0, 1.0
    y0_1 = y_floor
    h1 = max(0.0, min(Gp_db, y_top) - y0_1)  # recorta en y_top si Gp_db > 0
    if h1 > 0:
        rect1 = Rectangle((x0_1, y0_1), w1, h1, facecolor='red', edgecolor='red', alpha=0.25, label='Región 1')
        ax.add_patch(rect1)

    # --- Rectángulo 2: x ∈ [Wan, x_max], y ∈ [Ga_db, y_top] ---
    x0_2, w2 = Wan, max(0.0, x_max - Wan)
    y0_2 = max(Ga_db, y_floor)             # no bajar del piso
    h2 = max(0.0, y_top - y0_2)
    if w2 > 0 and h2 > 0:
        rect2 = Rectangle((x0_2, y0_2), w2, h2, facecolor='red', edgecolor='red', alpha=0.25, label='Región 2')
        ax.add_patch(rect2)

    # Líneas guía
    ax.axvline(1.0, color='k', linewidth=0.8, linestyle='--', label=None)
    ax.axvline(Wan, color='k', linewidth=0.8, linestyle='--', label=None)
    ax.axhline(Gp_db, color='r', linewidth=0.8, linestyle='--', label='Gp (dB)')
    ax.axhline(Ga_db, color='r', linewidth=0.8, linestyle='--', label='Ga (dB)')
    ax.axhline(0.0,   color='g', linewidth=0.8, linestyle='--', label='0 dB')
    ax.axhline(y_floor, color='m', linewidth=0.8, linestyle=':', label='Piso Ga−20 dB')

    # Vector de frecuencias para trazar (mismo rango del gráfico)
    w_curve = np.linspace(0.0, x_max, 2000)

    # Elegimos una referencia para normalizar la ganancia: ω_ref = 1 rad/s
    # Objetivo: |H(j*ω_ref)| = Gp_lin  (si preferís DC, cambialo a ω_ref=0)
    omega_ref = 1.0
    target_mag = Gp_lin

    # Calcular K tal que |H(j*ω_ref)| = target_mag
    H_ref = eval_Hjw(np.array([omega_ref]), zeros, np.array(useful_poles), K=1.0)[0]
    K = (target_mag / (np.abs(H_ref) if np.abs(H_ref) > 0 else 1.0))

    # Evaluar y graficar magnitud en dB
    H_jw = eval_Hjw(w_curve, zeros, np.array(useful_poles), K=K)
    mag_db = 20.0 * np.log10(np.clip(np.abs(H_jw), 1e-300, None))  # clip para evitar -inf

    ax.plot(w_curve, mag_db, linewidth=1.8, label='|H(jω)| desde ceros/polos')

    # Reafirmar límites
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(y_floor, y_top)

    # Leyenda (evita duplicados)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best')

    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    # Pausa antes de volver al menú
    try:
        input("\nEnter para volver al menú...")
    except (KeyboardInterrupt, EOFError):
        pass
    plt.close(fig)
    print("Volviendo al menú...\n")
