from states import HANDLERS

def pedir_estado():
    while True:
        try:
            
            raw = input("Seleccione:  \n" \
                        "1- Función implementable dado un polinomio. \n" \
                        "2- Funciones de aproximación para plantilla normalizada. \n" \
                        "3- Desnormalización para cumplimiento de plantilla genérica. \n" \
                        "4- A definir. \n"
                        "q- Salir. \n").strip().lower()
            if raw in ("q", "quit", "salir"):
                raise SystemExit(0)
            estado = int(raw)
            if estado in HANDLERS:
                return estado
            print("Error: ingresá un número válido entre 1 y 4.\n")
        except ValueError:
            print("Error: ingresá un número entero entre 1 y 4, o 'q' para salir.\n")
        except (KeyboardInterrupt, EOFError):
            print("\nSalida cancelada por el usuario.")
            raise SystemExit(1)

def main():
    while True:
        estado = pedir_estado()
        HANDLERS[estado]()   # Ejecuta el estado
        # Al terminar el estado, el bucle continúa y volvés al menú

if __name__ == "__main__":
    main()
