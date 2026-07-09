#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Resolución de la ecuación del calor 1D:
    u_t = u_xx + f(x,t)   (opcional)
con condiciones de contorno Dirichlet, Neumann o mixtas.

Esquemas implementados:
    - Explícito (Forward Euler)
    - Implícito (Backward Euler)
    - Crank-Nicolson

Uso:
    python heat_equation_1d.py

El programa guiará al usuario para ingresar los datos del problema.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os
from sympy import symbols, sympify, lambdify
import numpy as np
# ======================================================================
# 1. CLASE PRINCIPAL
# ======================================================================

class HeatEquation1D:
    """
    Resuelve la ecuación del calor 1D con diferencias finitas.

    Parámetros (pueden pasarse en el constructor o establecerse después):
        L (float): longitud del dominio [0, L].
        Nx (int): número de intervalos espaciales (puntos = Nx+1).
        T (float): tiempo final.
        mu (float): ratio mu = dt/dx^2 (si se da, se ignora Nt).
        Nt (int): número de pasos temporales (si no se da mu).
        u0 (callable): función u(x,0) que recibe x y devuelve float.
        bc_type (str): 'dirichlet', 'neumann' o 'mixed'.
        bc_values (tuple): para dirichlet: (u_left, u_right) (pueden ser funciones de t).
                          para neumann: (u_left, u_right) donde cada uno es (tipo, valor)
        source_func (callable): f(x,t) (por defecto 0).
        scheme (str): 'explicit', 'implicit' o 'crank_nicolson'.
        exact_solution (callable): opcional, para calcular errores.
    """

    def __init__(self, L=1.0, Nx=20, T=1.0, mu=None, Nt=None,
                 u0=None, bc_type='dirichlet', bc_values=(0, 0),
                 source_func=None, scheme='explicit', exact_solution=None):
        self.L = L
        self.Nx = Nx
        self.T = T
        self.mu = mu
        self.Nt = Nt
        self.u0 = u0 if u0 is not None else (lambda x: np.sin(np.pi * x / L))
        self.bc_type = bc_type
        self.bc_values = bc_values
        self.source_func = source_func if source_func is not None else (lambda x, t: 0.0)
        self.scheme = scheme
        self.exact_solution = exact_solution

        self.dx = L / Nx
        self.x = np.linspace(0, L, Nx + 1)

        # Determinar dt
        if mu is not None:
            self.dt = mu * self.dx**2
            self.Nt = int(np.ceil(T / self.dt))
        else:
            if Nt is None:
                raise ValueError("Debe proporcionar mu o Nt")
            self.dt = T / Nt
            self.mu = self.dt / self.dx**2

        # Ajustar T para que coincida con el último paso
        self.T = self.Nt * self.dt

        # Inicializar solución
        self.U = None  # matriz (Nt+1) x (Nx+1)
        self.t = np.linspace(0, self.T, self.Nt + 1)

        # Verificar estabilidad (solo para explícito)
        if self.scheme == 'explicit' and self.mu > 0.5:
            print(f"⚠️ ADVERTENCIA: mu = {self.mu:.4f} > 0.5. El esquema explícito será inestable.")
            print("   Considere reducir mu o usar otro esquema.")

    # ------------------------------------------------------------------
    # 2. MÉTODOS PRIVADOS PARA CONSTRUCCIÓN DEL SISTEMA
    # ------------------------------------------------------------------

    def _build_linear_system(self, n):
        """
        Construye la matriz tridiagonal y el vector RHS para el paso implícito.
        n: índice de tiempo actual (se usa para evaluar fuente en t_n y t_{n+1})
        """
        N = self.Nx - 1  # número de incógnitas interiores
        A = np.zeros((N, N))
        b = np.zeros(N)
        dt = self.dt
        dx = self.dx
        mu = self.mu

        # Coeficientes para el esquema elegido
        if self.scheme == 'implicit':
            theta = 1.0
        elif self.scheme == 'crank_nicolson':
            theta = 0.5
        else:
            raise ValueError("Esquema no soportado para sistema implícito")

        # Llenar matriz (tridiagonal)
        for i in range(N):
            xi = self.x[i+1]
            # Coeficientes
            diag = 1 + 2 * theta * mu
            sub = -theta * mu
            sup = -theta * mu
            # Ajustes en bordes (condiciones de contorno)
            if i == 0:
                # Lado izquierdo
                if self.bc_type == 'dirichlet':
                    # Se mueve al RHS
                    pass
                elif self.bc_type == 'neumann':
                    # Subdiagonal: -theta*mu, diag: 1+theta*mu (porque el punto fantasma)
                    diag = 1 + theta * mu
                    sub = 0.0
                    # El término de Neumann se incluye en b
            if i == N-1:
                # Lado derecho
                if self.bc_type == 'dirichlet':
                    pass
                elif self.bc_type == 'neumann':
                    diag = 1 + theta * mu
                    sup = 0.0

            A[i, i] = diag
            if i > 0:
                A[i, i-1] = sub
            if i < N-1:
                A[i, i+1] = sup

        # Construir RHS
        # Usamos la solución en el paso n (U_n) y la fuente
        U_n = self.U[n, 1:-1]  # valores interiores en t_n
        # Término de fuente en t_n y t_{n+1}
        f_n = np.array([self.source_func(self.x[i+1], self.t[n]) for i in range(N)])
        f_n1 = np.array([self.source_func(self.x[i+1], self.t[n+1]) for i in range(N)])

        b = U_n + dt * ((1 - theta) * f_n + theta * f_n1)

        # Ajustar por condiciones de contorno
        # Lado izquierdo
        if self.bc_type == 'dirichlet':
            # Evaluar condición en t_n y t_{n+1}
            if callable(self.bc_values[0]):
                u_left_n = self.bc_values[0](self.t[n])
                u_left_n1 = self.bc_values[0](self.t[n+1])
            else:
                u_left_n = self.bc_values[0]
                u_left_n1 = self.bc_values[0]
            # Contribución a la primera ecuación
            b[0] += theta * mu * u_left_n1 + (1 - theta) * mu * u_left_n
        elif self.bc_type == 'neumann':
            # Derivada en x=0: (u_1 - u_{-1})/(2dx) = g_left(t)
            # Evaluamos g_left en t_n y t_{n+1}
            g_left_n = self.bc_values[0](self.t[n]) if callable(self.bc_values[0]) else self.bc_values[0]
            g_left_n1 = self.bc_values[0](self.t[n+1]) if callable(self.bc_values[0]) else self.bc_values[0]
            # La ecuación en el punto fantasma se elimina
            # La contribución es: 2*theta*mu*dx*g_left_n1 + 2*(1-theta)*mu*dx*g_left_n
            b[0] += 2 * theta * mu * dx * g_left_n1 + 2 * (1 - theta) * mu * dx * g_left_n

        # Lado derecho
        if self.bc_type == 'dirichlet':
            if callable(self.bc_values[1]):
                u_right_n = self.bc_values[1](self.t[n])
                u_right_n1 = self.bc_values[1](self.t[n+1])
            else:
                u_right_n = self.bc_values[1]
                u_right_n1 = self.bc_values[1]
            b[-1] += theta * mu * u_right_n1 + (1 - theta) * mu * u_right_n
        elif self.bc_type == 'neumann':
            g_right_n = self.bc_values[1](self.t[n]) if callable(self.bc_values[1]) else self.bc_values[1]
            g_right_n1 = self.bc_values[1](self.t[n+1]) if callable(self.bc_values[1]) else self.bc_values[1]
            b[-1] += 2 * theta * mu * dx * g_right_n1 + 2 * (1 - theta) * mu * dx * g_right_n

        return A, b

    # ------------------------------------------------------------------
    # 3. SOLUCIÓN
    # ------------------------------------------------------------------

    def solve(self):
        """Resuelve la ecuación del calor."""
        Nx = self.Nx
        Nt = self.Nt
        dx = self.dx
        dt = self.dt
        mu = self.mu

        # Inicializar matriz de solución
        self.U = np.zeros((Nt + 1, Nx + 1))

        # Condición inicial
        for i in range(Nx + 1):
            self.U[0, i] = self.u0(self.x[i])

        # Bucle en tiempo
        for n in range(Nt):
            t_n = self.t[n]
            # Aplicar condiciones de contorno en el paso actual (para explícito)
            if self.scheme == 'explicit':
                # Esquema explícito: U^{n+1} = U^n + mu * (U_{i-1} - 2U_i + U_{i+1}) + dt*f
                U_n = self.U[n, :]
                U_n1 = np.zeros(Nx + 1)
                # Interior
                for i in range(1, Nx):
                    f_val = self.source_func(self.x[i], t_n)
                    U_n1[i] = (U_n[i] + mu * (U_n[i-1] - 2*U_n[i] + U_n[i+1]) + dt * f_val)
                # Condiciones de contorno
                self._apply_bc_explicit(U_n1, t_n, dt)
                self.U[n+1, :] = U_n1
            else:
                # Esquema implícito: resolver sistema
                A, b = self._build_linear_system(n)
                U_interior = np.linalg.solve(A, b)
                # Construir vector completo con fronteras
                U_n1 = np.zeros(Nx + 1)
                # Frontera izquierda
                if self.bc_type == 'dirichlet':
                    if callable(self.bc_values[0]):
                        U_n1[0] = self.bc_values[0](self.t[n+1])
                    else:
                        U_n1[0] = self.bc_values[0]
                else:
                    # Neumann: usamos la condición para obtener U[-1]? Mejor usar el valor interior
                    # Para simplificar, asumimos que la condición se ha aplicado en el sistema
                    # y el vector U_interior ya incluye el efecto.
                    # Para Neumann, no hay valor fijo en la frontera, usamos el valor interior extrapolado
                    pass
                # Interior
                U_n1[1:-1] = U_interior
                # Frontera derecha
                if self.bc_type == 'dirichlet':
                    if callable(self.bc_values[1]):
                        U_n1[-1] = self.bc_values[1](self.t[n+1])
                    else:
                        U_n1[-1] = self.bc_values[1]
                # Para Neumann, no se asigna valor fijo
                # Si es Neumann, necesitamos usar la condición para calcular el punto fantasma y actualizar el borde
                if self.bc_type == 'neumann':
                    # Aproximación de la derivada en los bordes usando el interior
                    # En x=0: (U1 - U_{-1})/(2dx) = g_left => U_{-1} = U1 - 2dx*g_left
                    # Entonces U0 (en la frontera) no es realmente necesario, pero podemos usar el valor interior
                    # Para mantener coherencia, dejamos U0 = U1 - dx*g_left (extrapolación lineal)
                    g_left = self.bc_values[0](self.t[n+1]) if callable(self.bc_values[0]) else self.bc_values[0]
                    U_n1[0] = U_n1[1] - dx * g_left  # aproximación de primer orden
                    g_right = self.bc_values[1](self.t[n+1]) if callable(self.bc_values[1]) else self.bc_values[1]
                    U_n1[-1] = U_n1[-2] + dx * g_right  # para derivada en x=L: (U_N - U_{N-1})/dx = g_right

                self.U[n+1, :] = U_n1

        return self.U

    def _apply_bc_explicit(self, U_n1, t, dt):
        """Aplica condiciones de contorno para el esquema explícito."""
        if self.bc_type == 'dirichlet':
            if callable(self.bc_values[0]):
                U_n1[0] = self.bc_values[0](t + dt)
            else:
                U_n1[0] = self.bc_values[0]
            if callable(self.bc_values[1]):
                U_n1[-1] = self.bc_values[1](t + dt)
            else:
                U_n1[-1] = self.bc_values[1]
        elif self.bc_type == 'neumann':
            # Implementación simple: usar el valor interior y la condición
            # Para evitar recursión, usamos el valor del paso anterior para extrapolar
            # Mejor: resolver el sistema completo (ya hecho en implícito)
            # Para explícito con Neumann, necesitamos incluir los puntos fantasma.
            # Aquí simplificamos: no implementamos Neumann en explícito.
            raise NotImplementedError("Neumann en explícito no implementado. Use implícito.")

    # ------------------------------------------------------------------
    # 4. MÉTODOS DE CONSULTA
    # ------------------------------------------------------------------

    def get_solution_at_time(self, t_target):
        """Devuelve la solución interpolada en el tiempo t_target."""
        if self.U is None:
            raise ValueError("Primero debe ejecutar solve()")
        # Encontrar índices
        idx = np.argmin(np.abs(self.t - t_target))
        return self.x, self.U[idx, :]

    def compute_error(self):
        """Calcula el error comparando con la solución exacta (si se proporcionó)."""
        if self.exact_solution is None:
            return None
        errors = []
        for n in range(self.Nt + 1):
            t_n = self.t[n]
            u_exact = np.array([self.exact_solution(xi, t_n) for xi in self.x])
            error = np.abs(self.U[n, :] - u_exact)
            errors.append((np.max(error), np.sqrt(np.mean(error**2))))
        return errors

    # ------------------------------------------------------------------
    # 5. GRÁFICAS
    # ------------------------------------------------------------------

    def plot_solution(self, times=None, save=True, filename='heat_solution.png'):
        """Grafica la solución en varios tiempos."""
        if self.U is None:
            self.solve()
        if times is None:
            times = [0, self.T/4, self.T/2, self.T]
        plt.figure(figsize=(10, 6))
        for t in times:
            idx = np.argmin(np.abs(self.t - t))
            plt.plot(self.x, self.U[idx, :], label=f't = {self.t[idx]:.3f}')
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.title(f'Ecuación del calor - {self.scheme.capitalize()} (mu={self.mu:.3f})')
        plt.grid(True)
        plt.legend()
        if save:
            plt.savefig(os.path.join(os.getcwd(), filename), dpi=150)
        plt.show()

    def plot_comparison(self, exact=True, save=True, filename='heat_comparison.png'):
        """Compara la solución numérica con la exacta en el tiempo final."""
        if self.U is None:
            self.solve()
        if self.exact_solution is None:
            print("No se proporcionó solución exacta.")
            return
        t_final = self.T
        idx = self.Nt
        u_num = self.U[idx, :]
        u_exact = np.array([self.exact_solution(xi, t_final) for xi in self.x])

        plt.figure(figsize=(10, 6))
        plt.plot(self.x, u_num, 'b-o', label='Numérica', markersize=4)
        plt.plot(self.x, u_exact, 'r--', label='Exacta', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('u(x, T)')
        plt.title(f'Comparación en t={t_final:.3f} - {self.scheme.capitalize()}')
        plt.grid(True)
        plt.legend()
        if save:
            plt.savefig(filename, dpi=150)
        plt.show()

        # Error
        error = np.abs(u_num - u_exact)
        plt.figure(figsize=(10, 6))
        plt.semilogy(self.x, error, 'g-', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('Error absoluto')
        plt.title(f'Error en t={t_final:.3f}')
        plt.grid(True)
        if save:
            plt.savefig(os.path.join(os.getcwd(), filename), dpi=150)
        plt.show()

    def plot_stability_analysis(self, mu_values=None, save=True):
        """Analiza el efecto de mu en el error final."""
        if mu_values is None:
            mu_values = np.linspace(0.1, 1.0, 10)
        errors = []
        for mu in mu_values:
            # Crear una copia con nuevo mu
            temp = HeatEquation1D(
                L=self.L, Nx=self.Nx, T=self.T, mu=mu,
                u0=self.u0, bc_type=self.bc_type, bc_values=self.bc_values,
                source_func=self.source_func, scheme=self.scheme,
                exact_solution=self.exact_solution
            )
            temp.solve()
            err = temp.compute_error()
            if err is not None:
                errors.append((mu, err[-1][0]))  # error máximo final
        if not errors:
            print("No se puede analizar estabilidad sin solución exacta.")
            return
        mu_vals, err_vals = zip(*errors)
        plt.figure(figsize=(10, 6))
        plt.plot(mu_vals, err_vals, 'bo-')
        plt.xlabel('mu = dt/dx^2')
        plt.ylabel('Error máximo en T')
        plt.title(f'Estabilidad - {self.scheme.capitalize()}')
        plt.grid(True)
        if save:
            plt.savefig(os.path.join(os.getcwd(), 'heat_stability.png'), dpi=150)
        plt.show()

    # ------------------------------------------------------------------
    # 6. ANIMACIÓN
    # ------------------------------------------------------------------

    def animate_solution(self, save=False, filename='heat_animation.gif'):
        """Crea una animación de la evolución de la solución."""
        if self.U is None:
            self.solve()
        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot(self.x, self.U[0, :], 'b-', linewidth=2)
        ax.set_xlim(0, self.L)
        ax.set_ylim(np.min(self.U) - 0.1, np.max(self.U) + 0.1)
        ax.set_xlabel('x')
        ax.set_ylabel('u(x,t)')
        ax.grid(True)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        def update(frame):
            line.set_ydata(self.U[frame, :])
            time_text.set_text(f't = {self.t[frame]:.3f}')
            return line, time_text

        ani = FuncAnimation(fig, update, frames=self.Nt+1, interval=50, blit=True)
        if save:
            ani.save(filename, writer='pillow', fps=20)
        plt.show()

    # ------------------------------------------------------------------
    # 7. TABLA DE RESULTADOS
    # ------------------------------------------------------------------

    def print_table(self, n_points=5):
        """Imprime una tabla de valores en puntos seleccionados."""
        if self.U is None:
            self.solve()
        print("\n" + "="*80)
        print(f"RESULTADOS - {self.scheme.upper()} (mu={self.mu:.4f}, Nx={self.Nx}, Nt={self.Nt})")
        print("="*80)
        print(f"{'x':^10} | ", end="")
        for n in range(0, self.Nt+1, max(1, self.Nt//n_points)):
            print(f"{self.t[n]:^12.4f}", end=" | ")
        print()
        print("-"*80)
        for i in range(0, self.Nx+1, max(1, self.Nx//10)):
            print(f"{self.x[i]:^10.4f} | ", end="")
            for n in range(0, self.Nt+1, max(1, self.Nt//n_points)):
                print(f"{self.U[n, i]:^12.6f}", end=" | ")
            print()
        print("="*80)

    # ------------------------------------------------------------------
    # 8. MÉTODO PARA GUARDAR RESULTADOS (CSV)
    # ------------------------------------------------------------------

    def save_to_csv(self, filename='heat_solution.csv'):
        """Guarda la solución en un archivo CSV."""
        if self.U is None:
            self.solve()
        import csv
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['t'] + list(self.x))
            for n in range(self.Nt+1):
                writer.writerow([self.t[n]] + list(self.U[n, :]))
        print(f"Soluciones guardadas en {filename}")


# ======================================================================
# 9. FUNCIÓN PARA DEFINIR FUNCIONES DESDE TEXTO (interactivo)
# ======================================================================

def make_func(expr, vars_dict=None):
    """Crea una función a partir de una expresión en string."""
    if vars_dict is None:
        vars_dict = {"math": math}
    return lambda *args: eval(expr, vars_dict, {"x": args[0], "t": args[1] if len(args)>1 else 0})


# ======================================================================
# 10. PROGRAMA PRINCIPAL (INTERACTIVO)
# ======================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("   RESOLUCIÓN DE LA ECUACIÓN DEL CALOR 1D")
    print("   u_t = u_xx + f(x,t)")
    print("="*70 + "\n")

    # ---- 1. DOMINIO Y MALLA ----
    L = float(input("Longitud del dominio [0, L] (ej. 1.0): ") or "1.0")
    Nx = int(input("Número de intervalos espaciales Nx (ej. 20): ") or "20")
    T = float(input("Tiempo final T (ej. 1.0): ") or "1.0")

    # ---- 2. CONDICIÓN INICIAL ----
    print("\nIngrese la condición inicial u(x,0) en términos de 'x' (ej. sin(pi*x/L)):")
    u0_str = input("  u0(x) = ") or "sin(pi*x/L)"
    # Usamos math y L en el espacio de nombres
    u0_func = make_func(u0_str, {"math": math, "L": L})

    # ---- 3. TÉRMINO FUENTE ----
    print("\nIngrese el término fuente f(x,t) (ej. 0, o exp(-t)*sin(pi*x)):")
    f_str = input("  f(x,t) = ") or "0"
    f_func = make_func(f_str, {"math": math})

    # ---- 4. CONDICIONES DE CONTORNO ----
    print("\nTipo de condiciones de contorno:")
    print("  1. Dirichlet (u(0,t)=g1, u(L,t)=g2)")
    print("  2. Neumann  (u_x(0,t)=g1, u_x(L,t)=g2)")
    bc_choice = input("Opción (1/2): ") or "1"
    if bc_choice == "1":
        bc_type = "dirichlet"
        print("Ingrese los valores de contorno (pueden ser constantes o funciones de t):")
        left_str = input("  u(0,t) = (ej. 0, o sin(t)): ") or "0"
        right_str = input("  u(L,t) = (ej. 0): ") or "0"
        # Creamos funciones que toman t
        left_func = make_func(left_str, {"math": math})
        right_func = make_func(right_str, {"math": math})
        bc_values = (left_func, right_func)
    else:
        bc_type = "neumann"
        print("Ingrese las condiciones de Neumann (u_x = g):")
        left_str = input("  u_x(0,t) = (ej. 0): ") or "0"
        right_str = input("  u_x(L,t) = (ej. 0): ") or "0"
        left_func = make_func(left_str, {"math": math})
        right_func = make_func(right_str, {"math": math})
        bc_values = (left_func, right_func)

    # ---- 5. ESQUEMA NUMÉRICO ----
    print("\nSeleccione el esquema numérico:")
    print("  1. Explícito (Forward Euler)")
    print("  2. Implícito (Backward Euler)")
    print("  3. Crank-Nicolson")
    scheme_choice = input("Opción (1/2/3): ") or "1"
    schemes = {"1": "explicit", "2": "implicit", "3": "crank_nicolson"}
    scheme = schemes.get(scheme_choice, "explicit")

    # ---- 6. PARÁMETRO mu o Nt ----
    print("\nPuede especificar mu = dt/dx^2 o el número de pasos temporales Nt.")
    use_mu = input("¿Especificar mu? (s/n): ").lower() == "s"
    if use_mu:
        mu = float(input("  mu = (ej. 0.4): ") or "0.4")
        Nt = None
    else:
        Nt = int(input("  Nt (número de pasos temporales, ej. 100): ") or "100")
        mu = None

    # ---- 7. SOLUCIÓN EXACTA (opcional) ----
    print("\n¿Dispone de solución exacta para comparar? (s/n)")
    has_exact = input().lower() == "s"
    exact_func = None
    if has_exact:
        exact_str = input("Ingrese u_exacta(x,t) en términos de 'x' y 't': ")
        exact_func = make_func(exact_str, {"math": math})

    # ---- 8. CREAR INSTANCIA Y RESOLVER ----
    heat = HeatEquation1D(
        L=L, Nx=Nx, T=T, mu=mu, Nt=Nt,
        u0=u0_func, bc_type=bc_type, bc_values=bc_values,
        source_func=f_func, scheme=scheme, exact_solution=exact_func
    )
    print("\nResolviendo...")
    heat.solve()

    # ---- 9. MOSTRAR RESULTADOS ----
    heat.print_table(n_points=4)

    # ---- 10. GRÁFICAS ----
    print("\nGenerando gráficas...")
    heat.plot_solution(save=True, filename=f'heat_{scheme}_solution.png')
    if exact_func is not None:
        heat.plot_comparison(save=True, filename=f'heat_{scheme}_comparison.png')
    heat.plot_stability_analysis(save=True)

    # ---- 11. ANIMACIÓN (opcional) ----
    anim = input("\n¿Desea ver la animación? (s/n): ").lower() == "s"
    if anim:
        heat.animate_solution(save=False)

    # ---- 12. GUARDAR CSV ----
    save_csv = input("\n¿Guardar resultados en CSV? (s/n): ").lower() == "s"
    if save_csv:
        heat.save_to_csv(f'heat_{scheme}_results.csv')

    print("\n✅ Programa finalizado.")
    
