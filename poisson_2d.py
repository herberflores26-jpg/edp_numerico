#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Resolución de la ecuación de Poisson en 2D:
    - (u_xx + u_yy) = f(x,y)   en (0,Lx) x (0,Ly)
con condiciones de contorno Dirichlet, Neumann o mixtas.

Método: diferencias finitas con esquema de 5 puntos.
Resolución del sistema lineal: método directo (numpy.linalg.solve) o iterativo (Gauss-Seidel).

Uso:
    python poisson_2d.py

El programa guiará al usuario para ingresar los datos del problema.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os 
from sympy import symbols, sympify, lambdify
import numpy as np
# ======================================================================
# 1. CLASE PRINCIPAL
# ======================================================================

class Poisson2D:
    """
    Resuelve la ecuación de Poisson en 2D con diferencias finitas.

    Parámetros:
        Lx, Ly (float): dimensiones del dominio [0, Lx] x [0, Ly].
        Nx, Ny (int): número de intervalos en cada dirección.
        f (callable): función fuente f(x,y).
        bc (dict): condiciones de contorno para cada lado:
                   {'left': (tipo, valor), 'right': ..., 'bottom': ..., 'top': ...}
                   tipo: 'dirichlet' o 'neumann'
                   valor: número o función de (x,y) para Dirichlet, o función de (x,y) para Neumann.
        solver (str): 'direct' (numpy.linalg.solve) o 'iterative' (Gauss-Seidel).
        tol (float): tolerancia para el método iterativo.
        max_iter (int): número máximo de iteraciones para el método iterativo.
        exact_solution (callable): opcional, para calcular errores.
    """

    def __init__(self, Lx=1.0, Ly=1.0, Nx=20, Ny=20,
                 f=None, bc=None, solver='direct',
                 tol=1e-8, max_iter=10000, exact_solution=None):
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.f = f if f is not None else (lambda x, y: 0.0)
        self.bc = bc if bc is not None else self._default_bc()
        self.solver = solver
        self.tol = tol
        self.max_iter = max_iter
        self.exact_solution = exact_solution

        self.dx = Lx / Nx
        self.dy = Ly / Ny
        self.x = np.linspace(0, Lx, Nx + 1)
        self.y = np.linspace(0, Ly, Ny + 1)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # Número de incógnitas interiores
        self.Ni = (Nx - 1) * (Ny - 1)
        self.U = None  # matriz (Nx+1) x (Ny+1)
        self.iterations = 0
        self.residual_history = []

    def _default_bc(self):
        """Condiciones de contorno por defecto: Dirichlet homogéneas."""
        return {
            'left': ('dirichlet', 0),
            'right': ('dirichlet', 0),
            'bottom': ('dirichlet', 0),
            'top': ('dirichlet', 0)
        }

    # ------------------------------------------------------------------
    # 2. CONSTRUCCIÓN DEL SISTEMA LINEAL
    # ------------------------------------------------------------------

    def _build_system(self):
        """Construye la matriz y el vector RHS para el sistema lineal A*u = b."""
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.dx, self.dy
        Ni = self.Ni

        # Preparamos matrices dispersas (usamos listas para diagonales)
        # Para simplificar, usamos matriz densa (para Nx,Ny hasta ~100 está bien)
        A = np.zeros((Ni, Ni))
        b = np.zeros(Ni)

        # Función para mapear (i,j) interior a índice global
        def idx(i, j):
            # i = 1..Nx-1, j = 1..Ny-1
            return (i - 1) + (j - 1) * (Nx - 1)

        # Coeficientes del esquema de 5 puntos
        coef_center = -2.0 / dx**2 - 2.0 / dy**2
        coef_x = 1.0 / dx**2
        coef_y = 1.0 / dy**2

        # Recorremos los puntos interiores
        for i in range(1, Nx):
            for j in range(1, Ny):
                k = idx(i, j)
                xi = self.x[i]
                yj = self.y[j]

                # Término fuente
                b[k] = -self.f(xi, yj)  # porque la ecuación es -u_xx - u_yy = f

                # Diagonal principal
                A[k, k] = coef_center

                # Vecinos (si existen)
                # Izquierda (i-1, j)
                if i > 1:
                    A[k, idx(i-1, j)] = coef_x
                else:
                    # Está en la frontera izquierda: usar condición de contorno
                    bc_type, bc_val = self.bc['left']
                    if bc_type == 'dirichlet':
                        if callable(bc_val):
                            val = bc_val(xi - dx, yj)
                        else:
                            val = bc_val
                        b[k] -= coef_x * val
                    else:
                        # Neumann: se maneja con puntos fantasma (se hará más adelante)
                        raise NotImplementedError("Neumann en frontera izquierda no implementado en esta versión.")

                # Derecha (i+1, j)
                if i < Nx - 1:
                    A[k, idx(i+1, j)] = coef_x
                else:
                    bc_type, bc_val = self.bc['right']
                    if bc_type == 'dirichlet':
                        if callable(bc_val):
                            val = bc_val(xi + dx, yj)
                        else:
                            val = bc_val
                        b[k] -= coef_x * val
                    else:
                        raise NotImplementedError("Neumann en frontera derecha no implementado.")

                # Abajo (i, j-1)
                if j > 1:
                    A[k, idx(i, j-1)] = coef_y
                else:
                    bc_type, bc_val = self.bc['bottom']
                    if bc_type == 'dirichlet':
                        if callable(bc_val):
                            val = bc_val(xi, yj - dy)
                        else:
                            val = bc_val
                        b[k] -= coef_y * val
                    else:
                        raise NotImplementedError("Neumann en frontera inferior no implementado.")

                # Arriba (i, j+1)
                if j < Ny - 1:
                    A[k, idx(i, j+1)] = coef_y
                else:
                    bc_type, bc_val = self.bc['top']
                    if bc_type == 'dirichlet':
                        if callable(bc_val):
                            val = bc_val(xi, yj + dy)
                        else:
                            val = bc_val
                        b[k] -= coef_y * val
                    else:
                        raise NotImplementedError("Neumann en frontera superior no implementado.")

        return A, b

    # ------------------------------------------------------------------
    # 3. SOLUCIÓN DEL SISTEMA
    # ------------------------------------------------------------------

    def solve(self):
        """Resuelve el sistema lineal y almacena la solución en self.U."""
        if self.solver == 'direct':
            self._solve_direct()
        elif self.solver == 'iterative':
            self._solve_iterative()
        else:
            raise ValueError("Método no soportado. Use 'direct' o 'iterative'.")

    def _solve_direct(self):
        """Resuelve el sistema mediante numpy.linalg.solve."""
        A, b = self._build_system()
        u_interior = np.linalg.solve(A, b)

        # Reconstruir matriz completa (incluyendo fronteras)
        self.U = self._interior_to_full(u_interior)

    def _solve_iterative(self):
        """Resuelve el sistema mediante Gauss-Seidel (método iterativo)."""
        # Inicialización: interpolación lineal entre bordes
        U = np.zeros((self.Nx + 1, self.Ny + 1))
        self._apply_bc_dirichlet(U)

        # Coeficientes
        dx2 = self.dx**2
        dy2 = self.dy**2
        denom = 2.0 / dx2 + 2.0 / dy2

        for it in range(self.max_iter):
            U_old = U.copy()
            # Recorremos los puntos interiores en orden natural
            for i in range(1, self.Nx):
                for j in range(1, self.Ny):
                    xi = self.x[i]
                    yj = self.y[j]
                    # Actualización Gauss-Seidel
                    U[i, j] = ((U[i-1, j] + U[i+1, j]) / dx2 +
                               (U[i, j-1] + U[i, j+1]) / dy2 +
                               self.f(xi, yj)) / denom
            # Aplicar condiciones de contorno (Dirichlet)
            self._apply_bc_dirichlet(U)

            # Calcular residuo
            residual = np.linalg.norm(U - U_old) / np.linalg.norm(U + 1e-12)
            self.residual_history.append(residual)
            if residual < self.tol:
                self.iterations = it + 1
                self.U = U
                return

        print(f"⚠️ Gauss-Seidel no convergió en {self.max_iter} iteraciones. Residuo final: {residual:.2e}")
        self.U = U
        self.iterations = self.max_iter

    def _apply_bc_dirichlet(self, U):
        """Aplica condiciones de contorno Dirichlet a la matriz U."""
        for side, (bc_type, bc_val) in self.bc.items():
            if bc_type != 'dirichlet':
                continue
            if side == 'left':
                for j in range(self.Ny + 1):
                    yj = self.y[j]
                    U[0, j] = bc_val(0, yj) if callable(bc_val) else bc_val
            elif side == 'right':
                for j in range(self.Ny + 1):
                    yj = self.y[j]
                    U[-1, j] = bc_val(self.Lx, yj) if callable(bc_val) else bc_val
            elif side == 'bottom':
                for i in range(self.Nx + 1):
                    xi = self.x[i]
                    U[i, 0] = bc_val(xi, 0) if callable(bc_val) else bc_val
            elif side == 'top':
                for i in range(self.Nx + 1):
                    xi = self.x[i]
                    U[i, -1] = bc_val(xi, self.Ly) if callable(bc_val) else bc_val

    def _interior_to_full(self, u_interior):
        """Convierte el vector de soluciones interiores a matriz completa."""
        U = np.zeros((self.Nx + 1, self.Ny + 1))
        # Rellenar fronteras
        self._apply_bc_dirichlet(U)
        # Rellenar interiores
        k = 0
        for j in range(1, self.Ny):
            for i in range(1, self.Nx):
                U[i, j] = u_interior[k]
                k += 1
        return U

    # ------------------------------------------------------------------
    # 4. MÉTODOS DE CONSULTA
    # ------------------------------------------------------------------

    def compute_error(self):
        """Calcula el error comparando con la solución exacta (si se proporcionó)."""
        if self.U is None:
            raise ValueError("Primero debe ejecutar solve()")
        if self.exact_solution is None:
            return None
        U_exact = np.zeros_like(self.U)
        for i in range(self.Nx + 1):
            for j in range(self.Ny + 1):
                U_exact[i, j] = self.exact_solution(self.x[i], self.y[j])
        error = np.abs(self.U - U_exact)
        max_error = np.max(error)
        l2_error = np.sqrt(np.mean(error**2))
        return max_error, l2_error, U_exact

    # ------------------------------------------------------------------
    # 5. GRÁFICAS
    # ------------------------------------------------------------------

    def plot_solution(self, save=True, filename='poisson_solution.png'):
        """Grafica la solución en 3D y en mapa de colores."""
        if self.U is None:
            self.solve()
        fig = plt.figure(figsize=(14, 6))

        # Subplot 1: 3D
        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(self.X, self.Y, self.U, cmap='viridis', edgecolor='none')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('u')
        ax1.set_title('Solución 3D')
        fig.colorbar(surf, ax=ax1, shrink=0.5)

        # Subplot 2: mapa de colores
        ax2 = fig.add_subplot(122)
        im = ax2.contourf(self.X, self.Y, self.U, levels=20, cmap='viridis')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Mapa de colores')
        fig.colorbar(im, ax=ax2)

        if save:
           plt.savefig(os.path.join(os.getcwd(), filename), dpi=150, bbox_inches='tight')
        plt.show()

    def plot_error(self, save=True, filename='poisson_error.png'):
        """Grafica el error (si se tiene solución exacta)."""
        if self.U is None:
            self.solve()
        error_info = self.compute_error()
        if error_info is None:
            print("No se proporcionó solución exacta.")
            return
        max_err, l2_err, U_exact = error_info
        error = np.abs(self.U - U_exact)

        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(self.X, self.Y, error, cmap='hot', edgecolor='none')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('Error')
        ax1.set_title(f'Error 3D (max={max_err:.2e}, L2={l2_err:.2e})')
        fig.colorbar(surf, ax=ax1, shrink=0.5)

        ax2 = fig.add_subplot(122)
        im = ax2.contourf(self.X, self.Y, error, levels=20, cmap='hot')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Mapa de error')
        fig.colorbar(im, ax=ax2)

        if save:
            plt.savefig(os.path.join(os.getcwd(), filename), dpi=150, bbox_inches='tight')
        plt.show()

    def plot_convergence(self, save=True, filename='poisson_convergence.png'):
        """Grafica la convergencia del error al refinar la malla."""
        if self.exact_solution is None:
            print("Se necesita solución exacta para analizar convergencia.")
            return

        N_values = [4, 8, 16, 32, 64]
        errors_max = []
        errors_l2 = []
        h_values = []

        for N in N_values:
            # Crear una nueva instancia con malla más fina
            temp = Poisson2D(
                Lx=self.Lx, Ly=self.Ly,
                Nx=N, Ny=N,
                f=self.f, bc=self.bc,
                solver=self.solver,
                exact_solution=self.exact_solution
            )
            temp.solve()
            err = temp.compute_error()
            if err is not None:
                max_err, l2_err, _ = err
                errors_max.append(max_err)
                errors_l2.append(l2_err)
                h_values.append(temp.dx)

        if not h_values:
            return

        plt.figure(figsize=(10, 6))
        plt.loglog(h_values, errors_max, 'bo-', label='Error máximo', linewidth=2)
        plt.loglog(h_values, errors_l2, 'ro-', label='Error L2', linewidth=2)
        # Referencia de pendiente O(h^2)
        h_ref = np.array(h_values)
        ref = errors_max[0] * (h_ref / h_ref[0])**2
        plt.loglog(h_ref, ref, 'k--', label='O(h^2)')
        plt.xlabel('h')
        plt.ylabel('Error')
        plt.title('Convergencia del método de diferencias finitas')
        plt.grid(True)
        plt.legend()
        if save:
            plt.savefig(os.path.join(os.getcwd(), filename), dpi=150, bbox_inches='tight')
        plt.show()

    # ------------------------------------------------------------------
    # 6. TABLA DE RESULTADOS
    # ------------------------------------------------------------------

    def print_table(self):
        """Imprime una tabla con algunos valores de la solución."""
        if self.U is None:
            self.solve()
        print("\n" + "="*80)
        print(f"RESULTADOS - POISSON 2D (Nx={self.Nx}, Ny={self.Ny})")
        print("="*80)
        print(f"{'x\\y':^8}", end="")
        # Mostrar algunos puntos en y
        y_indices = [0, self.Ny//4, self.Ny//2, 3*self.Ny//4, self.Ny]
        for j in y_indices:
            print(f"{self.y[j]:^10.3f}", end="")
        print()
        print("-"*80)
        for i in range(0, self.Nx+1, max(1, self.Nx//5)):
            print(f"{self.x[i]:^8.3f}", end="")
            for j in y_indices:
                print(f"{self.U[i, j]:^10.4f}", end="")
            print()
        print("="*80)

        # Si hay solución exacta, mostrar errores
        if self.exact_solution is not None:
            max_err, l2_err, _ = self.compute_error()
            print(f"Error máximo: {max_err:.2e}")
            print(f"Error L2:     {l2_err:.2e}")

    # ------------------------------------------------------------------
    # 7. MÉTODO PARA GUARDAR RESULTADOS (CSV)
    # ------------------------------------------------------------------

    def save_to_csv(self, filename='poisson_solution.csv'):
        """Guarda la solución en un archivo CSV."""
        if self.U is None:
            self.solve()
        import csv
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x\\y'] + list(self.y))
            for i in range(self.Nx + 1):
                writer.writerow([self.x[i]] + list(self.U[i, :]))
        print(f"Soluciones guardadas en {filename}")


# ======================================================================
# 8. FUNCIÓN PARA DEFINIR FUNCIONES DESDE TEXTO (interactivo)
# ======================================================================

def make_func(expr, vars_dict=None):
    """Crea una función a partir de una expresión en string."""
    if vars_dict is None:
        vars_dict = {"math": math}
    # Para función de dos variables: f(x,y)
    return lambda x, y: eval(expr, vars_dict, {"x": x, "y": y})


# ======================================================================
# 9. PROGRAMA PRINCIPAL (INTERACTIVO)
# ======================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("   RESOLUCIÓN DE LA ECUACIÓN DE POISSON EN 2D")
    print("   - (u_xx + u_yy) = f(x,y)")
    print("="*70 + "\n")

    # ---- 1. DOMINIO Y MALLA ----
    Lx = float(input("Longitud en x [0, Lx] (ej. 1.0): ") or "1.0")
    Ly = float(input("Longitud en y [0, Ly] (ej. 1.0): ") or "1.0")
    Nx = int(input("Número de intervalos en x Nx (ej. 20): ") or "20")
    Ny = int(input("Número de intervalos en y Ny (ej. 20): ") or "20")

    # ---- 2. FUNCIÓN FUENTE ----
    print("\nIngrese la función fuente f(x,y) (ej. 2*pi^2*sin(pi*x)*sin(pi*y)):")
    f_str = input("  f(x,y) = ") or "0"
    f_func = make_func(f_str, {"math": math})

    # ---- 3. CONDICIONES DE CONTORNO ----
    print("\nPara cada lado, elija el tipo de condición:")
    print("  d: Dirichlet (u = valor)")
    print("  n: Neumann  (u_n = valor)  [no implementado en esta versión]")
    bc = {}
    for side in ['left', 'right', 'bottom', 'top']:
        print(f"\nLado {side}:")
        bc_type = input("  Tipo (d/n): ").lower()
        if bc_type == 'd':
            val_str = input("  Valor (constante o función de x,y, ej. 0): ") or "0"
            val_func = make_func(val_str, {"math": math})
            bc[side] = ('dirichlet', val_func)
        elif bc_type == 'n':
            raise NotImplementedError("Condiciones Neumann no implementadas en esta versión.")
        else:
            print("  Opción no válida. Se usará Dirichlet 0.")
            bc[side] = ('dirichlet', 0)

    # ---- 4. MÉTODO DE SOLUCIÓN ----
    print("\nSeleccione el método de resolución del sistema lineal:")
    print("  1. Directo (numpy.linalg.solve) - recomendado para mallas pequeñas")
    print("  2. Iterativo (Gauss-Seidel) - para mallas grandes")
    solver_choice = input("Opción (1/2): ") or "1"
    solver = 'direct' if solver_choice == '1' else 'iterative'

    # ---- 5. SOLUCIÓN EXACTA (opcional) ----
    print("\n¿Dispone de solución exacta para comparar? (s/n)")
    has_exact = input().lower() == "s"
    exact_func = None
    if has_exact:
        exact_str = input("Ingrese u_exacta(x,y) en términos de 'x' e 'y': ")
        exact_func = make_func(exact_str, {"math": math})

    # ---- 6. CREAR INSTANCIA Y RESOLVER ----
    poisson = Poisson2D(
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        f=f_func, bc=bc,
        solver=solver,
        exact_solution=exact_func
    )
    print("\nResolviendo...")
    poisson.solve()

    # ---- 7. MOSTRAR RESULTADOS ----
    poisson.print_table()

    # ---- 8. GRÁFICAS ----
    print("\nGenerando gráficas...")
    poisson.plot_solution(save=True, filename='poisson_solution.png')
    if exact_func is not None:
        poisson.plot_error(save=True, filename='poisson_error.png')
    poisson.plot_convergence(save=True, filename='poisson_convergence.png')

    # ---- 9. GUARDAR CSV ----
    save_csv = input("\n¿Guardar resultados en CSV? (s/n): ").lower() == "s"
    if save_csv:
        poisson.save_to_csv('poisson_results.csv')

    print("\n✅ Programa finalizado.")
