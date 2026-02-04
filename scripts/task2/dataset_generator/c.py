import sympy as sp

a_p, b_p = sp.symbols('a_p, b_p')
a_q, b_q = sp.symbols('a_q, b_q')
a_r, b_r = sp.symbols('a_r, b_r')

p_coeff_list = [a_p, b_p] # x^2 + a_p*x + b_p
q_coeff_list = [a_q, b_q] # x^2 + a_q*x + b_q
r_coeff_list = [a_r, b_r] # x^2 + a_r*x + b_r

p_coeff_vector = sp.Matrix(p_coeff_list)
q_coeff_vector = sp.Matrix(q_coeff_list)
r_coeff_vector = sp.Matrix(r_coeff_list)

t = sp.symbols('t')
x = sp.symbols('x')
coeff_path = (1 - t) * p_coeff_vector + t * q_coeff_vector

print(coeff_path) # Matrix([[a_p*(1 - t) + a_q*t], [b_p*(1 - t) + b_q*t]])
print(coeff_path.diff(t)) # Matrix([[-a_p + a_q], [b_p - b_q]])

hc_poly = sp.Poly(x**2 + coeff_path[0]*x + coeff_path[1], x)
print(hc_poly) # Poly((-a_p*t + a_p + a_q*t)*x - b_p*t + b_p + b_q*t, x, domain='ZZ[t,a_p,a_q,b_p,b_q]')

print(hc_poly.coeffs())

a = 2 * t
print(a.subs(t, sp.Matrix([1, 2, 3])))

expr_1 = sp.Poly((x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5), x)
print(expr_1)
print(expr_1.coeffs())

expr_2 = sp.Poly((x - 1.1) * (x - 2.1) * (x - 3.1) * (x - 4.1) * (x - 5.1), x)
print(expr_2)
print(expr_2.coeffs())