from sympy import *

# compute composite functions derivates example to refresh the mind

x, s = symbols("x s")


def finite_diff(f):
    df = (f.subs(x, x + s) - f.subs(x, x - s)) / (2 * s)
    return df


def derivative(f):
    return limit(finite_diff(f), s, 0)


# Test it against the library
z = (x**2 + 1) ** 3 - 2
dz_dx = expand(diff(z, x))
est_dz = derivative(z)

print(dz_dx)
print(est_dz)
