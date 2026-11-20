"""Double-precision spherical Bessel transform for numerical s, p, d and f shells."""
import cython
from libc.math cimport cos, fabs, sin

@cython.cdivision(True)
cdef inline double spherical_bessel(int l, double x) noexcept nogil:
    cdef double term = 1., total, previous, current, following
    cdef int n
    if fabs(x) < 1.:
        # Convergent power series avoids cancellation at small argument, including zero.
        for n in range(1, l + 1):
            term *= x / (2*n + 1)
        total = term
        for n in range(1, 17):
            term *= -x*x / (2*n*(2*l + 2*n + 1))
            total += term
            if fabs(term) <= 1e-17 * fabs(total):
                break
        return total
    previous = sin(x) / x
    if l == 0:
        return previous
    current = (previous - cos(x)) / x
    for n in range(1, l):
        following = (2*n + 1) * current / x - previous
        previous, current = current, following
    return current

@cython.boundscheck(False)
@cython.wraparound(False)
def integrate(int l, const double[::1] momenta, const double[::1] radii,
              const double[::1] weighted_radial, double[::1] values):
    cdef Py_ssize_t i, j
    cdef double total
    if l < 0 or l > 3 or radii.shape[0] != weighted_radial.shape[0] or momenta.shape[0] != values.shape[0]:
        raise ValueError("Invalid radial kernel dimensions or angular momentum")
    with nogil:
        for i in range(momenta.shape[0]):
            total = 0.
            for j in range(radii.shape[0]):
                total += weighted_radial[j] * spherical_bessel(l, momenta[i] * radii[j])
            values[i] = total
