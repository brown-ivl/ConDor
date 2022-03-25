from sympy import *
import numpy as np
import torch

# from spherical_harmonics.clebsch_gordan_decomposition import tf_clebsch_gordan_decomposition
# from spherical_harmonics.tf_spherical_harmonics import normalized_real_sh

def associated_legendre_polynomial(l, m, z, r2):
    P = 0
    if l < m:
        return P
    for k in range(int((l-m)/2)+1):
        pk = (-1.)**k * (2.)**(-l) * binomial(l, k) * binomial(2*l-2*k, l)
        pk *= (factorial(l-2*k) / factorial(l-2*k-m)) * r2**k * z**(l-2*k-m)
        P += pk
    P *= np.sqrt(float(factorial(l-m)/factorial(l+m)))
    return P

def A(m, x, y):
    a = 0
    for p in range(m+1):
        a += binomial(m, p) * x**p * y**(m-p) * cos((m-p)*(pi/2.))
    return a

def B(m, x, y):
    b = 0
    for p in range(m+1):
        b += binomial(m, p) * x**p * y**(m-p) * sin((m-p)*(pi/2.))
    return b
"""
computes the unnormalized real spherical harmonic Y_{lm} as a polynomial
of the euclidean coordinates x, y, z
"""
def real_spherical_harmonic(l, m, x, y, z):
    K = np.sqrt((2*l+1)/(2*np.pi))
    r2 = x**2 + y**2 + z**2
    if m > 0:
        Ylm = K * associated_legendre_polynomial(l, m, z, r2) * A(m, x, y)
    elif m < 0:
        Ylm = K * associated_legendre_polynomial(l, -m, z, r2) * B(-m, x, y)
    else:
        K = np.sqrt((2 * l + 1) / (4 * np.pi))
        Ylm = K * associated_legendre_polynomial(l, 0, z, r2)
    return Ylm

def binom(n, k):
    if k == 0.:
        return 1.
    return gamma(n + 1) / (gamma(n-k+1)*gamma(k+1))

"""
computes radial Zernike polynomials (divided by r^{2l})
"""
def zernike_polynomial_radial(n, l, D, r2):
    if (l > n):
        return 0
    if ((n-l) % 2 != 0):
        return 0
    R = 0
    for s in range(int((n-l) / 2) + 1):
        R += (-1)**s * binom((n-l)/2, s)*binom(s-1+(n+l+D)/2., (n-l)/2)*r2**s
    R *= (-1)**((n-l)/2)*np.sqrt(2*n+D)
    return R

"""
computes the 3D Zernike polynomials.
"""
def zernike_kernel_3D(n, l, m, x, y, z):
    r2 = x**2 + y**2 + z**2
    return zernike_polynomial_radial(n, l, 3, r2)*real_spherical_harmonic(l, m, x, y, z)
    # return real_spherical_harmonic(l, m, x, y, z)

"""
computes the monomial basis in x, y, z up to degree d
"""
def monomial_basis_3D(d):
    monoms_basis = []
    for I in range((d + 1) ** 3):
        i = I % (d + 1)
        a = int((I - i) / (d + 1))
        j = a % (d + 1)
        k = int((a - j) / (d + 1))
        if (i + j + k <= d):
            monoms_basis.append((i, j, k))

    monoms_basis = list(set(monoms_basis))
    monoms_basis = sorted(monoms_basis)
    return monoms_basis

def torch_monomial_basis_3D_idx(d):
    m = monomial_basis_3D(d)
    idx = np.zeros((len(m), 3), dtype=np.int32)
    for i in range(len(m)):
        for j in range(3):
            idx[i, j] = m[i][j]
    return torch.from_numpy(idx).to(torch.int64)

"""
computes the coefficients of a list of polynomials in the monomial basis
"""
def np_monomial_basis_coeffs(polynomials, monoms_basis):
    n_ = len(monoms_basis)
    m_ = len(polynomials)
    M = np.zeros((m_, n_))
    for i in range(m_):
        for j in range(n_):
            M[i, j] = polynomials[i].coeff_monomial(monoms_basis[j])
    return M





"""
computes the coefficients of the spherical harmonics in the monomial basis
"""
def spherical_harmonics_3D_monomial_basis(l, monoms_basis):
    x, y, z = symbols("x y z")
    n_ = len(monoms_basis)
    M = np.zeros((2*l+1, n_))
    for m in range(2*l+1):
        Y = real_spherical_harmonic(l, m-l, x, y, z)
        Y = expand(Y)
        Y = poly(Y, x, y, z)
        for i in range(n_):
            M[m, i] = N(Y.coeff_monomial(monoms_basis[i]))
    return M

"""
computes the coefficients of the Zernike polynomials in the monomial basis
"""
def zernike_kernel_3D_monomial_basis(n, l, monoms_basis):
    x, y, z = symbols("x y z")
    n_ = len(monoms_basis)
    M = np.zeros((2*l+1, n_))
    for m in range(2*l+1):
        Z = zernike_kernel_3D(n, l, m-l, x, y, z)
        Z = expand(Z)
        Z = poly(Z, x, y, z)
        for i in range(n_):
            M[m, i] = N(Z.coeff_monomial(monoms_basis[i]))
    return M


"""
computes the matrix of an offset in the monomial basis (up to degree d)
(m_1(x-a), ..., m_k(x-a)) = A(a).(m_1(x), ..., m_k(x))
"""
def np_monom_basis_offset(d):
    monoms_basis = monomial_basis_3D(d)
    n = len(monoms_basis)
    idx = np.full(fill_value=-1, shape=(n, n), dtype=np.int32)
    coeffs = np.zeros(shape=(n, n))

    for i in range(n):
        pi = monoms_basis[i][0]
        qi = monoms_basis[i][1]
        ri = monoms_basis[i][2]
        for j in range(n):
            pj = monoms_basis[j][0]
            qj = monoms_basis[j][1]
            rj = monoms_basis[j][2]
            if (pj >= pi) and (qj >= qi) and (rj >= ri):
                idx[j, i] = monoms_basis.index((pj-pi, qj-qi, rj-ri))
                coeffs[j, i] = binomial(pj, pi)*binomial(qj, qi)*binomial(rj, ri)*((-1.)**(pj-pi+qj-qi+rj-ri))
    return coeffs, idx


"""
computes the 3D zernike basis up to degree d
"""
def np_zernike_kernel_basis(d):
    monoms_basis = monomial_basis_3D(d)
    Z = []
    for l in range(d+1):
        Zl = []
        # for n in range(min(2*d - l + 1, l + 1)):
        #    if (n - l) % 2 == 0:
        for n in range(l, d+1):
            if (n - l) % 2 == 0 and d >= n:
                Zl.append(zernike_kernel_3D_monomial_basis(n, l, monoms_basis))
        Z.append(np.stack(Zl, axis=0))
    return Z

# def tf_zernike_kernel_basis(d, stack_axis=1):
#     monoms_basis = monomial_basis_3D(d)
#     Z = []
#     for l in range(d+1):
#         Zl = []
#         for n in range(l, d+1):
#             if (n - l) % 2 == 0 and d >= n:
#                 Zl.append(zernike_kernel_3D_monomial_basis(n, l, monoms_basis))
#         Z.append(tf.convert_to_tensor(np.stack(Zl, axis=stack_axis), dtype=tf.float32))
#     return Z


"""
computes the 3D spherical harmonics basis up to degree l_max
"""
def torch_spherical_harmonics_basis(l_max, concat=False):
    monoms_basis = monomial_basis_3D(l_max)
    Y = []
    for l in range(l_max+1):
        Yl = spherical_harmonics_3D_monomial_basis(l, monoms_basis)
        Y.append(torch.from_numpy(Yl).to(torch.float32))
    if concat:
        Y = torch.concat(Y, dim=0)
    return Y

def np_zernike_kernel(d, n, l):
    monoms_basis = monomial_basis_3D(d)
    assert (n >= l and (n - l) % 2 == 0)
    return zernike_kernel_3D_monomial_basis(n, l, monoms_basis)

def torch_eval_monom_basis(x, d, idx=None):
    """
    evaluate monomial basis up to degree d
    """

    batch_size = x.shape[0]
    num_points = x.shape[1]

    if idx is None:
        idx = torch_monomial_basis_3D_idx(d)
    y = []
    for i in range(3):
        pows = torch.reshape(torch.range(0, d), (1, 1, d+1)).to(torch.float32)
        yi = torch.pow(x[..., i].unsqueeze(-1), pows.type_as(x))
        y.append(yi[..., idx[..., i]])
    y = torch.stack(y, dim=-1)
    y = torch.prod(y, dim=-1, keepdims=False)
    return y





# class SphericalHarmonicsGaussianKernels:
#     def __init__(self, l_max, gaussian_scale, num_shells, transpose=False, bound=True):
#         super(SphericalHarmonicsGaussianKernels, self).__init__()
#         self.l_max = l_max
#         self.monoms_idx = torch_monomial_basis_3D_idx(l_max)
#         self.gaussian_scale = gaussian_scale
#         self.num_shells = num_shells
#         self.transpose = True
#         self.Y = torch_spherical_harmonics_basis(l_max, concat=True)
#         self.split_size = []
#         self.sh_idx = []
#         self.bound = bound
#         for l in range(l_max + 1):
#             self.split_size.append(2*l+1)
#             self.sh_idx += [l]*(2*l+1)
#         self.sh_idx = torch.from_numpy(np.array(self.sh_idx)).to(torch.int32)

#     def build(self, input_shape):
#         super(SphericalHarmonicsGaussianKernels, self).build(input_shape)

#     def call(self, x):
#         if "patches dist" in x:
#             patches_dist = tf.expand_dims(x["patches dist"], axis=-1)
#         else:
#             patches_dist = tf.norm(x["patches"], axis=-1, keepdims=True)
#         normalized_patches = tf.divide(x["patches"], tf.maximum(patches_dist, 0.000001))
#         if self.transpose:
#             normalized_patches = -normalized_patches
#         monoms_patches = torch_eval_monom_basis(normalized_patches, self.l_max, idx=self.monoms_idx)
#         sh_patches = tf.einsum('ij,bvpj->bvpi', self.Y, monoms_patches)
#         shells_rad = tf.range(self.num_shells, dtype=tf.float32) / (self.num_shells-1)

#         shells_rad = tf.reshape(shells_rad, (1, 1, 1, -1))
#         shells = tf.subtract(patches_dist, shells_rad)
#         shells = tf.exp(-self.gaussian_scale*tf.multiply(shells, shells))
#         shells_sum = tf.reduce_sum(shells, axis=-1, keepdims=True)
#         shells = tf.divide(shells, tf.maximum(shells_sum, 0.000001))



#         shells = tf.expand_dims(shells, axis=-2)
#         if self.bound:
#             shells = tf.where(tf.expand_dims(patches_dist, axis=-1) <= 1., shells, 0.)




#         sh_patches = tf.expand_dims(sh_patches, axis=-1)
#         sh_patches = tf.multiply(shells, sh_patches)


#         # L2 norm
#         l2_norm = tf.reduce_sum(tf.multiply(sh_patches, sh_patches), axis=2, keepdims=True)
#         # l2_norm = tf.reduce_mean(l2_norm, axis=1, keepdims=True)
#         l2_norm = tf.split(l2_norm, num_or_size_splits=self.split_size, axis=-2)
#         Y = []
#         for l in range(len(l2_norm)):
#             ml = tf.reduce_sum(l2_norm[l], axis=-2, keepdims=True)
#             ml = tf.sqrt(ml)
#             Y.append(ml)
#         l2_norm = tf.concat(Y, axis=-2)
#         l2_norm = tf.reduce_mean(l2_norm, axis=1, keepdims=True)
#         l2_norm = tf.maximum(l2_norm, 1e-8)
#         l2_norm = tf.gather(l2_norm, self.sh_idx, axis=-2)
#         sh_patches = tf.divide(sh_patches, l2_norm)

#         return sh_patches



# def zernike_multiplicity(d):
#     m = [0]*(d+1)
#     for n in range(d + 1):
#         for l in range(n + 1):
#             if (n - l) % 2 == 0:
#                 m[l] += 1
#     return m

# def zernike_split_size(d):
#     m = zernike_multiplicity(d)
#     s = []
#     for l in range(len(m)):
#         s.append((2*l+1)*m[l])
#     return s

# def spherical_harmonics_to_zernike_format(x, split=True):
#     l_max = 0
#     for l in x:
#         if l.isnumeric():
#             l_max = max(l_max, int(l))
#     m = zernike_multiplicity(l_max)
#     stack = not split
#     if stack:
#         y = []
#     else:
#         y = dict()
#     for l in x:
#         if l.isnumeric():
#             sl = list(x[l].shape)
#             assert (x[l].shape[-1] % m[int(l)] == 0)
#             if stack:
#                 sl[-1] = int(sl[-1] / m[int(l)])
#                 sl[-2] = int(sl[-2] * m[int(l)])
#                 y.append(tf.reshape(x[l], sl))
#             else:
#                 sl = sl[:-1]
#                 yl = tf.reshape(x[l], sl + [m[int(l)], -1])
#                 y[l] = yl
#     if stack:
#         y = tf.concat(y, axis=-2)
#     return y

# def split_spherical_harmonics_coeffs(x):
#     l_max = int(np.sqrt(x.shape[-2]) - 0.99)
#     split_size = []
#     for l in range(l_max + 1):
#         split_size.append(2*l+1)
#     return tf.split(x, num_or_size_splits=split_size, axis=-2)

# def split_zernike_coeffs(x, d):
#     s = zernike_split_size(d)
#     return tf.split(x, num_or_size_splits=s, axis=-2)

# def spherical_harmonics_format(x):
#     y = dict()
#     m = zernike_multiplicity(len(x)-1)


#     for l in range(len(x)):
#         sl = list(x[l].shape)

#         sl[-2] = int(sl[-2] / m[l])
#         sl[-1] = -1

#         y[str(l)] = tf.reshape(x[l], sl)
#     return y

# def zernike_eval(coeffs, z):
#     """
#     evaluate zernike functions given their coeffs
#     z = Zernike(x) where x are the evaluation points
#     coeffs are given in a splited spherical harmonics format
#     """
#     # convert coeffs to Zernike format
#     z_coeffs = spherical_harmonics_to_zernike_format(coeffs, split=False)
#     return tf.einsum('bpz,bvzc->bvpc', z, z_coeffs)


def spherical_harmonics_coeffs(values, z, d):
    z_coeffs = tf.einsum('bpz,bvpc->bvzc', z, values)
    z_coeffs = z_coeffs / float(z.shape[1])
    z_coeffs = split_zernike_coeffs(z_coeffs, d)
    y = spherical_harmonics_format(z_coeffs)
    return y

# class ZernikePolynomials(tf.keras.layers.Layer):
#     def __init__(self, d, split=True, gaussian_scale=None):
#         super(ZernikePolynomials, self).__init__()
#         self.d = d
#         self.split = split
#         self.gaussian_scale = gaussian_scale
#         # self.add_channel_axis = add_channel_axis
#         self.monoms_idx = torch_monomial_basis_3D_idx(d)
#         Z = tf_zernike_kernel_basis(d)
#         self.split_size = []
#         self.Z = []
#         k = 0
#         for l in range(len(Z)):
#             self.split_size.append(Z[l].shape[0] * Z[l].shape[1])
#             self.Z.append(tf.reshape(Z[l], (-1, Z[l].shape[-1])))
#         self.Z = tf.concat(self.Z, axis=0)
#     def build(self, input_shape):
#         super(ZernikePolynomials, self).build(input_shape)

#     def call(self, x):
#         monoms = torch_eval_monom_basis(x, self.d, idx=self.monoms_idx)
#         z = tf.einsum('ij,...j->...i', self.Z, monoms)

#         if self.gaussian_scale is not None:
#             # c = tf.reduce_mean(x, axis=-2, keepdims=True)
#             # x = tf.subtract(x, c)
#             n2 = tf.multiply(x, x)
#             n2 = tf.reduce_sum(n2, axis=-1, keepdims=True)
#             g = tf.exp(-self.gaussian_scale*n2)
#             z = tf.multiply(g, z)


#         #  if self.add_channel_axis:
#             # z = tf.expand_dims(z, axis=-1)
#         if self.split:
#             z = tf.split(z, num_or_size_splits=self.split_size, axis=-1)
#         return z

# class SphericalHarmonics(tf.keras.layers.Layer):
#     def __init__(self, l_max, split=True, add_channel_axis=True):
#         super(SphericalHarmonics, self).__init__()
#         self.l_max = l_max
#         self.split = split
#         self.add_channel_axis = add_channel_axis
#         self.monoms_idx = torch_monomial_basis_3D_idx(l_max)
#         self.Y = torch_spherical_harmonics_basis(l_max=l_max, concat=True)
#         self.split_size = []
#         for l in range(l_max+1):
#             self.split_size.append(2*l+1)
#     def build(self, input_shape):
#         super(SphericalHarmonics, self).build(input_shape)

#     def call(self, x):
#         x = tf.math.l2_normalize(x, axis=-1)
#         monoms = torch_eval_monom_basis(x, self.l_max, idx=self.monoms_idx)
#         y = tf.einsum('ij,...j->...i', self.Y, monoms)
#         if self.add_channel_axis:
#             y = tf.expand_dims(z, axis=-1)
#         if self.split:
#             y = tf.split(y, num_or_size_splits=self.split_size, axis=-1)
#         return y

# class SphericalHarmonicsShellsKernels(tf.keras.layers.Layer):
#     def __init__(self, l_max, stack=True):
#         super(SphericalHarmonicsShellsKernels, self).__init__()
#         self.l_max = l_max
#         self.stack = stack
#         self.monoms_idx = torch_monomial_basis_3D_idx(l_max)
#         self.gaussian_scale = 4*0.69314718056*(l_max**2)
#         self.Y = torch_spherical_harmonics_basis(l_max, concat=True)
#         self.split_size = []
#         self.sh_idx = []
#         for l in range(l_max + 1):
#             self.split_size.append(2*l+1)


#     def build(self, input_shape):
#         super(SphericalHarmonicsShellsKernels, self).build(input_shape)

#     def call(self, x):
#         if "patches dist" in x:
#             patches_dist = tf.expand_dims(x["patches dist"], axis=-1)
#         else:
#             patches_dist = tf.norm(x["patches"], axis=-1, keepdims=True)

#         normalized_patches = tf.divide(x["patches"], tf.maximum(patches_dist, 0.000001))
#         monoms_patches = torch_eval_monom_basis(normalized_patches, self.l_max, idx=self.monoms_idx)
#         sh_patches = tf.einsum('ij,...j->...i', self.Y, monoms_patches)
#         shells_rad = tf.range(self.l_max+1, dtype=tf.float32) / self.l_max

#         shells_rad = tf.reshape(shells_rad, (1, 1, 1, -1))
#         shells = tf.subtract(patches_dist, shells_rad)
#         shells = tf.exp(-self.gaussian_scale*tf.multiply(shells, shells))
#         shells_sum = tf.reduce_sum(shells, axis=-1, keepdims=True)
#         shells = tf.divide(shells, tf.maximum(shells_sum, 0.000001))
#         g = tf.reduce_sum(shells, axis=2, keepdims=True)
#         g = tf.reduce_mean(g, axis=[1, -1], keepdims=True)
#         shells = tf.divide(shells, tf.maximum(g, 0.000001))
#         sh_patches = tf.expand_dims(sh_patches, axis=-1)
#         sh_patches = tf.split(sh_patches, num_or_size_splits=self.split_size, axis=-2)
#         shells = tf.expand_dims(shells, axis=-2)



#         sh_shells_patches = []
#         for l in range(len(sh_patches)):
#             sh_shells_patches.append(tf.multiply(shells[..., l:], sh_patches[l]))

#         if self.stack:
#             for l in range(len(sh_shells_patches)):
#                 sl = list(sh_shells_patches[l].shape)
#                 sl = sl[:-1]
#                 sl[-1] = -1
#                 sh_shells_patches[l] = tf.reshape(sh_shells_patches[l], sl)
#             sh_shells_patches = tf.concat(sh_shells_patches, axis=-1)
#         return sh_shells_patches



if __name__=="__main__":

    x = (torch.ones((2, 4, 3)) * torch.range(0, 3).unsqueeze(0).unsqueeze(-1)).cuda()
    y = torch_eval_monom_basis(x, 2)
    print(x)
    print(y, y.shape)