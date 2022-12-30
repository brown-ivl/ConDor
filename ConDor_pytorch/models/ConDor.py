import torch
from models.TFN import TFN
from models.layers import MLP, MLP_layer
from utils.group_points import GroupPoints
from utils.train_utils import orthonormalize_basis
from spherical_harmonics.spherical_cnn import torch_fibonnacci_sphere_sampling, SphericalHarmonicsEval, SphericalHarmonicsCoeffs, zernike_monoms
from spherical_harmonics.kernels import SphericalHarmonicsGaussianKernels, ShGaussianKernelConv
from models.layers import MLP, MLP_layer, set_sphere_weights, apply_layers, type_1
from utils.pooling import kd_pooling_1d

class ConDor(torch.nn.Module):

    def __init__(self, num_capsules = 10, num_frames = 1, sphere_samples = 64, bn_momentum = 0.75, mlp_units = [[32, 32], [64, 64], [128, 256]]):
 
        super(ConDor, self).__init__()

        self.bn_momentum = 0.75
        self.basis_dim = 3
        self.l_max = [3, 3, 3]
        self.l_max_out = [3, 3, 3]
        self.num_shells = [3, 3, 3]
        
        
        self.num_capsules = num_capsules
        self.num_frames = num_frames
        self.mlp_units = mlp_units
        self.TFN_arch = TFN(sphere_samples = sphere_samples, bn_momentum = bn_momentum, mlp_units = [[32, 32], [64, 64], [128, 256]], l_max = self.l_max, l_max_out = self.l_max_out, num_shells = self.num_shells)
        self.S2 = torch_fibonnacci_sphere_sampling(sphere_samples)

        self.basis_mlp = []
        self.basis_layer = []

        self.basis_units = [64]
        for frame_num in range(num_frames):
            self.basis_mlp.append(MLP_layer(in_channels = self.mlp_units[-1][-1], units = self.basis_units, bn_momentum = self.bn_momentum))
            self.basis_layer.append(MLP(in_channels = self.basis_units[-1], out_channels=self.basis_dim, apply_norm = False))

        self.basis_mlp = torch.nn.Sequential(*self.basis_mlp)
        self.basis_layer = torch.nn.Sequential(*self.basis_layer)

        self.translation_mlp = (MLP_layer(in_channels = self.mlp_units[-1][-1], units = self.basis_units, bn_momentum = self.bn_momentum))
        self.translation_layer = MLP(in_channels = self.basis_units[-1], out_channels=1, apply_norm = False)

        self.code_dim = 64
        self.code_layer_params = [128]
        self.code_mlp = MLP_layer(in_channels = self.mlp_units[-1][-1], units = self.code_layer_params, bn_momentum = self.bn_momentum)
        self.code_layer = MLP(in_channels = self.code_layer_params[-1], out_channels=self.code_dim, apply_norm = False)

        self.points_inv_layer = MLP(in_channels = 128, out_channels=self.basis_dim, apply_norm = False)

        self.segmentation_layer_params = [256, 128, num_capsules]
        self.capsules_mlp = MLP_layer(in_channels = 384, units=[256, 128, num_capsules], bn_momentum=self.bn_momentum)
        self.num_frames = num_frames
        self.type_1                     = SphericalHarmonicsCoeffs(l_list=[1], base=self.S2)
        self.latent_Ceoffs              = SphericalHarmonicsCoeffs(l_max=self.l_max_out[-1], base=self.S2)
        self.zernike_monoms_harmonics   = zernike_monoms(self.l_max_out[-1])
        


    def forward(self, x):
        """
        x - B, N, 3 - Batch of point clouds that are kdtree indexed for pooling
        """

        # Compute TFN features
        F = self.TFN_arch(x)
        F_translation = F
        
        # Equivariant Basis
        E = []
        # Equivariant translation
        T = []
        
        # Compute equivariant layers
        for frame_num in range(self.num_frames):
            basis = self.basis_mlp[frame_num](F)
            basis = self.basis_layer[frame_num](basis)
            y_dict = self.type_1.compute(basis)
            basis = y_dict['1']    
            basis = torch.nn.functional.normalize(basis, dim=-1, p = 2, eps = 1e-6)
            E.append(basis)

        # Predicting amodal translation
        translation = self.translation_mlp(F_translation)
        y_dict = self.type_1.compute(translation)
        translation = y_dict['1'] 
        translation = self.translation_layer(translation)
        translation = torch.stack([translation[:, 2], translation[:, 0], translation[:, 1]], dim = -1)
        T.append(translation)


        latent_code = self.code_mlp(F)
        latent_code = self.code_layer(latent_code)
        latent_code = self.latent_Ceoffs.compute(latent_code)
        
        z = self.zernike_monoms_harmonics.compute(x)
        points_code = []

        points_inv = None
        for l in latent_code:
            # Compute the invariant embedding <F, Y>
            p = torch.einsum('bmi,bvmj->bvij', latent_code[l], z[int(l)])
            shape = list(p.shape)
            shape = shape[:-1]
            shape[-1] = -1
            p = torch.reshape(p, shape)
            points_code.append(p)
            if int(l) == 1:
                points_inv = p

        points_code = torch.cat(points_code, dim=-1)

        capsules = 2.*self.capsules_mlp(points_code)
        capsules = torch.nn.functional.softmax(capsules, dim=-1)

        points_inv = self.points_inv_layer(points_inv)


        out = {"T": T, "caps": capsules, "points_inv": points_inv, "E": E}

        return out

class ConDor_SO3(torch.nn.Module):

    def __init__(self, num_capsules = 10, num_frames = 1, sphere_samples = 64, bn_momentum = 0.75, mlp_units = [[32, 32], [64, 64], [128, 256]]):
 
        super(ConDor_SO3, self).__init__()

        self.bn_momentum = 0.75
        self.basis_dim = 3
        self.l_max = [3, 3, 3]
        self.l_max_out = [3, 3, 3]
        self.num_shells = [3, 3, 3]
        
        
        self.num_capsules = num_capsules
        self.num_frames = num_frames
        self.mlp_units = mlp_units
        self.TFN_arch = TFN(sphere_samples = sphere_samples, bn_momentum = bn_momentum, mlp_units = [[32, 32], [64, 64], [128, 256]], l_max = self.l_max, l_max_out = self.l_max_out, num_shells = self.num_shells)
        self.S2 = torch_fibonnacci_sphere_sampling(sphere_samples)

        self.basis_mlp = []
        self.basis_layer = []

        self.basis_units = [64]
        for frame_num in range(num_frames):
            self.basis_mlp.append(MLP_layer(in_channels = self.mlp_units[-1][-1], units = self.basis_units, bn_momentum = self.bn_momentum))
            self.basis_layer.append(MLP(in_channels = self.basis_units[-1], out_channels=self.basis_dim - 1, apply_norm = False))

        self.basis_mlp = torch.nn.Sequential(*self.basis_mlp)
        self.basis_layer = torch.nn.Sequential(*self.basis_layer)

        self.translation_mlp = (MLP_layer(in_channels = self.mlp_units[-1][-1], units = self.basis_units, bn_momentum = self.bn_momentum))
        self.translation_layer = MLP(in_channels = self.basis_units[-1], out_channels=1, apply_norm = False)

        self.code_dim = 64
        self.code_layer_params = [128]
        self.code_mlp = MLP_layer(in_channels = self.mlp_units[-1][-1], units = self.code_layer_params, bn_momentum = self.bn_momentum)
        self.code_layer = MLP(in_channels = self.code_layer_params[-1], out_channels=self.code_dim, apply_norm = False)

        self.points_inv_layer = MLP(in_channels = 128, out_channels=self.basis_dim, apply_norm = False)

        self.segmentation_layer_params = [256, 128, num_capsules]
        self.capsules_mlp = MLP_layer(in_channels = 384, units=[256, 128, num_capsules], bn_momentum=self.bn_momentum)
        self.num_frames = num_frames
        self.type_1                     = SphericalHarmonicsCoeffs(l_list=[1], base=self.S2)
        self.latent_Ceoffs              = SphericalHarmonicsCoeffs(l_max=self.l_max_out[-1], base=self.S2)
        self.zernike_monoms_harmonics   = zernike_monoms(self.l_max_out[-1])
        

        # Frame with determinant -1
        self.symmetry_frame = torch.nn.Parameter(orthonormalize_basis(torch.randn(num_frames, 3, 3))) # F, 3, 3
        


    def forward(self, x):
        """
        x - B, N, 3 - Batch of point clouds that are kdtree indexed for pooling
        """

        # Compute TFN features
        F = self.TFN_arch(x)
        F_translation = F
        
        # Equivariant Basis
        E = []
        # Equivariant translation
        T = []
        
        # Compute equivariant layers
        for frame_num in range(self.num_frames):
            basis = self.basis_mlp[frame_num](F)
            basis = self.basis_layer[frame_num](basis)
            y_dict = self.type_1.compute(basis)
            basis = y_dict['1']    
            basis = torch.nn.functional.normalize(basis, dim=1, p = 2, eps = 1e-6)
            last_axis = torch.cross(basis[..., 0], basis[..., 1]).unsqueeze(-1)
            basis = torch.cat([basis, last_axis], -1) # B, 3, 3
            # print(basis.shape)
            E.append(basis)

        # Predicting amodal translation
        translation = self.translation_mlp(F_translation)
        y_dict = self.type_1.compute(translation)
        translation = y_dict['1'] 
        translation = self.translation_layer(translation)
        translation = torch.stack([translation[:, 2], translation[:, 0], translation[:, 1]], dim = -1)
        T.append(translation)


        latent_code = self.code_mlp(F)
        latent_code = self.code_layer(latent_code)
        latent_code = self.latent_Ceoffs.compute(latent_code)
        
        z = self.zernike_monoms_harmonics.compute(x)
        points_code = []

        points_inv = None
        for l in latent_code:
            # Compute the invariant embedding <F, Y>
            p = torch.einsum('bmi,bvmj->bvij', latent_code[l], z[int(l)])
            shape = list(p.shape)
            shape = shape[:-1]
            shape[-1] = -1
            p = torch.reshape(p, shape)
            points_code.append(p)
            if int(l) == 1:
                points_inv = p

        points_code = torch.cat(points_code, dim=-1)

        capsules = 2.*self.capsules_mlp(points_code)
        capsules = torch.nn.functional.softmax(capsules, dim=-1)

        points_inv = self.points_inv_layer(points_inv)


        out = {"T": T, "caps": capsules, "points_inv": points_inv, "E": E, "symmetry_frame": self.symmetry_frame}
        
        return out




        # if rotation_frame:


if __name__=="__main__":

    x = torch.randn((16, 256, 3))#.cuda()
    model = ConDor_SO3(num_capsules = 10, num_frames = 5)#.cuda()
    
    with torch.no_grad():
        out = model(x)

    for key in out:
        if type(out[key]) != list:
            print(key, " ", out[key].shape)
        else:
            print(key, " list length: ", len(out[key]), " dim: ", out[key][0].shape)
