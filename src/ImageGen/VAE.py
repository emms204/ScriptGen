import torch
from torch import nn
from diffusers import AutoencoderKL
from torchvision import transforms
from PIL import Image

class VAE(nn.Module):

    def __init__(self, pretrained_model_name_or_path="stabilityai/sd-vae-ft-mse"):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'VQVAE DEVICE: {self.device}')
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(self, I, is_sample = False, return_5dim = False, dtype = torch.float32):
        I = I.to(dtype = self.dtype)
        # print(f'Input img dtype: {I.dtype}')
        if is_sample:
            z0 = self.vae.encode(I).latent_dist.sample()
            z0 = z0 * self.vae.config.scaling_factor
        else:
            z0 = self.vae.encode(I).latent_dist.mean
            z0 = z0 * self.vae.config.scaling_factor
    
        if return_5dim:
            z0 = z0.unsqueeze(2)
        return z0
    
    def prepare_image(self, image, target_size=(512, 512)):
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        if type(image) == type(str()):
            image = Image.open(image).convert("RGB")
        elif type(image) == np.ndarray:
            image = Image.fromarray(image)
        return transform(image).unsqueeze(0)  # Add batch dimension
    
    @property
    def dtype(self):
        # Return the dtype of the vae model's parameters
        return next(self.vae.parameters()).dtype

    def to(self, device, dtype=None):
        # Move the model to the specified device and dtype if provided
        self.vae = self.vae.to(device, dtype)
        return self
    
    def batch_prepare_image(self, images, target_size = (512,512)):
        transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        image_tensors = []
        for image in images:
            if isinstance(image, str):
                img = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                img = Image.fromarray(image)
            image_tensor = transform(img)
            image_tensors.append(image_tensor)
        
        return torch.stack(image_tensors)
    
    @torch.no_grad()
    def batch_encode(self, images_batch):
        z0 = self.vae.encode(images_batch).latent_dist.sample()
        z0 = z0 * self.vae.config.scaling_factor
        return z0
    
    def unbatch(self, encoded_batch):
        return list(torch.unbind(encoded_batch, dim=0))

    @torch.no_grad()
    def decode(self, z):
        if len(z.shape) == 5:
            if z.shape[2] == 1:
                z = z.squeeze(2)
                # print(f'Z shape: {z.shape}')
            else:
                raise NotImplementedError('System cannot handle multiple frames per batch yet.')
        z = z / self.vae.config.scaling_factor
        I = self.vae.decode(z).sample
        return I
    
    def channelwise_concat(self, encoded_img, encoded_tsm_output, channel_dim = 1):
        """
        Concatenation of the encoding results. Obviously both of the object should have identical shapes.
        For now we will assume that shapes will be the following: (B, C, H, W)
        B - batch size
        C - channels
        H - height
        W - width
        """
        result_data = torch.cat((encoded_img, encoded_tsm_output), dim = channel_dim)
        return result_data
