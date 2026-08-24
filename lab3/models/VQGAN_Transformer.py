import torch 
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors'] # 1024是表示被遮住的id
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        # self.load_state_dict(torch.load(load_ckpt_path, map_location='cpu'))
        self.transformer.load_state_dict(torch.load(load_ckpt_path, map_location='cpu'))
        # self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path'], map_location='cpu'), strict=True)
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        codebook_mapping, codebook_indices, _ = self.vqgan.encode(x)
        z_indices = codebook_indices.view(x.shape[0], self.num_image_tokens) 
        return codebook_mapping, z_indices
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: math.cos((math.pi / 2) * r)
        elif mode == "square":
            return lambda r: 1 - (r ** 2)
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x):
        _, z_indices = self.encode_to_z(x)
        ratio = np.random.uniform(0, 1)

        device = z_indices.device
        B, N = z_indices.shape

        num_masked = math.ceil(ratio * N)
        rand_prob = torch.rand(B, N, device=device)
        mask_pos = torch.topk(rand_prob, num_masked, dim=1).indices

        masked_z_indices = z_indices.clone()
        masked_z_indices.scatter_(-1, mask_pos, self.mask_token_id)
        logits = self.transformer(masked_z_indices)
        logits = logits[..., :self.mask_token_id]

        return logits, z_indices
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, masked_z_indices, ratio, mask, total_mask_num):
        logits = self.transformer(masked_z_indices)
        
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        logits = logits[..., :self.mask_token_id]
        probs = F.softmax(logits, dim=-1)

        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = torch.max(probs, dim=-1)

        #predicted probabilities add temperature annealing gumbel noise as confidence
        u = torch.rand_like(z_indices_predict_prob)
        g = -torch.log(-torch.log(u + 1e-9))  # gumbel noise
        
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        mask = mask.bool()
        confidence[~mask] = float('inf')

        #define how much the iteration remain predicted tokens by mask scheduling
        mask_rate = self.gamma(ratio)
        mask_num = math.ceil(mask_rate * total_mask_num)

        #sort the confidence for the rank 
        mask_bc = torch.zeros_like(mask)
        if mask_num > 0:
            mask_pos = torch.topk(confidence, mask_num, dim=-1, largest=False).indices
            mask_bc.scatter_(-1, mask_pos, True)

        ##At the end of the decoding process, add back the original(non-masked) token values
        z_indices_predict[~mask] = masked_z_indices[~mask]

        return z_indices_predict, mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
