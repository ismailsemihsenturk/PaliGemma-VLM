from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:

    def __init__(
            self,
            hidden_size=768, # size of the embedding vector
            intermediate_size=3072, # lienar layer that we use in the feed forward network 
            num_hidden_layers=12, # number of layers of this vision transformer 
            num_attention_heads=12, # number of attention heads in the multi head attention
            num_channels=3, # how many channels is each image has which is RGB
            image_size=224, # for paligemma
            patch_size=16, # how many patches the image should be divided
            layer_norm_eps=1e-6, # layer normalization params
            attention_dropout=0.0,
            num_image_tokens: int = None, # how many output embeddings this Vision Transformer will output
            **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_size=hidden_size,
        self.intermediate_size=intermediate_size,
        self.num_hidden_layers=num_hidden_layers,
        self.num_attention_heads=num_attention_heads,
        self.num_channels=num_channels,
        self.image_size=image_size,
        self.patch_size=patch_size,
        self.layer_norm_eps=layer_norm_eps,
        self.attention_dropout=attention_dropout,
        self.num_image_tokens =num_image_tokens




class SiglipVisionEmbeddings(nn.Module):
    def __init__(self,config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size, # how we should slice the kernel
            padding="valid", # This indicates no padding is added
        )

        self.num_patches = (self.image_size // self.patch_size) **2
        self.num_positions= self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1,-1)),
            persistent=False
        )
    
    def forward(self,pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape 
        # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to the kernel size
        # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size

        patch_embeds = self.patch_embedding(pixel_values)  
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Embed_Dim, Num_Patches]
        # where Num_Patches = Num_Patches_H * Num_Patches_W

        embeddings = patch_embeds.flatten(2)
        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]

        embeddings = embeddings.transpose(1, 2)
        # Add position embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim]

        embeddings = embeddings + self.position_embedding(self.position_ids)
        # [Batch_Size, Num_Patches, Embed_Dim]

        return embeddings



class SiglipMLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size,Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)

        # hidden_states: [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = nn.functional.gelu(hidden_states,approximate="tanh")

        # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)
        return hidden_states



class SiglipEncoderLayer(nn.Module):
    def __init__(self,config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim,eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states:torch.Tensor) -> torch.Tensor:
        # residual: [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states

        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm1(hidden_states)

        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states, _ = self.self_attn(hidden_states = hidden_states)

        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states

        # residual: [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states

        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm2(hidden_states)

        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.mlp(hidden_states)

        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states

        return hidden_states




class SiglipVisionTransformer(nn.Module):
    def __İnit__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim,eps=config.layer_norm_eps)

    def forward(self,pixel_values: torch.Tensor) -> torch.Tensor:
        
        # Forward pass of the SiglipVisionTransformer model.

        # Args:
        # pixel_values (torch.Tensor): Input pixel values tensor.

        # Returns:
        # torch.Tensor: The final hidden state of the model.
 
        # pixel_values: [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]

        # Embed pixel values into a higher-dimensional space
        hidden_states = self.embeddings(pixel_values)
        
        # Process the embedded pixel values using the encoder
        last_hidden_state = self.encoder(inuts_embeds = hidden_states)
        
         # Apply layer normalization to the final hidden state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state



class SiglipVisionModel(nn.Module):

    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self,pixel_values) -> Tuple:
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches(num_image_tokens), Embed_Dim]
        return self.vision_model(pixel_values = pixel_values)