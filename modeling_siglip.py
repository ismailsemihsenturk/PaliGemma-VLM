from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:

    def __init__(
            self,
            hidden_size=768, # size of the embedding vector
            intermediate_size=3072, # linear layer that we use in the feed forward network 
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




class SiglipAttention(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim **-0.5 # Equivalent to 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim,self.embed_dim)


    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor,Optional[torch.Tensor]]:
        # hidden_states: [Batch_Size, Num_Patches, Embed_Dim]
        batch_size, seq_len, _ = hidden_states.size()

        # query_states: [Batch_Size, Num_Patches, Embed_Dim]
        query_states = self.q_proj(hidden_states)

        # key_states: [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)

        # value_states: [Batch_Size, Num_Patches, Embed_Dim]
        value_states = self.v_proj(hidden_states)

        # query_states: [Batch_Size, Num_Patches, Num_Heads, Head_Dim] -> [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        query_states = query_states.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)

        key_states = key_states.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)

        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        # Q = X.Wq = (4,1024) x (1024,8,128) = (4,8,128) 
        # 4: Sequence
        # 1024: Hidden.Size
        # 8: N_Head
        # 128: Head_Dim (Hidden Size / Head)

        # (4,8,128).transpose(1,2) -> (8,4,128) 
        # With this transpose operation we have a matrix with a 8 little matrices inside which all have 4 heads and all heads have 128 different dimension. 
        # So we can use every HEAD individually and compute them parallel. Thus we achieve MULTI HEAD ATTENTION

        # Calculate the attention using the formula Q * K^T / sqrt(d_k). attn_weights: 
        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim] 
        # * 
        # [Batch_Size, Num_Heads, Head_Dim, Num_Patches] 
        # -> 
        # [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = (torch.matmul(query_states,key_states.transpose(2,3)) * self.scale)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len,seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size,self.num_heads,seq_len,seq_len)}, but is"
                f"{attn_weights.size()}"
            )
        
        # Apply the softmax row-wise. attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply dropput only during training
        attn_weights = nn.functional.dropout(attn_weights,p=self.dropout,training=self.training)

        # Multiply the attention weights by the value states. attn_output: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        attn_output = torch.matmul(attn_weights,value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        # To get all the heads and concat them transpose ones more and thus we get the contextualized version back.
        # Used contiguous() because we are going to reshape the tensor.
        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim] -> [Batch_Size, Num_Patches, Num_Heads, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()

        # [Batch_Size, Num_Patches, Num_Heads, Head_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        # [Batch_Size, Num_Patches, Embed_Dim]
        # We are multiplying with Wo. Doing this because we want to mix the results of parallel computed attention heads.
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights



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


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )


    def forward(
        self,
        inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = encoder_layer(hidden_states)

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