import torch
from torch import nn
from typing import Optional,Tuple,List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel




class GemmaConfig():

    def __init__(
            self,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim = 256,
            max_position_embeddings = 8192,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            attention_bias=False,
            attention_dropout=0.0,
            pad_token_id=None,
            **kwargs,
    ):
        
        super().__init__()
        self.vocab_size = vocab_size
        # Indicates how much the maximum number of positions our modal has been trained upon. Which is necessary for the rotary positional encodings
        self.max_position_embeddings = max_position_embeddings
        # What is the size of embedding vector of each token  
        self.hidden_size = hidden_size
        # The intermediate size of the Feedforward layer (SigLip)
        self.intermediate_size = intermediate_size
        # How many hidden layers the transformer has
        self.num_hidden_layers = num_hidden_layers
        # We are going to use Group Query Attention. So we need attention head num and num_key_value_heads
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        # How many dimension each head will work with
        self.head_dim = head_dim
        # rms
        self.rms_norm_eps = rms_norm_eps
        # rope theta is another parameter of the rotary positional encoding which is the base frequency
        self.rope_theta = rope_theta
        # Indicates if in the attention matrices we want the bias.
        self.attention_bias = attention_bias
        # dropout we are not going to use it.
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id





class PaliGemmaConfig():

    def __init__(self, 
                 vision_config = None, text_config = None, ignore_index = -100, image_token_index=256000, 
                 vocab_size=257152, projection_dim = 2048, hidden_size = 2048, pad_token_id=None, **kwargs,):
        
        super().__init__()

        self.ignore_index=ignore_index
        # Token corresponding to the placeholder image token: "<image>"
        self.image_token_index=image_token_index
        # Vocab size of the modal
        self.vocab_size=vocab_size
        # What is the final dimension that the image feautes should be resized to before feeding to the language model. Basically it's the output size of the linear projection (linear layer)
        self.projection_dim=projection_dim
        # Embedding size of the language modal.
        self.hidden_size=hidden_size
        self.vision_config=vision_config
        self.is_encoder_decoder=False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        # How many patches do you get per image = image tokens
        self.text_config.num_image_token = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim



class GemmaModel(nn.Module):

    def __init__(self,config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad.token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config,layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size,eps=config.rms.norm_eps)


    def get_input_embeddings(self):
        return self.embed_tokens


    def forward(
            self,
            attention_mask: Optional[torch.Tensor] = None,
            positions_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = inputs_embeds
        # [Batch_Size, Seq_Len, Hidden_Size]
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            # [Batch_Size, Seq_Len, Hidden_Size]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                positions_ids=positions_ids,
                kv_cache=kv_cache,
            )

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states= self.norm(hidden_states)

        # [Batch_Size, Seq_Len, Hidden_Size]
        return hidden_states


# Transformer model + linear layer
class GemmaForCasualLM(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight


    def forward(
            self,
            attention_mask: Optional[torch.Tensor] = None,
            positions_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        # input_embeds: [Batch_Size, Seq_Len, Hidden_Size]
        # outputs: [Batch_Size, Seq_Len, Hidden_Size]
        outputs = self.model(
            attention_mask=attention_mask,
            positions_ids=positions_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            # Return the updated cache
            return_data["kv_cache"] = kv_cache

        return return_data



# Just a linear layer that converts the size of the image features extracted from the vision encoder into the same size of the embedded size that is used by the language model.
class PaliGemmaMultiModalProjector(nn.Module):

    def __init__(self,config:PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batches_Size, Num_Patches, Projection_Dim]
        hidden_states = self.linear(image_features)
        return hidden_states





 # We are going to attend all of the text tokens and all of the image tokens without any casuality so we are going to use them like "conditions"
class PaliGemmaForConditionalGeneration(nn.Module):

    def __init__(self,config:PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)

        # Linear projection which is the linear layer that will convert the size of the embedding output by the vision encoder into the size of the embedding of each text token so that they can be concatenated with together.
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        # Transformer Decoder
        language_model = GemmaForCasualLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1


    # It's a techinc of sharing parameters of one layer with another.
    def tie_weights(self):
        return self.language_model.tie_weights() 



    def _merge_input_ids_with_image_features(
            self,
            image_features: torch.Tensor,
            inputs_embeds: torch.Tensor,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            kv_cache: Optional[KvCache] = None,
    ):
        _,_, embed_dim = image_features.shape
        batch_size,sequence_length=input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        # Shape: [Batch_Size, Seq_Len, Hidden_Size]
        scaled_image_features = image_features / (self.config.hidden_size **0.5)

        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens.
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        # Shape: [Batch_Size, Seq_Len] True for text tokens
        # [567,567,567, 1, 53,25,75,2]
        # [0,0,0,1,1,1,1,1]
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        # Shape: [Batch_Size, Seq_Len] True for image tokens
        # [1,1,1,0,0,0,0,0]
        image_mask = input_ids == self.config.image_token_index
        # Shape: [Batch_Size, Seq_Len] True for padding tokens
        # [0,0,0,0,0,0,0,0]
        pad_mask = input_ids == self.pad_token_id

        # We need to expand the mask to the embedding dimension otherwise we can't use them in torch.where 
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1,-1,embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1,-1,embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1,-1,embed_dim)

        # Add the text embeddings
        final_embedding = torch.where(text_mask_expanded, inputs_embeds,final_embedding)
        # Insert image embeddings. We can't use torch.where because the sequence length of scaled_image_features is not equal to the sequence length of the final embedding
        final_embedding = final_embedding.masked_scatter(image_mask_expanded,scaled_image_features)
        # Zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding),final_embedding)

        

        #### CREATE THE ATTENTION MASK ####

        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]
    
        if kv_cache is None or kv_cache.num_items() == 0:
            # Do not mask any token, because we're in the prefill phase
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Since we are generating tokens, the query must be one single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Also in this case we don't need to mask anything, since each query should be able to attend all previous tokens. 
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )


        # Add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position.
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids




    def forward(self,
                 input_ids: torch.LongTensor = None, pixel_values: torch.FloatTensor = None, attention_mask: Optional[torch.Tensor] = None, kv_cache: Optional[KVCache] = None,) -> Tuple:
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # 1.Extra the input embeddings
        # shape: (Batch_Size, Seq_len, Hidden_State)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 2. Merge text and images
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        selected_image_feature = self.vision_tower(pixel_values.to(input_embeds.dtype))

        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_Size]
        image_feautes = self.multi_modal_projector(selected_image_feature)

        # Merge the embeddings of the text tokens and the image tokens
        # image_features: extracted from the visual encoder
        # input_embeds: extracted from the language model which already contains the placeholders (for image tokens)
        # input_ids: which are the original input ids the tokens fed to the language model
        # attention_mask:
        # kv_cache: 
        input_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_feautes,input_embeds,input_ids,attention_mask,kv_cache)

        outputs = self.language_model(attention_mask=attention_mask,position_ids=position_ids,input_embeds=input_embeds,kv_cache=kv_cache)
        return outputs