import math
import torch
import torch.nn as nn

from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _load_weights
import torch.utils.checkpoint as checkpoint

from mmcv_custom import load_checkpoint
from mmengine.logging import MMLogger
# MMSegmentation 1.0.0+ uses registry system
try:
    from mmseg.registry import MODELS as BACKBONES
except ImportError:
    # Fallback for older versions
    from mmseg.models.builder import BACKBONES

# add the root path to the system path
import sys, os
# Get the project root by going up from current file: seg/backbone/vim.py -> project root
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
_vim_path = os.path.join(_project_root, 'vim')
if _vim_path not in sys.path:
    sys.path.insert(0, _vim_path)
from models_mamba import VisionMamba, layer_norm_fn, rms_norm_fn, RMSNorm
from utils import interpolate_pos_embed


@BACKBONES.register_module()
class VisionMambaSeg(VisionMamba):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=16,
        in_chans=3,
        num_classes=80,
        embed_dim=192,
        depth=24,
        use_checkpoint=False,
        pretrained=None,
        out_indices=[3, 5, 7, 11],
        if_fpn=True,
        use_residual_as_feature=False,
        last_layer_process="none",
        **kwargs
    ):

        # for rope
        ft_seq_len = img_size // patch_size
        kwargs['ft_seq_len'] = ft_seq_len

        super().__init__(img_size, patch_size, stride, depth, embed_dim, in_chans, num_classes, **kwargs)

        self.use_checkpoint = use_checkpoint
        self.out_indices = out_indices
        self.if_fpn = if_fpn
        self.use_residual_as_feature = use_residual_as_feature
        self.last_layer_process = last_layer_process

        # del the parent class's head
        del self.head

        if if_fpn:
            if patch_size == 16:
                self.fpn1 = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                    nn.SyncBatchNorm(embed_dim),
                    nn.GELU(),
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                )

                self.fpn2 = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                )

                self.fpn3 = nn.Identity()

                self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
            elif patch_size == 8:
                self.fpn1 = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                )

                self.fpn2 = nn.Identity()

                self.fpn3 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )

                self.fpn4 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=4, stride=4),
                )

        self.init_weights(pretrained)

        # drop the pos embed for class token
        if self.if_cls_token:
            del self.cls_token
            self.pos_embed = torch.nn.Parameter(self.pos_embed[:, 1:, :])
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = MMLogger.get_instance(name='mmseg')

            # load_checkpoint(self, pretrained, strict=False, logger=logger)

            state_dict = torch.load(pretrained, map_location="cpu")
            # import ipdb; ipdb.set_trace()
            state_dict_model = state_dict["model"]
            state_dict_model.pop("head.weight")
            state_dict_model.pop("head.bias")
            # pop rope
            try:
                state_dict_model.pop("rope.freqs_cos")
                state_dict_model.pop("rope.freqs_sin")
            except:
                print("no rope in the pretrained model")

            # Check patch size and input channel mismatches
            if "patch_embed.proj.weight" in state_dict_model:
                checkpoint_patch_size = state_dict_model["patch_embed.proj.weight"].shape[-1]
                checkpoint_in_chans = state_dict_model["patch_embed.proj.weight"].shape[1]
                model_patch_size = self.patch_embed.patch_size[-1]
                model_in_chans = self.patch_embed.proj.weight.shape[1]
                
                if checkpoint_patch_size != model_patch_size:
                    logger.info(f"Skipping patch_embed.proj due to patch size mismatch: "
                               f"checkpoint has {checkpoint_patch_size}, model expects {model_patch_size}")
                    state_dict_model.pop("patch_embed.proj.weight", None)
                    state_dict_model.pop("patch_embed.proj.bias", None)
                elif checkpoint_in_chans != model_in_chans:
                    logger.info(f"Skipping patch_embed.proj due to input channel mismatch: "
                               f"checkpoint has {checkpoint_in_chans} channels, "
                               f"model expects {model_in_chans} channels")
                    state_dict_model.pop("patch_embed.proj.weight", None)
                    state_dict_model.pop("patch_embed.proj.bias", None)
            
            # Filter out mixer layer parameters with size mismatches
            # These depend on d_state and dt_rank which may differ between checkpoint and model
            # Get the model's state dict to compare shapes
            model_state_dict = self.state_dict()
            keys_to_remove = []
            
            for key in list(state_dict_model.keys()):
                # Check if this is a mixer parameter that might have size mismatches
                if any(mixer_param in key for mixer_param in ['mixer.A_log', 'mixer.A_b_log', 
                                                              'mixer.x_proj.weight', 'mixer.x_proj_b.weight']):
                    # Compare shapes if the key exists in the model
                    if key in model_state_dict:
                        checkpoint_shape = state_dict_model[key].shape
                        model_shape = model_state_dict[key].shape
                        
                        if checkpoint_shape != model_shape:
                            keys_to_remove.append(key)
                            logger.info(f"Skipping {key} due to size mismatch: "
                                       f"checkpoint {checkpoint_shape} vs model {model_shape}")
            
            # Remove mismatched parameters
            for key in keys_to_remove:
                state_dict_model.pop(key, None)
            
            interpolate_pos_embed(self, state_dict_model)

            res = self.load_state_dict(state_dict_model, strict=False) 
            logger.info(res)
            print(res)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')
        
    def get_num_layers(self):
        return len(self.layers)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward_plain_features(self, x, inference_params=None):
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        residual = None
        hidden_states = x
        features = []

        # todo: configure this
        # get two layers in a single for-loop
        for i, layer in enumerate(self.layers):

            # rope about
            if self.if_rope:
                hidden_states = self.rope(hidden_states)
                if residual is not None and self.if_rope_residual:
                    residual = self.rope(residual)

            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

        return residual
    
    def forward_features(self, x, inference_params=None):
        # MMSeg 1.x passes inputs as a list of tensors
        if isinstance(x, list):
            x = x[0]
        B, C, H, W = x.shape
        # x, (Hp, Wp) = self.patch_embed(x)
        x = self.patch_embed(x)

        batch_size, seq_len, _ = x.size()
        Hp = Wp = int(math.sqrt(seq_len))

        if self.pos_embed is not None:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        residual = None
        hidden_states = x
        features = []
        if not self.if_bidirectional:
            for i, layer in enumerate(self.layers):

                # rope about
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )

                if self.use_residual_as_feature:
                    if i-1 in self.out_indices:
                        # residual_p = residual.permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                        # features.append(residual_p.contiguous())
                        features.append(residual.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous())
                else:
                    if i in self.out_indices:
                        # hidden_states_p = hidden_states.permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                        # features.append(hidden_states_p.contiguous()) 
                        features.append(hidden_states.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous())

        else:
            # todo: configure this
            # get two layers in a single for-loop
            for i in range(len(self.layers) // 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states_f, residual_f = self.layers[i * 2](
                    hidden_states, residual, inference_params=inference_params
                )
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]), None if residual == None else residual.flip([1]), inference_params=inference_params
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        if self.last_layer_process == 'none':
            residual = hidden_states
        elif self.last_layer_process == 'add':
            residual = hidden_states + residual
        elif self.last_layer_process == 'add & norm':
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            residual = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        if self.use_residual_as_feature and self.out_indices[-1] == len(self.layers)-1:
            # residual_p = residual.permute(0, 2, 1).reshape(B, -1, Hp, Wp)
            # features.append(residual_p.contiguous())
            features.append(residual.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous())

        if self.if_fpn:
            ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]

            if len(features) == 1:
                for i in range(len(ops) - 1):
                    features.append(features[0])
                for i in range(len(features)):
                    features[i] = ops[i](features[i])
            else:
                for i in range(len(features)):
                    features[i] = ops[i](features[i])

        return tuple(features)

    def forward(self, x):
        x = self.forward_features(x)
        return x