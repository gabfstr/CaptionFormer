"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
# Modified by Gabriel Fiastre for CaptionFormer (https://github.com/gabfstr/CaptionFormer)
#   from https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_opt.py
import json
import logging
import os
from packaging import version
import contextlib

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

import torch.nn.functional as F
from math import sqrt

from detectron2.config import configurable
from detectron2.utils.registry import Registry

from .Qformer import BertLMHeadModel

from transformers import BertTokenizer, AutoTokenizer, OPTForCausalLM, OPTConfig
from transformers.models.bert.configuration_bert import BertConfig
import transformers

from captionformer_video.modeling.transformer_decoder.video_mask2former_transformer_decoder import SelfAttentionLayer, CrossAttentionLayer, FFNLayer

from .eva_vit import create_eva_vit_g

from utils.box_ops import attn_masking_from_bbox



def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


CAPTIONING_HEAD_REGISTRY = Registry("CAPTIONING_MODULE")
CAPTIONING_HEAD_REGISTRY.__doc__ = """
Registry for captioning head module in MaskFormer.
"""



def build_video_captioning_head(cfg, input_channels):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.MASK_FORMER.CAPTIONING_HEAD_NAME
    return CAPTIONING_HEAD_REGISTRY.get(name)(cfg, input_channels)


class LVVISFeatureLoader():
    """
    A trivial loader for frozen vision features.
    """
    def __init__(self, feature_mapping, feature_batch_size=64, dataset="lvvis"):
        assert dataset in ["lvvis", "vidstg", "bensmot", "vln_uvo_sparse"], "Dataset not supported"
        # embed dim for use in the model
        self.num_features = 1408
        with open(feature_mapping, "r") as f:
            self.mapping = json.load(f)

        self.feature_batch_size = feature_batch_size

        if dataset == "vidstg":
            self.move_index=True
        else:
            self.move_index=False

    def __call__(self, feat_ids):
        features = []
        for ft_id in feat_ids:
            video_id = ft_id["video_id"]
            index = ft_id["frame_index"]

            if self.move_index:
                index = index - 1
            try:
                mapping = self.mapping[str(video_id).zfill(5)][str(index)]
            except KeyError :
                print("ft_id['frame_index']: ", ft_id["frame_index"])
                print("index: ", index)
                print("self.mapping of index {} : {}".format(str(video_id).zfill(5), self.mapping[str(video_id).zfill(5)]))
                raise KeyError("Index not found in mapping")
            file_name = mapping["file_name"]
            batch_id = mapping["index"]

            try :
                features.append(torch.load(file_name)[batch_id])
            except IndexError:
                print("file_name: ", file_name)
                print("batch_id: ", batch_id)
                print("index: ", index)
                x=torch.load(file_name)
                print("x shape: ", x.shape)
                raise IndexError("Index out of range")
            except Exception as e:
                print("Error loading file: ", file_name)
                print("batch_id: ", batch_id)
                print("index: ", index)
                print("Error: ", e)
                raise e
        
        return torch.stack(features, dim=0)

# @registry.register_model("blip2_opt")
@CAPTIONING_HEAD_REGISTRY.register()
class Blip2OPTVideo(nn.Module):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",
        "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",
        "caption_coco_opt2.7b": "configs/models/blip2/blip2_caption_opt2.7b.yaml",
        "caption_coco_opt6.7b": "configs/models/blip2/blip2_caption_opt6.7b.yaml",
    }
    
    @configurable
    def __init__(
        self,
        input_channels,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        opt_model="facebook/opt-2.7b",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
        box_mode=False,
        saved_features=False,
        feature_mapper=None,
        feature_aggregation=False,
        aggregation_method="concat",
        opt_proj_before_agg=False,
        aggregation_num_t=4,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.27"), "BLIP-2 OPT requires transformers>=4.27"
        

        self.tokenizer = self.init_tokenizer()
        
        self.img_size = img_size

        self.box_mode = box_mode

        self.saved_features = saved_features

        if feature_mapper == "lvvis":
            self.feature_mapper = "lvvis"
            self.feature_mapping = "./datasets/LVVIS/train/features/index_mapping.json"
        elif feature_mapper == "vidstg":
            self.feature_mapper = "vidstg"
            self.feature_mapping = "./datasets/VidSTG/features/index_mapping.json"
        elif feature_mapper == "bensmot":
            self.feature_mapper = "bensmot"
            self.feature_mapping = "./datasets/bensmot/features/index_mapping.json"
        elif feature_mapper == "vln_uvo_sparse":
            self.feature_mapper = "vln_uvo_sparse"
            self.feature_mapping = "./datasets/VLN/features/index_mapping.json"
        else :
            raise ValueError("Feature mapping not found")
        # Take swin features as input
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )

        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.obj_linear_proj, self.query_tokens = self.init_Qformer(
            input_channels, num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        print("Loading OPT model")
        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(opt_model, torch_dtype=torch.float16)
        print(f"Loaded OPT from {opt_model}")
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        self.opt_model = self.opt_model.eval()
        self.opt_model.train = disabled_train

        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )

        self.feature_aggregation = feature_aggregation
        if self.feature_aggregation:
            self.opt_proj_before_agg = opt_proj_before_agg
        self.aggregation_method = aggregation_method
        assert self.aggregation_method in ["concat", "mean", "weighted mean", "arithmetic mean", "linear", "self-attention", "self-attention+linear", "self-attention+weighted", "self-attention+scores"], "Aggregation method must be either 'concat' or 'mean' or 'weighted mean' or 'arithmetic mean' or 'linear' or 'self-attention' or 'self-attention+linear' or 'self-attention+weighted' or 'self-attention+scores'"

        self.aggregation_num_t = aggregation_num_t
        if self.aggregation_method == "linear":
            if self.opt_proj_before_agg:
                self.agg_linear = nn.Linear(self.aggregation_num_t * self.opt_model.config.hidden_size, self.opt_model.config.hidden_size)
            else:
                self.agg_linear = nn.Linear(self.aggregation_num_t * self.Qformer.config.hidden_size, self.Qformer.config.hidden_size)
        elif "self-attention" in self.aggregation_method:
            if self.opt_proj_before_agg:
                sa_hidden_size = self.opt_model.config.hidden_size
            else:
                sa_hidden_size = self.Qformer.config.hidden_size
            if self.aggregation_method == "self-attention+scores":
                sa_hidden_size += 1
                raise NotImplementedError("self-attention+scores not implemented yet")

            self.agg_self_attention = SelfAttentionLayer(
                d_model=sa_hidden_size,
                nhead=8,
                dropout=0.0
            )
            if self.aggregation_method != "self-attention+weighted":
                self.agg_linear = nn.Linear(sa_hidden_size * self.aggregation_num_t, sa_hidden_size)
        
        elif self.aggregation_method == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.opt_model.config.hidden_size, nhead=8)
            self.agg_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.agg_pos_embedding = nn.Parameter(torch.zeros(1, self.aggregation_num_t, self.opt_model.config.hidden_size))
            nn.init.trunc_normal_(self.agg_pos_embedding, std=0.02)


        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)
        
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None    


        #Load weights using "huggin face from_pretrained" method
        
   
    @property
    def device(self):
        return list(self.parameters())[0].device
    
    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer
    
    def init_vision_encoder(
        self, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision
    ):
        if self.saved_features:
            visual_encoder = LVVISFeatureLoader(self.feature_mapping, dataset=self.feature_mapper)
        else :
            assert model_name in [
                "eva_clip_g",
                "eva2_clip_L",
                "clip_L",
            ], "vit model must be eva_clip_g, eva2_clip_L or clip_L"
            print("Intializing vision encoder")
            if model_name == "eva_clip_g":
                visual_encoder = create_eva_vit_g(
                    img_size, drop_path_rate, use_grad_checkpoint, precision
                )
    #         elif model_name == "eva2_clip_L":
    #             visual_encoder = create_eva2_vit_L(
    #                 img_size, drop_path_rate, use_grad_checkpoint, precision
    #             )
            elif model_name == "clip_L":
                raise NotImplementedError("clip_L is not supported yet")
                visual_encoder = create_clip_vit_L(img_size, use_grad_checkpoint, precision)
        ln_vision = LayerNorm(visual_encoder.num_features)
        self.vit_name = model_name
        return visual_encoder, ln_vision

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    
    @classmethod
    def init_Qformer(cls, input_channels, num_text_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_text_query_token + 1
        print("Initializing Qformer")
        Qformer = BertLMHeadModel.from_pretrained("bert-base-uncased", config=encoder_config)
        # query_tokens = nn.Parameter(
        #     torch.zeros(1, num_query_token, encoder_config.hidden_size)
        # )
        # query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        
        obj_query_linear = nn.Linear(input_channels, encoder_config.hidden_size)
        text_query_tokens = nn.Parameter(
            torch.zeros(1, num_text_query_token, encoder_config.hidden_size)
        )

        return Qformer, obj_query_linear, text_query_tokens



    @classmethod
    def from_config(cls, cfg, input_channels):
        # vit_model = cfg.get("vit_model", "eva_clip_g")
        # img_size = cfg.get("image_size")
        
        # num_query_token = cfg.get("num_query_token")
        # opt_model = cfg.get("opt_model")

        # drop_path_rate = cfg.get("drop_path_rate", 0)
        # use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        # vit_precision = cfg.get("vit_precision", "fp16")
        # freeze_vit = cfg.get("freeze_vit", True)

        # prompt = cfg.get("prompt", "")
        # max_txt_len = cfg.get("max_txt_len", 32)
        
        # apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        ret = {}
        ret["input_channels"] = input_channels

        ret["vit_model"] = cfg.MODEL.CAPTIONING_HEAD.VIT_MODEL_NAME
        ret["img_size"] = cfg.MODEL.CAPTIONING_HEAD.IMG_SIZE
        ret["num_query_token"] = cfg.MODEL.CAPTIONING_HEAD.NUM_TEXT_QUERIES
        ret["opt_model"] = cfg.MODEL.CAPTIONING_HEAD.OPT_MODEL or "facebook/opt-2.7b"
        ret["drop_path_rate"] = cfg.MODEL.CAPTIONING_HEAD.DROP_PATH_RATE
        ret["use_grad_checkpoint"] = cfg.MODEL.CAPTIONING_HEAD.USE_GRAD_CHECKPOINT
        ret["vit_precision"] = cfg.MODEL.CAPTIONING_HEAD.VIT_PRECISION
        ret["freeze_vit"] = cfg.MODEL.CAPTIONING_HEAD.FREEZE_VIT
        ret["prompt"] = cfg.MODEL.CAPTIONING_HEAD.PROMPT
        ret["max_txt_len"] = cfg.MODEL.CAPTIONING_HEAD.MAX_TXT_LEN
        ret["apply_lemmatizer"] = cfg.MODEL.CAPTIONING_HEAD.APPLY_LEMMATIZER
        ret["box_mode"] = cfg.MODEL.MASK_FORMER.BOX_MODE_ON
        ret["saved_features"] = cfg.MODEL.CAPTIONING_HEAD.SAVED_FEATURES
        ret["feature_mapper"] = cfg.MODEL.CAPTIONING_HEAD.FEATURE_MAPPING
        ret["feature_aggregation"] = cfg.MODEL.MASK_FORMER.MULTI_FRAME_CAPTIONING
        ret["aggregation_method"] = cfg.MODEL.CAPTIONING_HEAD.AGGREGATION_METHOD
        ret["opt_proj_before_agg"] = cfg.MODEL.CAPTIONING_HEAD.OPT_PROJ_BEFORE_AGGREGATION
        ret["aggregation_num_t"] = cfg.MODEL.CAPTIONING_HEAD.AGGREGATION_NUM_T

        return ret


    def forward(self, samples, object_query_tokens, object_pred_masks_or_boxes, object_scores, indices):
       
        if self.train :

            if self.feature_aggregation:
                return self.train_video(samples, object_query_tokens, object_pred_masks_or_boxes, object_scores, indices)

            og_image = samples["image"]
            
            image = F.interpolate(og_image, size=self.img_size, mode='bilinear', align_corners=False)

            # Separate image from video 
            bs, num_queries, _ = object_query_tokens.shape
            image = image.view(bs, -1, image.shape[-3], image.shape[-2], image.shape[-1])
            nframes = image.shape[1]

            #Sample a frame from each clip
            idx_sample = torch.randint(0, nframes, (bs,))
            image = image[torch.arange(bs), idx_sample]

            if "feature_id" in samples and self.saved_features==True:
                feature_id = samples["feature_id"]
                for x,idselect in zip(feature_id, idx_sample):
                    # sample frame
                    x["frame_index"] = x["frame_index"][idselect.item()]

            object_pred_masks_or_boxes = object_pred_masks_or_boxes.transpose(1,2)
            object_pred_masks_or_boxes = object_pred_masks_or_boxes[torch.arange(bs), idx_sample]
            

            src_idx = self._get_src_permutation_idx(indices)
            tgt_idx = self._get_tgt_permutation_idx(indices)
            
            src_query_tokens = object_query_tokens[src_idx].unsqueeze(1)
            src_pred_masks_or_boxes = object_pred_masks_or_boxes[src_idx].unsqueeze(1)
            
            target_labels = [t["captions"] for t in samples["target"]]

            loss_sum = torch.tensor(0.0, device=self.device, requires_grad=True)

            if len(tgt_idx[0]) == 0:
                return loss_sum * 0.0 + sum(p.sum() * 0.0 for p in self.parameters() if p.requires_grad)
            
            if "feature_id" in samples and self.saved_features==True:
                    feature_id = samples["feature_id"]
                    image_embeds = self.ln_vision(self.visual_encoder(feature_id).to(self.device))
            else :
                with self.maybe_autocast():
                    image_embeds = self.ln_vision(self.visual_encoder(image))

            attn_mask_target_size = int(sqrt(image_embeds.shape[-2]))
            if self.box_mode:
                attn_mask = attn_masking_from_bbox(src_pred_masks_or_boxes, (attn_mask_target_size, attn_mask_target_size), num_heads=1)
                
            else : 
                attn_mask = F.interpolate(src_pred_masks_or_boxes, size=attn_mask_target_size, mode="bilinear", align_corners=False)
                # must use bool type
                # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
                attn_mask = (attn_mask.sigmoid().flatten(-2,-1) < 0.5).bool()
                attn_mask = attn_mask.detach()
            
            # Add 1 dim for the cls token at the end (don't mask it)
            cls_attn = torch.ones(attn_mask.size(0), attn_mask.size(1), 1, dtype=torch.bool).to(attn_mask.device)
            attn_mask = torch.cat([attn_mask, cls_attn], dim=2)

            # Project object queries
            src_query_tokens = self.obj_linear_proj(src_query_tokens).unsqueeze(1)
            
            num_gt = 0
            for img_idx, cap_idx, obj_query, obj_attn_mask in zip(tgt_idx[0], tgt_idx[1], src_query_tokens, attn_mask):
                
                
                # Get text caption
                text = target_labels[img_idx][cap_idx]
                if text==None or text=="":
                    continue
                text = text + "\n"

                num_gt += 1

                #Cat with text tokens
                text_tokens = self.query_tokens.expand(obj_query.shape[0], -1, -1)
                query_tokens = torch.cat([obj_query, text_tokens], dim=1)
                
               
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds[img_idx].unsqueeze(0),
                    encoder_attention_mask=obj_attn_mask.unsqueeze(1),
                    return_dict=True,
                )

                # Opt input proj
                inputs_opt = self.opt_proj(query_output.last_hidden_state)
                atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(attn_mask.device)

                self.opt_tokenizer.padding_side = "right"                

                # # Ensure the EOS token is added to the text
                # text += self.opt_tokenizer.eos_token
                
                opt_tokens = self.opt_tokenizer(
                    text,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                ).to(attn_mask.device)
                
                
                targets = opt_tokens.input_ids.masked_fill(
                    opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
                )
                if self.prompt:
                    targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt
                empty_targets = (
                    torch.ones(atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100)
                )
                targets = torch.cat([empty_targets, targets], dim=1)

                inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
                inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
               
               
                with self.maybe_autocast():
                    outputs = self.opt_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        return_dict=True,
                        labels=targets,
                    )
                loss = outputs.loss
                loss_sum = loss_sum + loss

                del text_tokens, query_tokens, query_output, inputs_opt, atts_opt, opt_tokens, targets, empty_targets, inputs_embeds, attention_mask, outputs, loss
                torch.cuda.empty_cache()
            
            if num_gt == 0:
                return loss_sum * 0.0 + sum(p.sum() * 0.0 for p in self.parameters() if p.requires_grad)

            loss_reduced = loss_sum / num_gt

            return loss_reduced
        else:
            if not self.feature_aggregation:
                return self.generate(self, samples, object_query_tokens, object_pred_masks_or_boxes)
            else:
                return self.generate_video(self, samples, object_query_tokens, object_pred_masks_or_boxes)



    def train_video(self, samples_list, object_query_tokens_list, object_pred_masks_or_boxes_list, object_scores_list, indices):

        num_instances = len(indices[0][0])
        loss_sum = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        src_idx_permutation = self._get_src_permutation_idx(indices)
        tgt_idx_permutation = self._get_tgt_permutation_idx(indices)
        if len(tgt_idx_permutation[0]) == 0:
            return loss_sum * 0.0 + sum(p.sum() * 0.0 for p in self.parameters() if p.requires_grad)

        
        num_gts = 0 
        for instance_idx in range(num_instances):
            img_idx = tgt_idx_permutation[0][instance_idx]
            tgt_idx = tgt_idx_permutation[1][instance_idx]
            

            text = ''
            input_opt_list = []
            input_opt_scores_list = []
            # Iterate over clips to aggregate
            for samples, object_query_tokens, object_pred_masks_or_boxes, object_scores in zip(samples_list, object_query_tokens_list, object_pred_masks_or_boxes_list, object_scores_list):

                og_image = samples["image"]
            
                image = F.interpolate(og_image, size=self.img_size, mode='bilinear', align_corners=False)

                # Separate image from video 
                bs, num_queries, _ = object_query_tokens.shape
                image = image.view(bs, -1, image.shape[-3], image.shape[-2], image.shape[-1])
                nframes = image.shape[1]

                #Sample a frame from each clip
                idx_sample = torch.randint(0, nframes, (bs,))
                image = image[torch.arange(bs), idx_sample]

                if "feature_id" in samples and self.saved_features==True:
                    feature_id = samples["feature_id"]
                    
                    for x,idselect in zip(feature_id, idx_sample):
                        # sample frame
                        if isinstance(x["frame_index"], list):
                            x["frame_index"] = x["frame_index"][idselect.item()]

                object_pred_masks_or_boxes = object_pred_masks_or_boxes.transpose(1,2)
                object_pred_masks_or_boxes = object_pred_masks_or_boxes[torch.arange(bs), idx_sample]


                src_query_tokens = object_query_tokens[src_idx_permutation].unsqueeze(1)
                src_pred_masks_or_boxes = object_pred_masks_or_boxes[src_idx_permutation].unsqueeze(1)
                
                src_object_scores = object_scores[src_idx_permutation[1]]
                
                if "feature_id" in samples and self.saved_features==True:
                        feature_id = samples["feature_id"]
                        image_embeds = self.ln_vision(self.visual_encoder(feature_id).to(self.device))
                else :
                    with self.maybe_autocast():
                        image_embeds = self.ln_vision(self.visual_encoder(image))

                
                attn_mask_target_size = int(sqrt(image_embeds.shape[-2]))
                if self.box_mode:
                    attn_mask = attn_masking_from_bbox(src_pred_masks_or_boxes, (attn_mask_target_size, attn_mask_target_size), num_heads=1)
                    
                else : 
                    attn_mask = F.interpolate(src_pred_masks_or_boxes, size=attn_mask_target_size, mode="bilinear", align_corners=False)
                    # must use bool type
                    # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
                    attn_mask = (attn_mask.sigmoid().flatten(-2,-1) < 0.5).bool()
                    attn_mask = attn_mask.detach()
                
                # Add 1 dim for the cls token at the end (don't mask it)
                cls_attn = torch.ones(attn_mask.size(0), attn_mask.size(1), 1, dtype=torch.bool).to(attn_mask.device)
                attn_mask = torch.cat([attn_mask, cls_attn], dim=2)
                

                target_labels = [t["captions"] for t in samples["target"]]
                
                if text=='' or text=='\n':
                    text = target_labels[img_idx][tgt_idx]
                    if text==None or text=="" or text=="\n":
                        text = ''
                    text = text + "\n"
                
                
                # Project object queries
                src_query_tokens = self.obj_linear_proj(src_query_tokens).unsqueeze(1)

                obj_query = src_query_tokens[instance_idx]
                obj_attn_mask = attn_mask[instance_idx]
                obj_scores = src_object_scores[instance_idx]
                

                #Cat with text tokens
                text_tokens = self.query_tokens.expand(obj_query.shape[0], -1, -1)
                query_tokens = torch.cat([obj_query, text_tokens], dim=1)
                

                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds[img_idx].unsqueeze(0),
                    encoder_attention_mask=obj_attn_mask.unsqueeze(1),
                    return_dict=True,
                )

                # Opt input proj
                if self.opt_proj_before_agg:
                    inputs_opt = self.opt_proj(query_output.last_hidden_state)
                else:
                    inputs_opt = query_output.last_hidden_state

                input_opt_list.append(inputs_opt)
                input_opt_scores_list.append(obj_scores)


            if text == '' or text == '\n':
                continue
            num_gts += 1

            if self.aggregation_method == "concat":
                inputs_opt = torch.cat(input_opt_list, dim=1)
            elif self.aggregation_method == "mean":
                inputs_opt = torch.mean(torch.stack(input_opt_list, dim=0), dim=0)
            elif self.aggregation_method == "weighted mean":
                stacked_input_opt = torch.stack(input_opt_list, dim=0)  # shape num_t, bs, nq, ndim
                stacked_input_opt_scores = torch.stack(input_opt_scores_list, dim=0).unsqueeze(-1)  # shape num_t, 1
            
                # Normalize scores with sum over sum
                normed_scores = stacked_input_opt_scores / (torch.sum(stacked_input_opt_scores, dim=0, keepdim=True) + 1e-6) # shape num_t, 1
                inputs_opt = torch.sum(stacked_input_opt * normed_scores.unsqueeze(-1).unsqueeze(-1), dim=0)
                
            elif self.aggregation_method == "arithmetic mean":
                stacked_input_opt = torch.stack(input_opt_list, dim=0)  # shape num_t, bs, nq, ndim
                inputs_opt = torch.mean(stacked_input_opt, dim=0)
            elif self.aggregation_method == "linear":
                inputs_opt = torch.stack(input_opt_list, dim=0)  # shape num_t, bs, nq, ndim
                inputs_opt = inputs_opt.permute(1,2,0,3).flatten(-2,-1)  # bs, nq, num_t*ndim
                inputs_opt = self.agg_linear(inputs_opt)
            elif "self-attention" in self.aggregation_method:
                inputs_opt = torch.stack(input_opt_list, dim=0)  # shape num_t, bs, nq, ndim
                # Get scores
                stacked_input_opt_scores = torch.stack(input_opt_scores_list, dim=0).unsqueeze(-1).unsqueeze(-1)  # shape num_t, 1, 1
                normed_scores = stacked_input_opt_scores / (torch.sum(stacked_input_opt_scores, dim=0, keepdim=True) + 1e-6) # shape num_t, 1, 1

                num_t, bs, nq, hidden = inputs_opt.shape

                if self.aggregation_method == 'self-attention+scores':
                    print("Using self-attention with scores aggregation")
                    print("shape before adding scores dim: ", inputs_opt.shape)
                    print("normed_scores shape: ", normed_scores.shape)
                    # Stack score as an additional dimension
                    inputs_opt = torch.cat([inputs_opt, normed_scores.expand(-1, inputs_opt.size(1), inputs_opt.size(2), -1)], dim=-1)
                    print("inputs_opt shape after adding scores dim: ", inputs_opt.shape)
                    
                    sa_input = inputs_opt.permute(2, 1, 0, 3).reshape(num_t, bs*nq, hidden+1)
                    print("sa_input shape for self-attn+scores-linear: ", sa_input.shape)
                    sa_output = self.agg_self_attention(sa_input)
                    inputs_opt = sa_output.reshape(num_t, bs, nq, hidden+1).permute(1, 2, 0, 3)
                    print("inputs_opt shape after self-attn+scores-linear: ", inputs_opt.shape)
                    raise ValueError("Stop here")

                else:
                    # Permute to (seq_len, batch, hidden) for MHA
                    sa_input = inputs_opt.permute(2, 1, 0, 3).reshape(num_t, bs*nq, hidden)
                    sa_output = self.agg_self_attention(sa_input)
                    
                    inputs_opt = sa_output.reshape(num_t, bs, nq, hidden).permute(1, 2, 0, 3)  # bs, nq, num_t, hidden

                if self.aggregation_method == 'self-attention+weighted':
                    
                    # Weighted mean over time dimension (num_t)
                    inputs_opt = torch.sum(inputs_opt * normed_scores.permute(1,2,0).unsqueeze(-1), dim=2)  # bs, nq, hidden

                else :
                    # Flatten time for linear
                    inputs_opt = inputs_opt.flatten(2,3)  # bs, nq, num_t*hidden
                    inputs_opt = self.agg_linear(inputs_opt)

            else:
                raise ValueError("Aggregation method must be either 'concat' or 'mean'")

            if not self.opt_proj_before_agg:
                inputs_opt = self.opt_proj(inputs_opt)

            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(attn_mask.device)

            self.opt_tokenizer.padding_side = "right"                

            # # Ensure the EOS token is added to the text
            # text += self.opt_tokenizer.eos_token
            opt_tokens = self.opt_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(attn_mask.device)
            
           
            targets = opt_tokens.input_ids.masked_fill(
                opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
            )
            if self.prompt:
                targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

            empty_targets = (
                torch.ones(atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100)
            )
            targets = torch.cat([empty_targets, targets], dim=1)

            inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            
            with self.maybe_autocast():
                outputs = self.opt_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            loss = outputs.loss
            loss_sum = loss_sum + loss

            del text_tokens, query_tokens, query_output, inputs_opt, atts_opt, opt_tokens, targets, empty_targets, inputs_embeds, attention_mask, outputs, loss
        
        if num_gts == 0:
            return loss_sum * 0.0 + sum(p.sum() * 0.0 for p in self.parameters() if p.requires_grad)
        
        elif not(torch.isfinite(loss_sum)):
            print("Loss is not finite, returning 0")
            raise ValueError("Loss is not finite, returning 0")
            return torch.tensor(0.0, device=self.device)
        else:
            reduced_loss = loss_sum / num_gts

            return reduced_loss
          

    @torch.no_grad()
    def generate(
        self,
        samples,
        object_query_tokens,
        object_pred_masks_or_boxes,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            object_query_tokens (torch.Tensor): A tensor of shape (batch_size, num_queries, hidden_size) containing the object query tokens used for object-centric captioning.
            object_pred_masks_or_boxes (torch.Tensor): A tensor of shape (batch_size, num_queries, H, W) containing the predicted masks or boxes for attending to the objects.
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        og_image = samples["image"]

        image = F.interpolate(og_image, size=self.img_size, mode='bilinear', align_corners=False)

        # Separate image from video 
        bs, num_queries, _ = object_query_tokens.shape
        image = image.view(bs, -1, image.shape[-3], image.shape[-2], image.shape[-1])
        nframes = image.shape[1]
        
        if nframes > 1:
            #Sample a frame from each clip
            idx_sample = torch.randint(0, nframes, (bs,))
            image = image[torch.arange(bs), idx_sample]
            object_pred_masks_or_boxes = object_pred_masks_or_boxes.transpose(1,2)
            object_pred_masks_or_boxes = object_pred_masks_or_boxes[torch.arange(bs), idx_sample]
        else :
            image = image.squeeze(1)
            idx_sample = torch.zeros(bs, dtype=torch.long, device=image.device)
            object_pred_masks_or_boxes = object_pred_masks_or_boxes.transpose(1,2)
            object_pred_masks_or_boxes = object_pred_masks_or_boxes.squeeze(1)
            

        if "feature_id" in samples and self.saved_features:
                feature_id = samples["feature_id"]

                for x,idselect in zip(feature_id, idx_sample):
                    # sample frame
                    x["frame_index"] = x["frame_index"][idselect.item()]
                image_embeds = self.ln_vision(self.visual_encoder(feature_id).to(self.device))

        else :
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))

        attn_mask_target_size = int(sqrt(image_embeds.shape[-2]))
        if self.box_mode:
            attn_mask = attn_masking_from_bbox(object_pred_masks_or_boxes, (attn_mask_target_size, attn_mask_target_size), num_heads=1)
            
        else : 
            attn_mask = F.interpolate(object_pred_masks_or_boxes, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(-2,-1) < 0.5).bool()
            attn_mask = attn_mask.detach()
        
        # Add 1 dim for the cls token at the end (don't mask it)
        cls_attn = torch.ones(attn_mask.size(0), attn_mask.size(1), 1, dtype=torch.bool).to(attn_mask.device)
        attn_mask = torch.cat([attn_mask, cls_attn], dim=2)

        text = [t + "\n" for t in samples["text_input"]]

        # Project object queries
        proj_object_query = self.obj_linear_proj(object_query_tokens)#.unsqueeze(1)
        
        output_text_list = [] 
        for obj_query, object_attn_mask in zip(proj_object_query.transpose(0,1), attn_mask.transpose(0,1)):

            text_tokens = self.query_tokens.expand(obj_query.shape[0], -1, -1)
            query_tokens = torch.cat([obj_query.unsqueeze(1), text_tokens], dim=1)
            
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=object_attn_mask,
                return_dict=True,
            )
            
            # Project object queries
            inputs_opt = self.opt_proj(query_output.last_hidden_state)

            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            prompt = [prompt] * image.size(0)

            opt_tokens = self.opt_tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            
            # new version for transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt,inputs_embeds],dim=1)
            
            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds, 
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                # max_new_tokens=max_length,  # <--- use max_new_tokens
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            
            output_text = [text.strip() for text in output_text]
            output_text_list.append(output_text)

        return output_text_list
    
    
    @torch.no_grad()
    def generate_video(
        self,
        samples_list,
        object_query_tokens_list,
        object_pred_masks_or_boxes_list,
        object_scores_list,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (list of dict): A list of dictionaries containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            object_query_tokens (list of torch.Tensor): A tensor of shape (batch_size, num_queries, hidden_size) containing the object query tokens used for object-centric captioning.
            object_pred_masks_or_boxes (list of torch.Tensor): A list of tensor of shape (batch_size, num_queries, H, W) containing the predicted masks or boxes for attending to the objects.
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """

        input_opt_list = []
        input_opt_scores_list = []
        output_text_list = []
        
        for sample, object_query_token, object_pred_mask_or_box, object_scores in zip(samples_list, object_query_tokens_list, object_pred_masks_or_boxes_list, object_scores_list):
            
            og_image = sample["image"]
            image = F.interpolate(og_image, size=self.img_size, mode='bilinear', align_corners=False)

            # Separate image from video 
            bs, num_queries, _ = object_query_token.shape
            image = image.view(bs, -1, image.shape[-3], image.shape[-2], image.shape[-1])
            nframes = image.shape[1]

            if nframes > 1:
                #Sample a frame from each clip
                idx_sample = torch.randint(0, nframes, (bs,))
                image = image[torch.arange(bs), idx_sample]

                object_pred_mask_or_box = object_pred_mask_or_box.transpose(1,2)
                object_pred_mask_or_box = object_pred_mask_or_box[torch.arange(bs), idx_sample]
            else :
                image = image.squeeze(1)
                object_pred_mask_or_box = object_pred_mask_or_box.transpose(1,2)
                object_pred_mask_or_box = object_pred_mask_or_box.squeeze(1)
                

            if "feature_id" in sample and self.saved_features:
                    feature_id = sample["feature_id"]
                    for x,idselect in zip(feature_id, idx_sample):
                        # sample frame
                        x["frame_index"] = x["frame_index"][idselect.item()]
                    image_embeds = self.ln_vision(self.visual_encoder(feature_id).to(self.device))
            else :
                with self.maybe_autocast():
                    image_embeds = self.ln_vision(self.visual_encoder(image))

            attn_mask_target_size = int(sqrt(image_embeds.shape[-2]))
            if self.box_mode:
                attn_mask = attn_masking_from_bbox(object_pred_mask_or_box, (attn_mask_target_size, attn_mask_target_size), num_heads=1)
                
            else : 
                attn_mask = F.interpolate(object_pred_mask_or_box, size=attn_mask_target_size, mode="bilinear", align_corners=False)
                # must use bool type
                # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
                attn_mask = (attn_mask.sigmoid().flatten(-2,-1) < 0.5).bool()
                attn_mask = attn_mask.detach()
            
            # Add 1 dim for the cls token at the end (don't mask it)
            cls_attn = torch.ones(attn_mask.size(0), attn_mask.size(1), 1, dtype=torch.bool).to(attn_mask.device)
            attn_mask = torch.cat([attn_mask, cls_attn], dim=2)

            if "prompt" in sample.keys():
                prompt = sample["prompt"]
            else:
                prompt = self.prompt
            
            proj_object_query = self.obj_linear_proj(object_query_token)#.unsqueeze(1)
            
            
            for obj_query, object_attn_mask, obj_score in zip(proj_object_query.transpose(0,1), attn_mask.transpose(0,1), object_scores.unsqueeze(-1)):
                
                text_tokens = self.query_tokens.expand(obj_query.shape[0], -1, -1)
                query_tokens = torch.cat([obj_query.unsqueeze(1), text_tokens], dim=1)
                
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=object_attn_mask,
                    return_dict=True,
                )

                # Opt input proj
                if self.opt_proj_before_agg:
                    inputs_opt = self.opt_proj(query_output.last_hidden_state)
                else:
                    inputs_opt = query_output.last_hidden_state
                
                input_opt_list.append(inputs_opt)
                input_opt_scores_list.append(obj_score)
                
        # Concatenate input embed on dimension 1
        if self.aggregation_method == "mean":
            input_opts = torch.stack(input_opt_list, dim=1)
            inputs_opt = input_opts.mean(dim=1)
        elif self.aggregation_method == "weighted mean":
            stacked_input_opt = torch.stack(input_opt_list, dim=0)  # shape num_t, bs, nq, ndim
            stacked_input_opt_scores = torch.stack(input_opt_scores_list, dim=0).unsqueeze(-1)  # shape num_t, 1

            # normalize scores with sum over sum
            normed_scores = stacked_input_opt_scores / (torch.sum(stacked_input_opt_scores, dim=0, keepdim=True) + 1e-6) # shape num_t, 1
            inputs_opt = torch.sum(stacked_input_opt * normed_scores.unsqueeze(-1).unsqueeze(-1), dim=0)
        elif self.aggregation_method == "arithmetic mean" :
            stacked_input_opt = torch.stack(input_opt_list, dim=0)  # shape num_t, bs, nq, ndim
            inputs_opt = torch.mean(stacked_input_opt, dim=0)
        elif self.aggregation_method == "concat":
            inputs_opt = torch.cat(input_opt_list, dim=1)
        elif self.aggregation_method == "linear":
            inputs_opt = torch.stack(input_opt_list, dim=0)  # shape num_t, bs, nq, ndim
            inputs_opt = inputs_opt.permute(1,2,0,3).flatten(-2,-1)  # bs, nq, num_t*ndim
            inputs_opt = self.agg_linear(inputs_opt)
        elif self.aggregation_method == "self-attention":
            inputs_opt = torch.stack(input_opt_list, dim=0)  # shape num_t, bs, nq, ndim
            # normalize 
            ##
            inputs_opt = inputs_opt.permute(1,2,0,3).flatten(-2,-1)  # bs, nq, num_t*ndim
            inputs_opt = self.agg_self_attention(inputs_opt)
            inputs_opt = self.agg_linear(inputs_opt)
        else : 
            raise ValueError("Unknown aggregation method: {}".format(self.aggregation_method))
        
        if not self.opt_proj_before_agg:
            inputs_opt = self.opt_proj(inputs_opt)

        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
            image.device
        )


        prompt = [prompt] * image.size(0)

        opt_tokens = self.opt_tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
        
        # new version for transformers>=4.27
        inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt,inputs_embeds],dim=1)
        
        outputs = self.opt_model.generate(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )
        output_text = self.opt_tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        # print("output_text shape: ", len(output_text))
        # print("output_text: ", output_text)
        
        output_text = [text.strip() for text in output_text]
        output_text_list.append(output_text)
        
        return output_text_list
        
        
    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs
    ):
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if isinstance(samples["text_input"], str):
                samples["text_input"] = [samples["text_input"]]
            if prompt:
                text_input = [prompt.format(question) for question in samples["text_input"]]
            else:
                text_input = samples["text_input"]

            self.opt_tokenizer.padding_side = "left"
            opt_tokens = self.opt_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
        
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            
            # require transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt,inputs_embeds],dim=1)
            
            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                eos_token_id=self.eos_token_id,
                length_penalty=length_penalty,
            )
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
        if self._apply_lemmatizer or ("apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]):
            output_text = self._lemmatize(output_text)

        return output_text
    
    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
        