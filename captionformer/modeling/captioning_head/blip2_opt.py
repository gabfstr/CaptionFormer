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
from packaging import version
import contextlib
import os

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

import torch.nn.functional as F
from math import sqrt, ceil

from detectron2.config import configurable
from detectron2.utils.registry import Registry

from .Qformer import BertLMHeadModel

from transformers import BertTokenizer, AutoTokenizer, OPTForCausalLM, OPTConfig
from transformers.models.bert.configuration_bert import BertConfig
import transformers

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



def build_captioning_head(cfg, input_channels):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.MASK_FORMER.CAPTIONING_HEAD_NAME
    return CAPTIONING_HEAD_REGISTRY.get(name)(cfg, input_channels)


class LVISFeatureLoader():
    """
    A trivial loader for frozen vision features.
    """
    def __init__(self, feature_mapping, feature_batch_size=64, dataset="lvis"):
        assert dataset in ["lvis", "vg"], "Dataset not supported"
        # embed dim for use in the model
        self.num_features = 1408

        with open(feature_mapping, "r") as f:
            self.mapping = json.load(f)
    
        self.feature_batch_size = feature_batch_size
        
    def __call__(self, img_ids):
        features = []
        for img_id in img_ids:
            file_name = self.mapping[str(img_id)]["file_name"]
            idx = self.mapping[str(img_id)]["index"]
            features.append(torch.load(file_name)[idx])
        
        return torch.stack(features, dim=0)



# @registry.register_model("blip2_opt")
@CAPTIONING_HEAD_REGISTRY.register()
class Blip2OPT(nn.Module):
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
        
        if feature_mapper == "lvis":
            self.feature_mapper='lvis'
            self.feature_mapping = "./datasets/lvis/features/index_mapping.json"
        elif feature_mapper == "vg":
            self.feature_mapper='vg'
            self.feature_mapping = "./datasets/VisualGenome/features/index_mapping.json"
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
            visual_encoder = LVISFeatureLoader(self.feature_mapping, dataset=self.feature_mapper)
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

        return ret


    def forward(self, samples, object_query_tokens, object_pred_masks_or_boxes, indices):
        
        if not self.saved_features:
            og_image = samples["image"]
            image = F.interpolate(og_image, size=self.img_size, mode='bilinear', align_corners=False)

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_query_tokens = object_query_tokens[src_idx].unsqueeze(1)
        src_pred_masks_or_boxes = object_pred_masks_or_boxes[src_idx].unsqueeze(1)
        
        target_labels = [t["captions"] for t in samples["target"]]

        if self.train :
            if len(tgt_idx[0]) == 0:
                return torch.tensor(0.0, device=self.device)
            
            num_queries = object_query_tokens.size(1)

            if "feature_id" in samples and self.saved_features:
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
            src_query_tokens = self.obj_linear_proj(src_query_tokens)

            loss_sum = torch.tensor(0.0, device=self.device)
            num_gt = 0
            for img_idx, cap_idx, obj_query, object_attn_mask in zip(tgt_idx[0], tgt_idx[1], src_query_tokens, attn_mask):

                text = target_labels[img_idx][cap_idx]
                if text==None or text=="":
                    continue
                text = text + "\n"
                num_gt += 1
                

                #Cat with text tokens
                text_tokens = self.query_tokens.expand(obj_query.shape[0], -1, -1)
                query_tokens = torch.cat([obj_query.unsqueeze(1), text_tokens], dim=1)
                

                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds[img_idx].unsqueeze(0),
                    encoder_attention_mask=object_attn_mask.unsqueeze(1),
                    return_dict=True,
                )

                # Opt input proj
                inputs_opt = self.opt_proj(query_output.last_hidden_state)
                atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(attn_mask.device)

                self.opt_tokenizer.padding_side = "right"                
                
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
                    torch.ones(atts_opt.size(), dtype=torch.long).to(attn_mask.device).fill_(-100)
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
                loss_sum += loss

                del text_tokens, query_tokens, query_output, inputs_opt, atts_opt, opt_tokens, targets, empty_targets, inputs_embeds, attention_mask, outputs, loss
                torch.cuda.empty_cache()
            
            if num_gt == 0:
                return loss_sum
            loss_reduced = loss_sum / num_gt

            return loss_reduced
        else:
            return self.generate(self, samples, object_query_tokens, object_pred_masks_or_boxes)

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
        with self.maybe_autocast():
            num_queries = object_query_tokens.size(1)
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
            proj_object_query = self.obj_linear_proj(object_query_tokens)

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
        