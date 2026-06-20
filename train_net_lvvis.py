# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Gabriel Fiastre for CaptionFormer (https://github.com/gabfstr/CaptionFormer)
#   from https://github.com/fanghaook/OVFormer/blob/main/train_net_lvvis.py
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import warnings
warnings.filterwarnings("ignore")

import copy
import itertools
import logging
import os

import wandb

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import (
    DefaultTrainer,
    HookBase,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from detectron2.utils.events import get_event_storage

# CaptionFormer
from captionformer import (
    YTVISDatasetMapper,
    YTVISDenseDVOCDatasetMapper,
    LVVISEvaluator_video,
    build_detection_train_loader,
    build_detection_test_loader,
    get_detection_dataset_dicts,
    add_captionformer_config,
)

from captionformer_video import add_captionformer_video_config


class WandbLoggingHook(HookBase):
    """
    A custom hook to log training losses to wandb during training.
    """
    def __init__(self, cfg, wandb_logger):
        self.cfg = cfg
        self.wandb_logger = wandb_logger

    def after_step(self):
        # Log loss values after each step
        storage = get_event_storage()
        if comm.is_main_process():  # Log only in the main process
            # get last loss values
            loss_dict = {k: v[0] for k, v in storage.latest().items() if "loss" in k}            
            self.wandb_logger.log(loss_dict, step=storage.iter)


class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super(ValidationLoss, self).__init__()
        self._loader = Trainer.build_train_loader(cfg)
        dataset_name = cfg.DATASETS.TRAIN[0]
        dataset_dicts = get_detection_dataset_dicts(dataset_name)
        dataset_len = len(dataset_dicts)
        print("Dataset name:", dataset_name)
        print("dataset len:", dataset_len)
        self._data_loader_iter = iter(self._loader)
        self.val_logger = logging.getLogger(__name__)
        self._period = cfg.TEST.EVAL_PERIOD
        self.eval_num_batch = dataset_len

    def before_step(self):
        print("Validation Loss Hook: Running validation loss calculation every {} steps".format(self._period))
        losses = {}
        with torch.no_grad():
            for idx in range(self.eval_num_batch):
                batch = next(self._data_loader_iter)  # Get the next batch from the loader
                loss_dict = self.trainer.model(batch)
                for k, v in loss_dict.items():
                    if k not in losses:
                        losses[k] = v.item()
                    else:
                        losses[k] += v.item()
        # Average the losses
        for k in losses.keys():
            losses[k] /= (idx + 1)
        # Save a file in the model directory with the losses 
        file_name = self.trainer.cfg.MODEL.WEIGHTS.replace(".pth", ".json").replace("model_", "val_loss_")
        import json
        # losses : torch to item 
        for k in losses.keys():
            if isinstance(losses[k], torch.Tensor):
                losses[k] = losses[k].item()
        print("\n\nValidation Losses:", losses)
        with open(file_name, "w") as f:
            json.dump(losses, f)
        raise NotImplementedError("Validation Losses")


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)
        evaluation_mode = "bbox" if cfg.MODEL.MASK_FORMER.BOX_MODE_ON == True else "segm"
        return LVVISEvaluator_video(dataset_name, cfg, True, output_folder, eval_mode=evaluation_mode)

    @classmethod
    def build_train_loader(cls, cfg):
        dataset_name = cfg.DATASETS.TRAIN[0]

        if cfg.MODEL.MASK_FORMER.BOX_MODE_ON == True and ("vidstg" in dataset_name or "bensmot" in dataset_name or "vln" in dataset_name):
            if "vln" in dataset_name:
                print("\n\nUSING VLN DATASET MAPPER\n\n")
                mapper = YTVISDenseDVOCDatasetMapper(cfg, is_train=True, box_only=True, anno_has_msk=True)
            else:
                if "bensmot" in dataset_name:
                    print("\n\nUSING BENSMOT DATASET MAPPER\n\n")
                mapper = YTVISDenseDVOCDatasetMapper(cfg, is_train=True, box_only=True)
        elif cfg.MODEL.MASK_FORMER.USE_BOXES == True and  ("lvvis" in dataset_name or "vln" in dataset_name):
            if "vln" in dataset_name:
                print("\n\nUSING VLN DATASET MAPPER\n\n")
                raise NotImplementedError("VLN with box not implemented yet")
            mapper = YTVISDenseDVOCDatasetMapper(cfg, is_train=True)
        elif cfg.MODEL.MASK_FORMER.MASK_CAPTIONING == True:
            print("babab")
            mapper = YTVISDenseDVOCDatasetMapper(cfg, is_train=True)
        else:
            mapper = YTVISDatasetMapper(cfg, is_train=True)

        dataset_dict = get_detection_dataset_dicts(
            dataset_name,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )

        return build_detection_train_loader(cfg, mapper=mapper, dataset=dataset_dict)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        dataset_name = cfg.DATASETS.TEST[0]
        if cfg.MODEL.MASK_FORMER.MASK_CAPTIONING == True:
            mapper = YTVISDenseDVOCDatasetMapper(cfg, is_train=False)
        elif cfg.MODEL.MASK_FORMER.BOX_MODE_ON == True and "vidstg" in dataset_name:
            mapper = YTVISDenseDVOCDatasetMapper(cfg, is_train=False, box_only=True)
        else :
            mapper = YTVISDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        from torch.cuda.amp import autocast
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        # Enable timing if requested via config or command-line
        # enable_timing = cfg.TEST.get('ENABLE_TIMING', False)
        enable_timing = True
        if enable_timing:
            logger.info("Enabling inference timing statistics collection...")
            model.enable_timing = True
            if hasattr(model, 'reset_timing_statistics'):
                model.reset_timing_statistics()

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            with autocast():
                results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        # Save timing statistics if enabled
        if hasattr(model, 'enable_timing') and model.enable_timing and comm.is_main_process():
            timing_output_path = os.path.join(cfg.OUTPUT_DIR, 'timing_statistics.json')
            logger.info(f"Saving timing statistics to {timing_output_path}...")
            try:
                stats = model.save_timing_statistics(timing_output_path)
                # Also add to results for easy access
                if stats:
                    results['timing_statistics'] = stats
            except Exception as e:
                logger.warning(f"Failed to save timing statistics: {e}")

        if len(results) == 1:
            results = list(results.values())[0]
        return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_captionformer_config(cfg)
    add_captionformer_video_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Enable timing statistics if requested
    if hasattr(args, 'enable_timing') and args.enable_timing:
        cfg.defrost()
        cfg.TEST.ENABLE_TIMING = True
        cfg.freeze()

    # if box_only set box to true
    if cfg.MODEL.MASK_FORMER.BOX_ONLY == True:
        cfg.MODEL.MASK_FORMER.USE_BOXES = True
        cfg.MODEL.MASK_FORMER.USE_MASKS = False
        cfg.MODEL.MASK_FORMER.BOX_MODE_ON = True # temporary
    # temporary box mode on means the same
    if cfg.MODEL.MASK_FORMER.BOX_MODE_ON == True:
        cfg.MODEL.MASK_FORMER.USE_BOXES = True
        cfg.MODEL.MASK_FORMER.USE_MASKS = False
        cfg.MODEL.MASK_FORMER.BOX_ONLY = True
        
    if args.eval_only:
        dataset = cfg.DATASETS.TEST[0]
    else:
        dataset = cfg.DATASETS.TRAIN[0]
    if "vidstg" in dataset:
        if not(args.eval_only):
            assert cfg.MODEL.MASK_FORMER.BOX_MODE_ON == True ; "Box mode only for vidstg dataset"
        elif cfg.MODEL.MASK_FORMER.BOX_MODE_ON == False :
            print("Computing masks on vidstg dataset")
        cfg.MODEL.CAPTIONING_HEAD.FEATURE_MAPPING = "vidstg"
    if "bensmot" in dataset:
        assert cfg.MODEL.MASK_FORMER.BOX_MODE_ON == True ; "Box mode only for bensmot dataset"
        cfg.MODEL.CAPTIONING_HEAD.FEATURE_MAPPING = "bensmot"
    if "vln" in dataset:
        cfg.MODEL.CAPTIONING_HEAD.FEATURE_MAPPING = "vln_uvo_sparse"

    if cfg.MODEL.MASK_FORMER.VIDEO_LEVEL_TRAINING == True:
        cfg.INPUT.SAMPLING_FULL_VIDEO = True
        cfg.MODEL.MASK_FORMER.INFERENCE_CLIP_LEN = cfg.INPUT.SAMPLING_FRAME_NUM
        # batch size = world size
        # cfg.SOLVER.IMS_PER_BATCH=

    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(name="mask2former")
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former_video")
    return cfg


def main(args):
    cfg = setup(args)

    # if rank is 0 
    if args.wandb is True and comm.is_main_process():
        if args.wandb_id is not None:
            wandb_logger=wandb.init(project=args.proj_name, entity=args.entity, name = args.exp_name, mode = "offline", id=args.wandb_id, resume='must')
        else :
            wandb_logger=wandb.init(project=args.proj_name, entity=args.entity, name = args.exp_name, mode = "offline")
    
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            raise NotImplementedError
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    if args.eval_val_loss:
        trainer = Trainer(cfg)
        trainer.register_hooks([ValidationLoss(cfg)])
        trainer.resume_or_load(resume=args.resume)
        return trainer.train()
    
    trainer = Trainer(cfg)
    if args.wandb is True and comm.is_main_process():
        trainer.register_hooks([WandbLoggingHook(cfg, wandb_logger)])
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser()
    args.add_argument("-wandb", "--wandb", action="store_true")
    args.add_argument("-proj_name", "--proj_name", type=str, default="captionformer")
    args.add_argument("-entity", "--entity", type=str, default=None)
    args.add_argument("-exp_name", "--exp_name", type=str, default="")
    args.add_argument("--wandb_id", type=str, default=None)
    args.add_argument("--eval-val-loss", action="store_true", help="Evaluate validation loss for a given checkpoint")
    args.add_argument("--enable-timing", action="store_true", help="Enable inference timing statistics collection and save to file")
    args = args.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
