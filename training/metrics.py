# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Segmentation metrics for SAM2 model evaluation.
Computes IoU, Dice, and other performance metrics.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist

from training.utils.distributed import get_world_size, is_dist_avail_and_initialized


class SegmentationMetrics:
    """
    Base class for computing segmentation metrics.
    Supports multi-stage outputs and multiple objects.
    """

    def __init__(
        self,
        iou_thresholds: Optional[List[float]] = None,
        num_objects: int = 1,
        name: str = "SegmentationMetrics",
    ):
        """
        Args:
            iou_thresholds: List of IoU thresholds for evaluation (e.g., [0.5, 0.75])
            num_objects: Maximum number of objects per video
            name: Name of the metric
        """
        self.iou_thresholds = iou_thresholds or [0.5, 0.75]
        self.num_objects = num_objects
        self.name = name

        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.iou_values = []
        self.dice_values = []
        self.mae_values = []  # Mean Absolute Error
        self.count = 0

        # Per-threshold metrics
        self.iou_at_threshold = {f"iou_{t}": [] for t in self.iou_thresholds}

    def update(
        self,
        find_stages: Dict[str, Any],
        find_metadatas: List[Dict[str, Any]],
        targets,
        **kwargs,
    ) -> None:
        """
        Update metrics with predictions and ground truth.

        Args:
            find_stages: Model outputs containing predictions (dict with 'pred_masks', 'iou_predictions')
            find_metadatas: List of metadata dicts for each sample (may contain ground truth)
        """
        # Support both dict and list-shaped `find_stages` (per-sample outputs)
        pred_masks = None
        iou_predictions = None

        def _extract_pred_and_iou(fs: Any):
            """Return (pred_masks_tensor_or_none, iou_tensor_or_none) for a single find_stage entry."""
            if not isinstance(fs, dict):
                return None, None

            # Common keys to try (ordered by preference)
            if "pred_masks" in fs and fs["pred_masks"] is not None:
                return fs["pred_masks"], fs.get("iou_predictions", None)

            # multimask / multistep variants are often lists; take the last step
            for key in (
                "multistep_pred_multimasks",
                "multistep_pred_multimasks_high_res",
                "multistep_pred_masks",
                "multistep_pred_masks_high_res",
                "pred_masks_high_res",
            ):
                if key in fs and fs[key] is not None:
                    v = fs[key]
                    if isinstance(v, list) and len(v) > 0:
                        return v[-1], fs.get("multistep_pred_ious", None) or fs.get(
                            "iou_predictions", None
                        )
                    return v, fs.get("multistep_pred_ious", None) or fs.get(
                        "iou_predictions", None
                    )

            return None, fs.get("iou_predictions", None)

        if isinstance(find_stages, list):
            batch_preds = []
            iou_list = []
            for fs in find_stages:
                pm, ip = _extract_pred_and_iou(fs)
                if pm is not None:
                    # Ensure pm has a batch dimension
                    try:
                        if isinstance(pm, torch.Tensor):
                            if pm.dim() == 3:
                                pm = pm.unsqueeze(0)
                            # If pm has shape [1, M, H, W] or [B, M, H, W], concat on dim=0
                        else:
                            pm = torch.from_numpy(np.array(pm))
                            if pm.dim() == 3:
                                pm = pm.unsqueeze(0)
                    except Exception:
                        # fallback: ignore this sample
                        pm = None

                if pm is not None:
                    batch_preds.append(pm)
                if ip is not None:
                    iou_list.append(ip)

            if len(batch_preds) == 0:
                return

            try:
                pred_masks = torch.cat(batch_preds, dim=0)
            except Exception:
                # If concatenation fails, try stacking as a last resort
                pred_masks = torch.stack(batch_preds, dim=0)

            if len(iou_list) > 0:
                try:
                    iou_predictions = torch.cat(iou_list, dim=0)
                except Exception:
                    iou_predictions = None

        elif isinstance(find_stages, dict):
            pred_masks = find_stages.get("pred_masks", None)
            iou_predictions = find_stages.get("iou_predictions", None)

            if pred_masks is None:
                # Try multimask fallbacks
                for key in (
                    "multistep_pred_multimasks",
                    "multistep_pred_multimasks_high_res",
                    "pred_masks_high_res",
                ):
                    v = find_stages.get(key, None)
                    if v is not None:
                        pred_masks = v[-1] if isinstance(v, list) and len(v) > 0 else v
                        break

        # If no predictions found, nothing to update
        if pred_masks is None:
            return

        # Extract ground truth masks: prefer explicit targets, but allow metadata override
        gt_masks = targets
        if find_metadatas and len(find_metadatas) > 0:
            if isinstance(find_metadatas, list):
                for meta in find_metadatas:
                    if isinstance(meta, dict) and "gt_masks" in meta:
                        gt_masks = meta["gt_masks"]
                        break
            elif isinstance(find_metadatas, dict) and "gt_masks" in find_metadatas:
                gt_masks = find_metadatas["gt_masks"]

        # If targets is a tensor with channel dim (B,1,H,W), squeeze it to (B,H,W)
        if isinstance(gt_masks, torch.Tensor) and gt_masks.dim() == 4 and gt_masks.shape[1] == 1:
            gt_masks_proc = gt_masks.squeeze(1)
        else:
            gt_masks_proc = gt_masks

        if gt_masks_proc is None:
            # If no ground truth, only log iou predictions if available
            if iou_predictions is not None:
                self._update_iou_predictions(iou_predictions)
            return

        # Compute metrics against ground truth for the batch
        self._compute_metrics(pred_masks, gt_masks_proc)

        # Increment count by number of samples processed (if available)
        try:
            batch_n = pred_masks.shape[0]
        except Exception:
            batch_n = 1
        self.count += int(batch_n)

    def _update_iou_predictions(self, iou_predictions: torch.Tensor) -> None:
        """Update metrics using model's own IoU predictions"""
        if iou_predictions is None:
            return

        iou_pred = iou_predictions.detach().cpu().numpy()

        if iou_pred.ndim == 3:  # [B, M, 1] - batch, multimask, score
            iou_pred = iou_pred.squeeze(-1)  # [B, M]
            iou_pred = iou_pred.max(axis=1)  # Take max across masks per sample [B]
        elif iou_pred.ndim == 2:  # [B, M]
            iou_pred = iou_pred.max(axis=1)
        elif iou_pred.ndim == 1:  # [B]
            pass
        else:
            return

        self.iou_values.extend(iou_pred.tolist())

    def _compute_metrics(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> None:
        """
        Compute IoU, Dice, and MAE metrics.

        Args:
            pred_masks: Predicted masks [B, M, H, W] or [B, H, W]
            gt_masks: Ground truth masks [B, H, W] or [H, W]
            iou_predictions: Model's IoU predictions (for correlation analysis)
        """
        pred_masks = self._prepare_masks(pred_masks)
        gt_masks = self._prepare_masks(gt_masks, target_spatial_size=pred_masks.shape[-2:] if pred_masks is not None else None)

        if pred_masks is None or gt_masks is None:
            return

        # Ensure same device
        if pred_masks.device != gt_masks.device:
            pred_masks = pred_masks.to(gt_masks.device)

        # Move to CPU for numpy operations
        pred_masks = pred_masks.detach().cpu().numpy().astype(np.float32)
        gt_masks = gt_masks.detach().cpu().numpy().astype(np.float32)

        # Handle case where gt_masks is 2D (single sample)
        if gt_masks.ndim == 2:
            gt_masks = np.expand_dims(gt_masks, 0)
        if pred_masks.ndim == 2:
            pred_masks = np.expand_dims(pred_masks, 0)
        
        # Ensure batch dimension matches
        batch_size = min(pred_masks.shape[0], gt_masks.shape[0])

        for b in range(batch_size):
            pred = pred_masks[b]  # [M, H, W] or [H, W]
            gt = gt_masks[b]  # [H, W]

            # Take best prediction mask if multiple available
            if pred.ndim == 3:  # [M, H, W]
                # Select mask with highest IoU to GT
                ious = [self._compute_iou(pred[m], gt) for m in range(pred.shape[0])]
                best_idx = np.argmax(ious)
                pred = pred[best_idx]
                iou = ious[best_idx]
            else:  # [H, W]
                iou = self._compute_iou(pred, gt)

            dice = self._compute_dice(pred, gt)
            mae = self._compute_mae(pred, gt)

            self.iou_values.append(iou)
            self.dice_values.append(dice)
            self.mae_values.append(mae)

            # Compute metrics at different thresholds
            for threshold in self.iou_thresholds:
                at_threshold = 1.0 if iou >= threshold else 0.0
                key = f"iou_{threshold}"
                self.iou_at_threshold[key].append(at_threshold)

    def _prepare_masks(self, masks: torch.Tensor, target_spatial_size=None) -> Optional[torch.Tensor]:
        """Prepare and validate masks, optionally resize to target spatial size"""
        if masks is None:
            return None

        if not isinstance(masks, torch.Tensor):
            try:
                masks = torch.from_numpy(masks)
            except (TypeError, ValueError):
                return None

        # Convert bool to float if needed
        if masks.dtype == torch.bool:
            masks = masks.float()

        # Ensure float type for sigmoid and interpolation
        if masks.dtype not in [torch.float32, torch.float64]:
            masks = masks.float()

        # Sigmoid for logits (values outside [0, 1] range)
        if masks.numel() > 0 and (masks.min() < 0 or masks.max() > 1):
            masks = torch.sigmoid(masks)

        # Resize to target spatial size if provided and different
        if target_spatial_size is not None and masks.numel() > 0:
            current_spatial = masks.shape[-2:]
            if tuple(current_spatial) != tuple(target_spatial_size):
                # Ensure tensor has at least 4 dims for interpolate: [B, C, H, W]
                while masks.dim() < 4:
                    masks = masks.unsqueeze(1)
                masks = torch.nn.functional.interpolate(
                    masks, size=tuple(target_spatial_size), mode='bilinear', align_corners=False
                )

        return masks

    @staticmethod
    def _compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute Intersection over Union"""
        pred_binary = (pred > 0.5).astype(np.float32)
        gt_binary = (gt > 0.5).astype(np.float32)

        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()

        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        return float(intersection / union)

    @staticmethod
    def _compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute Dice coefficient"""
        pred_binary = (pred > 0.5).astype(np.float32)
        gt_binary = (gt > 0.5).astype(np.float32)

        intersection = np.logical_and(pred_binary, gt_binary).sum()
        dice = 2.0 * intersection / (pred_binary.sum() + gt_binary.sum() + 1e-8)

        return float(dice)

    @staticmethod
    def _compute_mae(pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute Mean Absolute Error"""
        return float(np.mean(np.abs(pred - gt)))

    def compute_synced(self) -> Dict[str, float]:
        """
        Compute final metrics and synchronize across distributed processes.
        Returns a dict suitable for logging.
        """
        metrics_dict = {}

        # Compute averages
        if len(self.iou_values) > 0:
            iou_mean = np.mean(self.iou_values)
            metrics_dict["IoU"] = iou_mean
            metrics_dict["IoU_std"] = float(np.std(self.iou_values))
        else:
            metrics_dict["IoU"] = 0.0
            metrics_dict["IoU_std"] = 0.0

        if len(self.dice_values) > 0:
            dice_mean = np.mean(self.dice_values)
            metrics_dict["Dice"] = dice_mean
            metrics_dict["Dice_std"] = float(np.std(self.dice_values))
        else:
            metrics_dict["Dice"] = 0.0
            metrics_dict["Dice_std"] = 0.0

        if len(self.mae_values) > 0:
            mae_mean = np.mean(self.mae_values)
            metrics_dict["MAE"] = mae_mean
        else:
            metrics_dict["MAE"] = 0.0

        # Threshold-based metrics
        for threshold in self.iou_thresholds:
            key = f"iou_{threshold}"
            if len(self.iou_at_threshold[key]) > 0:
                at_threshold_val = np.mean(self.iou_at_threshold[key])
                metrics_dict[f"IoU@{threshold}"] = at_threshold_val
            else:
                metrics_dict[f"IoU@{threshold}"] = 0.0

        # Synchronize across distributed processes if needed
        if is_dist_avail_and_initialized():
            for key in list(metrics_dict.keys()):
                metrics_dict[key] = self._sync_metric(
                    torch.tensor(metrics_dict[key], dtype=torch.float32)
                )

        return metrics_dict

    def compute(self) -> Dict[str, float]:
        """
        Alias for compute_synced() for compatibility with ProgressMeter.
        Called during training progress display.
        """
        return self.compute_synced()

    @staticmethod
    def _sync_metric(value: torch.Tensor) -> float:
        """Synchronize metric across all processes (average)"""
        if not is_dist_avail_and_initialized():
            return value.item()

        # Ensure tensor is on GPU for distributed operations (NCCL backend requires GPU)
        # If on CPU, skip distributed sync for metrics
        if value.device.type == "cpu":
            return value.item()

        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        world_size = get_world_size()
        return (value / world_size).item()

    def is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best (higher is better for IoU/Dice)"""
        return current > best

    def __repr__(self) -> str:
        return (
            f"{self.name}(iou_thresholds={self.iou_thresholds}, "
            f"num_objects={self.num_objects})"
        )


class SAM2SegmentationMetrics(SegmentationMetrics):
    """
    SAM2-specific segmentation metrics.
    Handles multi-stage predictions and object tracking scenarios.
    """

    def __init__(
        self,
        iou_thresholds: Optional[List[float]] = None,
        num_objects: int = 1,
    ):
        super().__init__(
            iou_thresholds=iou_thresholds,
            num_objects=num_objects,
            name="SAM2SegmentationMetrics",
        )

        # Additional SAM2-specific metrics
        self.obj_score_values = []
        self.num_frames_tracked = []
        self.track_stability = []

    def reset(self):
        super().reset()
        self.obj_score_values = []
        self.num_frames_tracked = []
        self.track_stability = []

    def update(
        self,
        find_stages: Dict[str, Any],
        find_metadatas: List[Dict[str, Any]],
        **kwargs,
    ) -> None:
        """Update with SAM2 multi-stage outputs"""
        super().update(find_stages, find_metadatas, **kwargs)

        # Track object scores if available
        if isinstance(find_stages, dict):
            obj_scores = find_stages.get("obj_scores", None)
            if obj_scores is not None:
                obj_scores = obj_scores.detach().cpu().numpy()
                if obj_scores.ndim >= 1:
                    self.obj_score_values.append(obj_scores.mean())

    def compute_synced(self) -> Dict[str, float]:
        """Compute SAM2-specific metrics"""
        metrics_dict = super().compute_synced()

        # Add object tracking confidence
        if len(self.obj_score_values) > 0:
            obj_score_mean = np.mean(self.obj_score_values)
            metrics_dict["ObjectConfidence"] = float(obj_score_mean)

        return metrics_dict


class EfficientTAMMetrics(SegmentationMetrics):
    """
    EfficientTAM-specific metrics for efficient tracking and segmentation.
    """

    def __init__(
        self,
        iou_thresholds: Optional[List[float]] = None,
        num_objects: int = 1,
    ):
        super().__init__(
            iou_thresholds=iou_thresholds,
            num_objects=num_objects,
            name="EfficientTAMMetrics",
        )
        self.inference_time_values = []
        self.memory_usage_values = []

    def reset(self):
        super().reset()
        self.inference_time_values = []
        self.memory_usage_values = []

    def compute_synced(self) -> Dict[str, float]:
        """Compute EfficientTAM metrics"""
        metrics_dict = super().compute_synced()

        if len(self.inference_time_values) > 0:
            metrics_dict["AvgInferenceTime"] = float(
                np.mean(self.inference_time_values)
            )

        if len(self.memory_usage_values) > 0:
            metrics_dict["AvgMemoryUsage"] = float(np.mean(self.memory_usage_values))

        return metrics_dict
