"""
Test script to verify segmentation metrics implementation
Run: python test_metrics.py
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

from training.metrics import (
    SegmentationMetrics,
    SAM2SegmentationMetrics,
    EfficientTAMMetrics,
)


def test_basic_metric_creation():
    """Test metric instantiation"""
    print("=" * 60)
    print("Test 1: Metric Creation")
    print("=" * 60)
    
    metric = SegmentationMetrics(
        iou_thresholds=[0.5, 0.75],
        num_objects=1,
        name="TestMetric"
    )
    print(f"✓ Created: {metric}")
    print(f"  - IoU thresholds: {metric.iou_thresholds}")
    print(f"  - Max objects: {metric.num_objects}")
    print()


def test_metric_update():
    """Test metric updates with dummy data"""
    print("=" * 60)
    print("Test 2: Metric Updates")
    print("=" * 60)
    
    metric = SAM2SegmentationMetrics(
        iou_thresholds=[0.5, 0.75, 0.9],
        num_objects=1
    )
    
    # Create dummy data
    batch_size = 4
    height, width = 256, 256
    
    pred_masks = torch.sigmoid(torch.randn(batch_size, 1, height, width))
    gt_masks = torch.randint(0, 2, (batch_size, height, width)).float()
    iou_predictions = torch.rand(batch_size, 1, 1)
    
    # Create outputs dict
    outputs = {
        'pred_masks': pred_masks,
        'iou_predictions': iou_predictions,
    }
    
    # Create metadata with ground truth (batch-level)
    metadata = {
        'gt_masks': gt_masks
    }
    
    # Update metric
    metric.update(
        find_stages=outputs,
        find_metadatas=metadata
    )
    
    print(f"✓ Updated metric with {batch_size} samples")
    print(f"  - Accumulated IoU values: {len(metric.iou_values)}")
    print(f"  - Accumulated Dice values: {len(metric.dice_values)}")
    print(f"  - Accumulated MAE values: {len(metric.mae_values)}")
    print()


def test_metric_computation():
    """Test metric computation and output"""
    print("=" * 60)
    print("Test 3: Metric Computation")
    print("=" * 60)
    
    metric = SAM2SegmentationMetrics(
        iou_thresholds=[0.5, 0.75, 0.9],
        num_objects=1
    )
    
    # Add multiple batches of data
    for batch_idx in range(3):
        batch_size = 2
        height, width = 256, 256
        
        # Create slightly better predictions each time
        noise = 0.1 * (1 - batch_idx * 0.2)
        pred_masks = torch.sigmoid(torch.randn(batch_size, 1, height, width) * noise)
        gt_masks = torch.randint(0, 2, (batch_size, height, width)).float()
        iou_predictions = torch.rand(batch_size, 1, 1)
        
        outputs = {
            'pred_masks': pred_masks,
            'iou_predictions': iou_predictions,
        }
        
        metadata = {
            'gt_masks': gt_masks
        }
        
        metric.update(find_stages=outputs, find_metadatas=metadata)
    
    # Compute results
    results = metric.compute_synced()
    
    print("✓ Metric computation results:")
    for key, value in sorted(results.items()):
        print(f"  - {key}: {value:.4f}")
    print()


def test_efficiency_metrics():
    """Test EfficientTAM-specific metrics"""
    print("=" * 60)
    print("Test 4: EfficientTAM Metrics")
    print("=" * 60)
    
    metric = EfficientTAMMetrics(
        iou_thresholds=[0.5, 0.75],
        num_objects=1
    )
    
    # Create dummy data
    batch_size = 4
    pred_masks = torch.rand(batch_size, 1, 256, 256)
    gt_masks = torch.randint(0, 2, (batch_size, 256, 256)).float()
    
    outputs = {
        'pred_masks': pred_masks,
        'iou_predictions': torch.rand(batch_size, 1, 1),
    }
    
    metadata = {
        'gt_masks': gt_masks
    }
    
    metric.update(find_stages=outputs, find_metadatas=metadata)
    
    # Manually add timing/memory data
    metric.inference_time_values.extend([0.045, 0.048, 0.046, 0.047])
    metric.memory_usage_values.extend([2.1, 2.15, 2.0, 2.2])
    
    results = metric.compute_synced()
    
    print("✓ EfficientTAM metrics:")
    for key, value in sorted(results.items()):
        print(f"  - {key}: {value:.4f}")
    print()


def test_reset():
    """Test metric reset"""
    print("=" * 60)
    print("Test 5: Metric Reset")
    print("=" * 60)
    
    metric = SAM2SegmentationMetrics(iou_thresholds=[0.5])
    
    # Add data
    batch_size = 2
    outputs = {
        'pred_masks': torch.rand(batch_size, 1, 256, 256),
        'iou_predictions': torch.rand(batch_size, 1, 1),
    }
    metadata = {
        'gt_masks': torch.rand(batch_size, 256, 256)
    }
    
    metric.update(find_stages=outputs, find_metadatas=metadata)
    print(f"✓ Before reset: {len(metric.iou_values)} IoU values")
    
    # Reset
    metric.reset()
    print(f"✓ After reset: {len(metric.iou_values)} IoU values")
    print()


def test_is_better():
    """Test is_better comparison method"""
    print("=" * 60)
    print("Test 6: Better Metric Comparison")
    print("=" * 60)
    
    metric = SegmentationMetrics()
    
    assert metric.is_better(0.85, 0.80) == True, "Higher should be better"
    assert metric.is_better(0.70, 0.80) == False, "Lower should not be better"
    assert metric.is_better(0.80, 0.80) == False, "Equal should not be better"
    
    print("✓ is_better() comparisons:")
    print(f"  - 0.85 > 0.80? {metric.is_better(0.85, 0.80)}")
    print(f"  - 0.70 > 0.80? {metric.is_better(0.70, 0.80)}")
    print(f"  - 0.80 > 0.80? {metric.is_better(0.80, 0.80)}")
    print()


def test_multi_threshold():
    """Test multiple IoU thresholds"""
    print("=" * 60)
    print("Test 7: Multiple IoU Thresholds")
    print("=" * 60)
    
    thresholds = [0.3, 0.5, 0.7, 0.9]
    metric = SAM2SegmentationMetrics(
        iou_thresholds=thresholds,
        num_objects=1
    )
    
    # Create data
    batch_size = 10
    outputs = {
        'pred_masks': torch.rand(batch_size, 1, 256, 256),
        'iou_predictions': torch.rand(batch_size, 1, 1),
    }
    metadata = {
        'gt_masks': torch.rand(batch_size, 256, 256)
    }
    
    metric.update(find_stages=outputs, find_metadatas=metadata)
    results = metric.compute_synced()
    
    print(f"✓ Thresholds: {thresholds}")
    print("  Threshold-based metrics:")
    for threshold in thresholds:
        key = f"IoU@{threshold}"
        if key in results:
            print(f"    - {key}: {results[key]:.4f}")
    print()


def test_no_ground_truth():
    """Test metric update without ground truth"""
    print("=" * 60)
    print("Test 8: No Ground Truth (Model Predictions Only)")
    print("=" * 60)
    
    metric = SAM2SegmentationMetrics(iou_thresholds=[0.5])
    
    # Create data WITHOUT ground truth
    batch_size = 4
    outputs = {
        'pred_masks': torch.rand(batch_size, 1, 256, 256),
        'iou_predictions': torch.rand(batch_size, 1, 1) * 0.9,  # High confidence
    }
    metadata = {}  # Empty metadata, no ground truth
    
    metric.update(find_stages=outputs, find_metadatas=metadata)
    results = metric.compute_synced()
    
    print("✓ Update without ground truth (using model predictions):")
    print(f"  - IoU values accumulated: {len(metric.iou_values)}")
    if results['IoU'] >= 0:
        print(f"  - Results available: {True}")
    print()


def run_all_tests():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  SAM2 SEGMENTATION METRICS - TEST SUITE".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    try:
        test_basic_metric_creation()
        test_metric_update()
        test_metric_computation()
        test_efficiency_metrics()
        test_reset()
        test_is_better()
        test_multi_threshold()
        test_no_ground_truth()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print()
        print("Metrics system is ready to use!")
        print()
        return True
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
