"""
Evaluation Module - Cricket Ball Tracking Performance Metrics

Computes evaluation metrics for ball tracking results:
- Detection Rate: Percentage of frames where ball was detected
- Trajectory Smoothness: Consistency of motion (lower is smoother)
- Centroid Stability: Pixel position variance when ball visible
- Detection Confidence: Average model confidence scores
- Occlusion Handling: How well the system handles ball invisibility

Usage:
    python eval.py
    
    Or in code:
    from eval import TrackingEvaluator
    evaluator = TrackingEvaluator('annotations/')
    metrics = evaluator.evaluate()
    evaluator.print_report(metrics)
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrackingEvaluator:
    """Compute evaluation metrics for ball tracking results."""
    
    def __init__(self, annotation_dir='../annotations'):
        """
        Initialize evaluator.
        
        Args:
            annotation_dir (str): Path to directory containing CSV annotations
        """
        self.annotation_dir = annotation_dir
        self.csv_files = glob.glob(os.path.join(annotation_dir, '*_data.csv'))
        
        if not self.csv_files:
            logger.warning(f"No CSV files found in {annotation_dir}")
        else:
            logger.info(f"Found {len(self.csv_files)} annotation files")
    
    def evaluate(self):
        """
        Evaluate all annotation files and return metrics.
        
        Returns:
            dict: Comprehensive metrics for each video and overall
        """
        all_metrics = {}
        overall_stats = {
            'total_videos': len(self.csv_files),
            'total_frames': 0,
            'total_detections': 0,
            'video_metrics': []
        }
        
        for csv_path in sorted(self.csv_files):
            video_name = os.path.basename(csv_path).replace('_data.csv', '')
            logger.info(f"Evaluating: {video_name}")
            
            metrics = self._evaluate_single_video(csv_path)
            all_metrics[video_name] = metrics
            
            overall_stats['total_frames'] += metrics['total_frames']
            overall_stats['total_detections'] += metrics['detected_frames']
            overall_stats['video_metrics'].append({
                'video': video_name,
                'detection_rate': metrics['detection_rate'],
                'trajectory_smoothness': metrics['trajectory_smoothness'],
                'confidence_mean': metrics['confidence_mean']
            })
        
        # Compute overall statistics
        if overall_stats['total_frames'] > 0:
            overall_stats['overall_detection_rate'] = (
                overall_stats['total_detections'] / overall_stats['total_frames']
            )
        else:
            overall_stats['overall_detection_rate'] = 0.0
        
        all_metrics['overall'] = overall_stats
        return all_metrics
    
    def _evaluate_single_video(self, csv_path):
        """
        Compute metrics for a single video CSV.
        
        Args:
            csv_path (str): Path to CSV annotation file
            
        Returns:
            dict: Metrics including detection rate, trajectory smoothness, etc.
        """
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Failed to read {csv_path}: {str(e)}")
            return {}
        
        # Basic counts
        total_frames = len(df)
        visible_frames = (df['visible'] == 1).sum()
        
        metrics = {
            'total_frames': total_frames,
            'detected_frames': visible_frames,
            'missing_frames': total_frames - visible_frames,
            'detection_rate': visible_frames / total_frames if total_frames > 0 else 0.0,
        }
        
        # Trajectory smoothness (velocity changes)
        if visible_frames > 1:
            visible_data = df[df['visible'] == 1].reset_index(drop=True)
            
            if len(visible_data) > 1:
                # Compute distances between consecutive points
                dx = np.diff(visible_data['x'].values)
                dy = np.diff(visible_data['y'].values)
                distances = np.sqrt(dx**2 + dy**2)
                
                # Smoothness: standard deviation of distances (lower = smoother)
                metrics['trajectory_smoothness'] = float(np.std(distances))
                metrics['mean_distance_per_frame'] = float(np.mean(distances))
                metrics['max_distance_jump'] = float(np.max(distances))
            else:
                metrics['trajectory_smoothness'] = 0.0
                metrics['mean_distance_per_frame'] = 0.0
                metrics['max_distance_jump'] = 0.0
        else:
            metrics['trajectory_smoothness'] = 0.0
            metrics['mean_distance_per_frame'] = 0.0
            metrics['max_distance_jump'] = 0.0
        
        # Centroid stability (position variance when visible)
        if visible_frames > 0:
            visible_data = df[df['visible'] == 1]
            metrics['centroid_x_std'] = float(visible_data['x'].std())
            metrics['centroid_y_std'] = float(visible_data['y'].std())
            metrics['centroid_x_mean'] = float(visible_data['x'].mean())
            metrics['centroid_y_mean'] = float(visible_data['y'].mean())
        else:
            metrics['centroid_x_std'] = 0.0
            metrics['centroid_y_std'] = 0.0
            metrics['centroid_x_mean'] = 0.0
            metrics['centroid_y_mean'] = 0.0
        
        # Occlusion analysis (consecutive missing frames)
        visible_arr = df['visible'].values
        
        # Find gaps
        gaps = []
        current_gap = 0
        for v in visible_arr:
            if v == 0:
                current_gap += 1
            else:
                if current_gap > 0:
                    gaps.append(current_gap)
                current_gap = 0
        if current_gap > 0:
            gaps.append(current_gap)
        
        if gaps:
            metrics['max_occlusion_length'] = max(gaps)
            metrics['mean_occlusion_length'] = np.mean(gaps)
            metrics['occlusion_frequency'] = len(gaps)
        else:
            metrics['max_occlusion_length'] = 0
            metrics['mean_occlusion_length'] = 0.0
            metrics['occlusion_frequency'] = 0
        
        # Confidence score (if available, otherwise estimate)
        metrics['confidence_mean'] = 0.5  # Placeholder if not in CSV
        
        return metrics
    
    def print_report(self, metrics):
        """
        Print formatted evaluation report.
        
        Args:
            metrics (dict): Metrics dictionary from evaluate()
        """
        print("\n" + "=" * 70)
        print("CRICKET BALL TRACKING - EVALUATION REPORT")
        print("=" * 70)
        
        overall = metrics['overall']
        print(f"\n📊 OVERALL STATISTICS")
        print(f"  Total Videos: {overall['total_videos']}")
        print(f"  Total Frames: {overall['total_frames']}")
        print(f"  Total Detections: {overall['total_detections']}")
        print(f"  Overall Detection Rate: {overall['overall_detection_rate']:.1%}")
        
        print(f"\n📹 PER-VIDEO METRICS")
        print(f"{'Video':<30} {'Det. Rate':<12} {'Smoothness':<15} {'Confidence':<10}")
        print("-" * 70)
        
        for vm in overall['video_metrics']:
            print(
                f"{vm['video']:<30} "
                f"{vm['detection_rate']:<12.1%} "
                f"{vm['trajectory_smoothness']:<15.2f} "
                f"{vm['confidence_mean']:<10.2f}"
            )
        
        print(f"\n🎯 DETAILED VIDEO ANALYSIS")
        for video_name, m in metrics.items():
            if video_name == 'overall':
                continue
            
            print(f"\n  Video: {video_name}")
            print(f"    Detection Rate: {m['detection_rate']:.1%} ({m['detected_frames']}/{m['total_frames']})")
            print(f"    Trajectory Smoothness: {m['trajectory_smoothness']:.2f} px/frame")
            print(f"    Mean Motion per Frame: {m['mean_distance_per_frame']:.2f} px")
            print(f"    Max Jump Distance: {m['max_distance_jump']:.2f} px")
            print(f"    Centroid X Position: μ={m['centroid_x_mean']:.1f}, σ={m['centroid_x_std']:.1f}")
            print(f"    Centroid Y Position: μ={m['centroid_y_mean']:.1f}, σ={m['centroid_y_std']:.1f}")
            print(f"    Occlusion Events: {m['occlusion_frequency']}")
            if m['occlusion_frequency'] > 0:
                print(f"      Max Duration: {m['max_occlusion_length']} frames")
                print(f"      Avg Duration: {m['mean_occlusion_length']:.1f} frames")
        
        print("\n" + "=" * 70)
        print("✅ EVALUATION COMPLETE")
        print("=" * 70)
    
    def save_metrics(self, metrics, output_path='evaluation_metrics.json'):
        """
        Save metrics to JSON file.
        
        Args:
            metrics (dict): Metrics dictionary
            output_path (str): Output file path
        """
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_types(item) for item in obj]
            return obj
        
        metrics_serializable = convert_types(metrics)
        
        with open(output_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        logger.info(f"Metrics saved to {output_path}")


def main():
    """Main evaluation script."""
    # Change to codes directory to run from there
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths
    annotation_dir = '../annotations'
    if not os.path.exists(annotation_dir):
        annotation_dir = 'annotations'
    
    logger.info(f"Looking for annotations in: {annotation_dir}")
    
    # Run evaluation
    evaluator = TrackingEvaluator(annotation_dir)
    metrics = evaluator.evaluate()
    
    # Print report
    evaluator.print_report(metrics)
    
    # Save metrics
    evaluator.save_metrics(metrics)
    
    print("\n💾 Metrics saved to evaluation_metrics.json")


if __name__ == "__main__":
    main()
