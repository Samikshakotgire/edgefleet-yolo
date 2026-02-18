"""
Create example annotated frames using CSV trajectory data
Shows what the overlays would look like
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from pathlib import Path

print("Creating example visualizations from CSV data...")

output_dir = Path('example_frames')
output_dir.mkdir(exist_ok=True)

# Use video 2 (100% detection rate)
csv_file = 'annotations/2_data.csv'

if not os.path.exists(csv_file):
    print(f"CSV not found: {csv_file}")
    exit(1)

# Read CSV
df = pd.read_csv(csv_file)
print(f"Loaded {csv_file}: {len(df)} frames")

# Create figure showing trajectory
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Cricket Ball Tracking - Example Output Visualization', fontsize=16, fontweight='bold')

# Plot 1: Full trajectory
ax = axes[0, 0]
visible_data = df[df['visible'] == 1]
ax.plot(visible_data['x'], visible_data['y'], 'r-', linewidth=2, label='Detected positions')
ax.scatter(visible_data['x'], visible_data['y'], c='red', s=30, alpha=0.6, label='Ball centroid')
ax.set_xlabel('X Position (pixels)')
ax.set_ylabel('Y Position (pixels)')
ax.set_title('Full Trajectory (Video 2 - 100% Detection Rate)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.invert_yaxis()

# Plot 2: Detection rate over time
ax = axes[0, 1]
window_size = 50
detect_rate = df['visible'].rolling(window=window_size, center=True).mean()
ax.plot(df['frame'], detect_rate * 100, 'b-', linewidth=2)
ax.fill_between(df['frame'], detect_rate * 100, alpha=0.3)
ax.set_xlabel('Frame Number')
ax.set_ylabel('Detection Rate (%)')
ax.set_title(f'Detection Rate (rolling {window_size}-frame window)')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 105])

# Plot 3: X and Y position over time
ax = axes[1, 0]
ax.plot(df['frame'], df['x'], 'r-', label='X position', alpha=0.7)
ax.plot(df['frame'], df['y'], 'b-', label='Y position', alpha=0.7)
ax.set_xlabel('Frame Number')
ax.set_ylabel('Position (pixels)')
ax.set_title('Ball Position Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Statistics
ax = axes[1, 1]
ax.axis('off')

stats_text = f"""
VIDEO STATISTICS (Video 2)

Total Frames: {len(df)}
Detected Frames: {(df['visible'] == 1).sum()}
Missing Frames: {(df['visible'] == 0).sum()}
Detection Rate: {(df['visible'] == 1).sum() / len(df) * 100:.1f}%

Position Statistics:
  X Range: {df[df['visible'] == 1]['x'].min():.0f} - {df[df['visible'] == 1]['x'].max():.0f} px
  Y Range: {df[df['visible'] == 1]['y'].min():.0f} - {df[df['visible'] == 1]['y'].max():.0f} px
  
  X Std Dev: {df[df['visible'] == 1]['x'].std():.2f} px
  Y Std Dev: {df[df['visible'] == 1]['y'].std():.2f} px

Motion per Frame:
  Mean: {df[df['visible'] == 1][['x', 'y']].diff().apply(lambda x: (x[0]**2 + x[1]**2)**0.5).mean():.2f} px
  Max: {df[df['visible'] == 1][['x', 'y']].diff().apply(lambda x: (x[0]**2 + x[1]**2)**0.5).max():.2f} px

Output Format (CSV):
  frame,x,y,visible
  0,512.3,298.1,1
  1,518.7,305.4,1
  2,-1.0,-1.0,0  (not detected)
"""

ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / 'example_output_analysis.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved: example_output_analysis.png")

# Create a simulated frame visualization
fig, ax = plt.subplots(figsize=(12, 8))

# Create fake image background (represent video frame)
# Assume frame is 640x480 or similar
frame_width, frame_height = 2560, 1920
ax.set_xlim(0, frame_width)
ax.set_ylim(frame_height, 0)
ax.set_aspect('equal')

# Draw background
ax.imshow([[0.3, 0.3], [0.3, 0.3]], cmap='gray', extent=[0, frame_width, frame_height, 0], alpha=0.3)

# Draw sample trajectory (frames 100-150 from dataframe)
sample_df = df[(df['frame'] >= 100) & (df['frame'] <= 150) & (df['visible'] == 1)]

if len(sample_df) > 0:
    # Draw trail (yellow)
    ax.plot(sample_df['x'], sample_df['y'], 'y-', linewidth=3, label='Trajectory Trail', alpha=0.7)
    
    # Draw detected positions (red circles)
    ax.scatter(sample_df['x'], sample_df['y'], c='red', s=100, edgecolors='darkred', 
              linewidth=2, label='Ball Centroid (RED)', zorder=5)
    
    # Draw predicted position (blue) for last point
    last_point = sample_df.iloc[-1]
    ax.scatter(last_point['x'], last_point['y'], c='blue', s=120, edgecolors='darkblue',
              linewidth=2, marker='s', label='Predicted Position (BLUE)', zorder=5)
    
    # Add annotations
    ax.text(last_point['x'] + 50, last_point['y'] - 50, f"PREDICTED\nConf: 0.85", 
           fontsize=9, color='blue', ha='left', weight='bold',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

ax.set_xlabel('X Position (pixels)', fontsize=12)
ax.set_ylabel('Y Position (pixels)', fontsize=12)
ax.set_title('Example Frame: Cricket Ball Tracking Visualization\n(Shows centroid detection and trajectory prediction)', 
            fontsize=14, weight='bold')
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(output_dir / 'example_frame_visualization.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved: example_frame_visualization.png")

plt.close('all')

print(f"\n✅ Example visualizations created in {output_dir}/")
print("   - example_output_analysis.png")
print("   - example_frame_visualization.png")
