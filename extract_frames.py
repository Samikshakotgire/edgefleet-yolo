"""
Extract example annotated frames from processed videos
"""
import cv2
import os
from pathlib import Path

print("=" * 60)
print("EXTRACTING EXAMPLE FRAMES")
print("=" * 60)

output_dir = Path('example_frames')
output_dir.mkdir(exist_ok=True)

#Select specific videos and frames to extract
videos = [
    'results/2_processed.mp4',
    'results/3_processed.mp4', 
    'results/1_processed.mp4',
]

for video_path in videos:
    if not os.path.exists(video_path):
        print(f"❌ {video_path} not found")
        continue
    
    print(f"\n📹 Processing {video_path}...")
    cap = cv2.VideoCapture(video_path)
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   Frames: {total_frames}, FPS: {fps}, Size: {width}x{height}")
    
    # Extract frames
    video_name = os.path.basename(video_path).replace('_processed.mp4', '')
    
    # Extract frame at 50, 100, 150
    for frame_idx in [50, 100, 150]:
        if frame_idx >= total_frames:
            continue
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            output_file = output_dir / f'video_{video_name}_frame_{frame_idx:03d}.jpg'
            cv2.imwrite(str(output_file), frame)
            print(f"   ✅ Extracted frame {frame_idx} -> {output_file.name}")
    
    cap.release()

print("\n✅ Frame extraction complete!")
print(f"   Frames saved to: {output_dir}/")
