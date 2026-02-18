"""
Extract example annotated frames from processed videos
and convert TECHNICAL_REPORT.md to report.pdf
"""
import cv2
import os
from pathlib import Path

# ============================================
# 1. Extract Example Frames
# ============================================
print("=" * 60)
print("EXTRACTING EXAMPLE FRAMES FROM PROCESSED VIDEOS")
print("=" * 60)

output_dir = Path('example_frames')
output_dir.mkdir(exist_ok=True)

# Select best performing videos for examples
video_files = [
    ('results/2_processed.mp4', '100% detection'),      # Best
    ('results/3_processed.mp4', '83.7% detection'),     # Good
    ('results/10_processed.mp4', '9.6% detection'),     # Show variance
]

frame_numbers = [50, 100, 150]  # Extract 3 frames from each

for video_path, description in video_files:
    if not os.path.exists(video_path):
        print(f"❌ {video_path} not found, skipping...")
        continue
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.basename(video_path).replace('_processed.mp4', '')
    
    print(f"\n📹 {video_name} ({description}, {total_frames} frames)")
    
    for frame_num in frame_numbers:
        if frame_num >= total_frames:
            continue
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            output_file = output_dir / f'{video_name}_frame{frame_num:03d}.jpg'
            cv2.imwrite(str(output_file), frame)
            print(f"  ✅ Frame {frame_num}: {output_file}")
    
    cap.release()

print(f"\n✅ All frames extracted to {output_dir}/")

# ============================================
# 2. Convert Markdown to HTML/PDF
# ============================================
print("\n" + "=" * 60)
print("CONVERTING TECHNICAL_REPORT.md TO report.pdf")
print("=" * 60)

try:
    # Try using markdown2 if available
    import markdown2
    
    # Read markdown with proper encoding
    with open('TECHNICAL_REPORT.md', 'r', encoding='utf-8') as f:
        md_text = f.read()
    
    # Convert to HTML
    html = markdown2.markdown(md_text, extras=['tables', 'fenced-code-blocks'])
    
    # Create styled HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Cricket Ball Tracking - Technical Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 40px;
                max-width: 900px;
                color: #333;
            }}
            h1 {{
                color: #007bff;
                border-bottom: 3px solid #007bff;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #0056b3;
                margin-top: 30px;
            }}
            h3 {{
                color: #444;
            }}
            code {{
                background: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }}
            pre {{
                background: #f4f4f4;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                border-left: 4px solid #007bff;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 15px 0;
                border: 1px solid #ddd;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #f8f9fa;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            blockquote {{
                border-left: 4px solid #007bff;
                padding-left: 15px;
                color: #666;
                font-style: italic;
            }}
            a {{
                color: #007bff;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
        </style>
    </head>
    <body>
        {html}
    </body>
    </html>
    """
    
    # Save HTML
    with open('report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("✅ Created report.html")
    
    # Try to convert HTML to PDF using wkhtmltopdf or other method
    try:
        import pdfkit
        options = {
            'page-size': 'A4',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'enable-local-file-access': None
        }
        pdfkit.from_file('report.html', 'report.pdf', options=options)
        print("✅ Created report.pdf")
    except Exception as e_pdf:
        print(f"⚠️  pdfkit not available ({str(e_pdf)})")
        print("   HTML version saved as report.html (can be printed as PDF)")
        print("   To create PDF manually:")
        print("     - Open report.html in browser")
        print("     - Press Ctrl+P and 'Save as PDF'")

except ImportError as e:
    print(f"⚠️  markdown2 not installed: {str(e)}")
    print("   Installing markdown2...")
    import subprocess
    subprocess.run(['pip', 'install', 'markdown2'], check=False)
    print("   Please run this script again.")

print("\n" + "=" * 60)
print("✅ ALL TASKS COMPLETE!")
print("=" * 60)
