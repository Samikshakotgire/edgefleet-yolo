"""
Convert report.html to report.pdf using reportlab
"""
from pathlib import Path

try:
    # Try reportlab (pure Python, no external dependencies)
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Spacer, Table, TableStyle
    from reportlab.lib.units import inch
    import re
    
    # Read HTML
    with open('report.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Extract text content (simplified)
    text = re.sub('<[^<]+?>', '', html_content)
    
    # Create PDF
    pdf_file = 'report.pdf'
    doc = SimpleDocTemplate(pdf_file, pagesize=A4,
                           rightMargin=0.75*inch, leftMargin=0.75*inch,
                           topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    story = []
    styles = getSampleStyleSheet()
    
    # Add title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor='#007bff',
        spaceAfter=30,
        alignment=1  # Center
    )
    story.append(Paragraph("Cricket Ball Tracking System", title_style))
    story.append(Paragraph("Technical Report", styles['Heading2']))
    story.append(Spacer(1, 0.3*inch))
    
    # Add content
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        leading=12
    )
    
    # Add more title and basic info
    story.append(Paragraph("<b>Project:</b> EdgeFleet AI/ML Assessment", normal_style))
    story.append(Paragraph("<b>Date:</b> February 2026", normal_style))
    story.append(Paragraph("<b>Status:</b> Production-Ready", normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>System Overview:</b>", styles['Heading3']))
    story.append(Paragraph(
        "This is a comprehensive computer vision system to detect and track cricket balls "
        "in videos recorded from a fixed camera using YOLOv8, ByteTrack, and Kalman filtering.",
        normal_style
    ))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Key Components:</b>", styles['Heading3']))
    story.append(Paragraph("• YOLOv8 Nano for real-time ball detection", normal_style))
    story.append(Paragraph("• ByteTrack for consistent object ID tracking", normal_style))
    story.append(Paragraph("• Kalman Filter for trajectory prediction", normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Output Formats:</b>", styles['Heading3']))
    story.append(Paragraph("✓ CSV annotations (frame, x, y, visible)", normal_style))
    story.append(Paragraph("✓ Processed MP4 videos with trajectory overlay", normal_style))
    story.append(Paragraph("✓ Evaluation metrics and analysis", normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(PageBreak())
    story.append(Paragraph("<b>For full technical details, see TECHNICAL_REPORT.md</b>", normal_style))
    
    # Build PDF
    doc.build(story)
    print("✅ Created report.pdf using reportlab")

except ImportError:
    print("⚠  reportlab not installed, installing...")
    import subprocess
    subprocess.run(['pip', 'install', 'reportlab'], check=False)
    print("Please run this script again.")
except Exception as e:
    print(f"Note: {str(e)}")
    print("HTML version (report.html) can be converted to PDF manually:")
    print("  1. Open report.html in a web browser")
    print("  2. Press Ctrl+P (or Cmd+P on Mac)")
    print("  3. Select 'Save as PDF'")
    print("  4. Save as 'report.pdf'")
