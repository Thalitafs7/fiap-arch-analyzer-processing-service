"""
Gera um diagrama de arquitetura simples para testes E2E.
Requer: pip install Pillow
"""
from PIL import Image, ImageDraw, ImageFont
import os

WIDTH, HEIGHT = 800, 600
img = Image.new("RGB", (WIDTH, HEIGHT), "white")
draw = ImageDraw.Draw(img)

# Title
draw.text((250, 20), "Architecture Diagram", fill="black")

# Boxes representing components
components = [
    ("API Gateway", 50, 100, 200, 160),
    ("Auth Service", 300, 80, 480, 140),
    ("User Service", 300, 170, 480, 230),
    ("Database", 550, 130, 700, 190),
    ("Cache (Redis)", 550, 220, 700, 280),
    ("Message Queue", 300, 300, 480, 360),
    ("Worker Service", 550, 340, 700, 400),
    ("S3 Storage", 550, 430, 700, 490),
    ("Load Balancer", 50, 250, 200, 310),
    ("CDN", 50, 400, 200, 460),
]

for name, x1, y1, x2, y2 in components:
    draw.rectangle([x1, y1, x2, y2], outline="navy", width=2)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    draw.text((x1 + 5, cy - 5), name, fill="navy")

# Arrows (simple lines)
arrows = [
    (200, 130, 300, 110),   # Gateway -> Auth
    (200, 130, 300, 200),   # Gateway -> User
    (480, 200, 550, 160),   # User -> DB
    (480, 200, 550, 250),   # User -> Cache
    (480, 330, 550, 370),   # Queue -> Worker
    (700, 370, 700, 430),   # Worker -> S3
    (200, 280, 300, 330),   # LB -> Queue
]

for x1, y1, x2, y2 in arrows:
    draw.line([(x1, y1), (x2, y2)], fill="gray", width=2)

out_path = os.path.join(os.path.dirname(__file__), "test-diagram.png")
img.save(out_path, "PNG")
print(f"Saved: {out_path}")
