#!/usr/bin/env python3
from shellforgepy.simple import create_basic_box, translate, rotate, chain_translations

box = create_basic_box(10, 10, 10)
print('Original center:', box.BoundBox.Center)

# Test individual transforms
translated = translate(10, 0, 0)(box)
print('After translate(10,0,0):', translated.BoundBox.Center)

rotated = rotate(90, center=(10, 0, 0), axis=(0, 0, 1))(translated)
print('After rotate 90° around (10,0,0):', rotated.BoundBox.Center)

# Test chained
transform = chain_translations(
    translate(10, 0, 0),
    rotate(90, center=(10, 0, 0), axis=(0, 0, 1))
)
chained = transform(box)
print('After chained transforms:', chained.BoundBox.Center)