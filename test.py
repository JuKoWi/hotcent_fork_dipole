import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

h = 1.0  # clipping line z = h

fig, ax = plt.subplots()

# clipping region (everything below the line)
clip = Rectangle((-10, -10), 20, 10 + h)

# circle
circle = Circle((0, 0), 4, fill=False, lw=2)

# important: give the transform
circle.set_clip_path(clip)

ax.add_patch(circle)

# boundary line
ax.axhline(h, color="red")

ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
ax.set_aspect("equal")

plt.show()
