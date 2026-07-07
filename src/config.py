"""Shared demo-scene geometry. The detection zone is expressed in frame fractions so
overlay / stub / clip generator agree without coupling to a resolution."""

# Zone = projected floor ellipse in front of the forklift (fractions of width/height).
# From the oblique CCTV angle the projection is a ROTATED ellipse.
ZONE_CENTER = (0.423, 0.594)
ZONE_AXES = (0.112, 0.086)   # semi-axes (from Blender camera projection, oblique framing)
ZONE_ANGLE = -172.8          # degrees (screen-space rotation of the ellipse)
NEAR_FACTOR = 2.1            # ellipse scaled by this = "near" ring

DISCLAIMER = "Concept demo - not a functional safety system"
