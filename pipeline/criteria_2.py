"""Shared error definitions distilled from Prompt Version 1.

These are reused across both subtasks and across the auto-CoT step
generation call. Keeping them in one file means we never let the
two subtasks drift apart on what counts as a logical/semantic/decision
error.
"""

CRITERIA = """\
You are evaluating a generative driving simulation. Each video has 8 frames
captured by 6 surrounding cameras of the ego vehicle. Each frame contains a
ground-truth (GT) image, a 3D bounding-box visualization, and a generated
image. Your job is to find safety problems in the *generated* image relative
to the GT.

There are three error categories. Score each on a 0.0 - 1.0 scale.

LOGICAL ERRORS - violations of physical or temporal coherence:
  - Sudden disappearance / appearance of vehicles or pedestrians.
  - Objects placed in physically implausible positions.
  - Jittery / impossible trajectories of the ego or other vehicles.
  - Velocity / angular-velocity inconsistencies between consecutive frames.
  - Cross-view inconsistency: the same object looks different (or moves
    in incompatible ways) when viewed from two adjacent cameras at the
    same time-step.

SEMANTIC ERRORS - mis-identifiable or distorted traffic elements in a single
image, while overall visual realism is preserved:
  - Blurring of vehicles, pedestrians, traffic lights that leads to
    misrepresentation.
  - Object class deformation: a car turning into a truck, a person into a
    bicycle, etc.
  - Multiple objects merging into one (two cars become one).
  - Distortion of road symbols, lane markings, traffic signs.

DECISION ERRORS - unsafe or illegal ego-vehicle behavior:
  - Ego driving toward a sidewalk, lawn, or other non-drivable area.
  - Ego speeding toward another vehicle or pedestrian.
  - Crossing multiple lanes at once / illegal lane changes.
  - Failing to stop at a red light, ignoring obvious obstacles.

SCORING SCALE (apply to every category):
  0.0       - no issue at all. THIS IS THE DEFAULT.
  0.1 - 0.2 - a visible artifact that a human would still consider the
              clip "clean"; e.g. slight blur of a distant object that is
              still clearly identifiable.
  0.3 - 0.4 - a clearly wrong element that does not change driving
              understanding.
  0.5 - 0.7 - dangerous but with room for course correction.
  0.8 - 1.0 - severe, near-certain harm.

BACKGROUND NOISE IS NOT AN ERROR. The following things are NEVER errors
and MUST be scored 0.0:
  - Minor lighting, exposure, or color differences between GT and GEN.
  - Mild overall blur or compression artifacts that do not hide or
    distort any specific traffic element.
  - Distant objects being slightly less sharp in GEN than in GT, as long
    as their class and position are still recognizable.
  - Small positional shifts of a static background object (e.g. a parked
    car a few pixels off) that do not affect ego-vehicle decisions.

CALIBRATION ANCHORS (use these as your internal reference; the human
annotators calibrated their scale this way):
  - "No visible issue, only minor lighting drift"               -> 0.0
  - "A pedestrian is briefly slightly blurred but still clearly
     a pedestrian in the same position"                         -> 0.1
  - "A road sign is distorted so you cannot read it"            -> 0.4
  - "A parked car blinks in/out for one frame, not on ego path" -> 0.5
  - "Two pedestrians merge into one on the road"                -> 0.8
  - "Ego vehicle drives straight into a pedestrian crossing"    -> 0.9

HARD RULES YOU MUST FOLLOW:
  1. If you cannot name a specific error from the three category
     definitions above, the score MUST be 0.0.
  2. "Slight blur", "minor distortion", "looks a bit different" without
     naming WHAT element is wrong and HOW it is wrong -> score 0.0.
  3. Scores >= 0.5 require concrete evidence on safety effects; if you
     are unsure on the evidence, score it at most 0.2. Note that there
     must be an extremely strong case to mark a clip beyond 0.8 and above.
  4. Be conservative and assume most clips are clean without safety issues
     before analysis. A score of 0.0 for a whole category is the expected
     outcome on an analyzed clean clip.
  5. Score the safety issues based on how they affect the ego vehicle.
"""
