"""Failure-aware perception policy v1.

Turns one image's list of detections into a single policy decision
(ACCEPT_SORT / ROUTE_TO_REJECT / SKIP_NO_DETECTION / RETRY_VIEW /
MANUAL_REVIEW) plus a machine-readable reason, instead of handing every
raw detection straight to the sort pipeline. This is a plain-Python
decision layer with no ROS2 message dependency, so it's callable both from
vision_perception_node.py at runtime and from offline log analysis
(tools/analyze_real_image_detections.py).

Design notes on the two branches not spelled out by a numbered policy rule:
  - mapping_gap_possible: today's YOLO/COCO class mapping
    (COCO_CLASS_ID_TO_PROJECT_CLASS in vision_perception_node.py) never
    emits "can"/"glass_bottle" as class_name -- only SUPPORTED_EMITTED_
    CLASSES actually occurs in practice. If a detection ever shows up with
    one of these class names anyway (future model/mapping change), that's
    outside today's verified behavior, so it goes to MANUAL_REVIEW instead
    of being silently sorted or rejected.
  - unsupported_class_mapping: a class_name outside KNOWN_CLASSES entirely
    (neither a known project class nor "unknown") is treated the same way
    -- MANUAL_REVIEW rather than guessing.

Priority order (most confident/safety-critical checks first):
  1. no detections at all
  2. every detection is "unknown"
  3. more detections than max_detections_for_auto_sort (cluttered scene --
     don't trust any single top pick)
  4. top1 vs top2 confidence too close to call (ambiguous)
  5. top1 is "unknown" but detections weren't unanimous (still not
     sortable -- the best-supported read of the scene is "not a known
     object")
  6. top1's class isn't one the current mapping can even emit
     (mapping_gap_possible / unsupported_class_mapping)
  7. plain confidence-tier decision on a known, currently-emittable top1
"""

# Objects the sorting cell is designed to route to a specific bin.
KNOWN_CLASSES = {"plastic_bottle", "paper_cup", "can", "glass_bottle"}

# Project classes the current YOLO/COCO mapping can actually emit as
# class_name today. can/glass_bottle are in KNOWN_CLASSES but not here.
SUPPORTED_EMITTED_CLASSES = {"plastic_bottle", "paper_cup", "unknown"}

ACCEPT_SORT = "ACCEPT_SORT"
ROUTE_TO_REJECT = "ROUTE_TO_REJECT"
SKIP_NO_DETECTION = "SKIP_NO_DETECTION"
RETRY_VIEW = "RETRY_VIEW"
MANUAL_REVIEW = "MANUAL_REVIEW"

DECISIONS = (
    ACCEPT_SORT, ROUTE_TO_REJECT, SKIP_NO_DETECTION, RETRY_VIEW,
    MANUAL_REVIEW,
)

_DECISION_TO_ACTION = {
    ACCEPT_SORT: "SORT",
    ROUTE_TO_REJECT: "REJECT",
    SKIP_NO_DETECTION: "SKIP",
    RETRY_VIEW: "RETRY_VIEW",
    MANUAL_REVIEW: "MANUAL_REVIEW",
}


def _result(decision, reason, detections, selected_index=None,
            is_ambiguous=False):
    if selected_index is None:
        selected_class = None
        selected_confidence = None
    else:
        selected_class = detections[selected_index]['class_name']
        selected_confidence = detections[selected_index]['confidence']
    return {
        'decision': decision,
        'reason': reason,
        'selected_class': selected_class,
        'selected_confidence': selected_confidence,
        'selected_index': selected_index,
        'is_ambiguous': is_ambiguous,
        'num_detections': len(detections),
        'recommended_action': _DECISION_TO_ACTION[decision],
    }


def evaluate_detections(
        detections,
        confidence_threshold=0.5,
        low_confidence_threshold=0.35,
        ambiguity_margin=0.15,
        max_detections_for_auto_sort=3):
    """Decide what to do with one image's detections.

    detections: list of {"class_name": str, "confidence": float,
    "object_id": str (optional)} -- ranking by confidence is done here, so
    input order doesn't matter.

    Returns a dict: decision, reason, selected_class, selected_confidence,
    selected_index (into the input `detections` list), is_ambiguous,
    num_detections, recommended_action.
    """
    if not detections:
        return _result(SKIP_NO_DETECTION, 'no_detection', detections)

    ranked = sorted(
        range(len(detections)),
        key=lambda i: detections[i]['confidence'],
        reverse=True)
    top1_idx = ranked[0]
    top1 = detections[top1_idx]

    if all(d['class_name'] == 'unknown' for d in detections):
        return _result(
            ROUTE_TO_REJECT, 'unknown_object', detections, top1_idx)

    if len(detections) > max_detections_for_auto_sort:
        return _result(
            MANUAL_REVIEW, 'too_many_detections', detections, top1_idx)

    if len(ranked) >= 2:
        top2_idx = ranked[1]
        top2 = detections[top2_idx]
        confidence_gap = top1['confidence'] - top2['confidence']
        if confidence_gap < ambiguity_margin \
                and top1['class_name'] != top2['class_name']:
            top1_unknown = top1['class_name'] == 'unknown'
            top2_unknown = top2['class_name'] == 'unknown'
            if top1_unknown != top2_unknown:
                # One candidate is unknown, the other isn't, and we can't
                # tell them apart confidently -- reject rather than risk
                # sorting something that might not be sortable at all.
                return _result(
                    ROUTE_TO_REJECT, 'ambiguous_multi_detection', detections,
                    top1_idx, is_ambiguous=True)
            return _result(
                RETRY_VIEW, 'ambiguous_multi_detection', detections,
                top1_idx, is_ambiguous=True)

    # top1 is unknown but detections weren't unanimous (else caught above
    # by the all-unknown check) -- the most-confident read of the scene is
    # still "not a known object".
    if top1['class_name'] == 'unknown':
        return _result(
            ROUTE_TO_REJECT, 'unknown_object', detections, top1_idx)

    if top1['class_name'] not in SUPPORTED_EMITTED_CLASSES:
        if top1['class_name'] in KNOWN_CLASSES:
            return _result(
                MANUAL_REVIEW, 'mapping_gap_possible', detections, top1_idx)
        return _result(
            MANUAL_REVIEW, 'unsupported_class_mapping', detections,
            top1_idx)

    # top1 is a known, currently-emittable class (plastic_bottle/
    # paper_cup) -- decide purely on its confidence.
    confidence = top1['confidence']
    if confidence >= confidence_threshold:
        return _result(
            ACCEPT_SORT, 'known_high_confidence', detections, top1_idx)
    if confidence >= low_confidence_threshold:
        return _result(RETRY_VIEW, 'low_confidence', detections, top1_idx)
    return _result(
        ROUTE_TO_REJECT, 'low_confidence_reject', detections, top1_idx)
