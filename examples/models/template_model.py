"""Copy to src/my_model/__init__.py, reinstall editable, then use my_model:predict.

This working starting point delegates to the cheap pixel baseline. Replace its
implementation with your method; keep the public request/output contract.
"""


def predict(request: dict, config: dict) -> list[dict]:
    # request: image path, image_size, public annotation boxes, full-image window.
    # Diagnostic mode may also supply legend/depth_calibration. Never open truth,
    # use a page ID to select stored answers, or train/tune during prediction.
    # Return sorted disjoint {top_y,bottom_y,lithology,cluster_id} rows. Both depth
    # endpoints are optional together; cluster_id is an integer within this page.
    # Keep options/checkpoint paths in config; declare training_groups there.
    from viridian402.models import baseline
    return baseline(request, config)
