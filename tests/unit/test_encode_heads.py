"""Tests for model heads."""
from src.models.components.encode.heads import PresentHead, WhereHead


def test_where_head_dimensions(yolov4_preds):
    """Verify if output shape is correct when parsing model boxes prediction."""
    boxes, _ = yolov4_preds
    batch_size, n_anchors, *_ = boxes.shape
    where_head = WhereHead()
    z_where = where_head(boxes)
    assert z_where.shape == (batch_size, n_anchors, 4)


def test_present_head_dimensions(yolov4_preds):
    """Verify if outputs shape is correct when parsing model confidence prediction."""
    _, confs = yolov4_preds
    batch_size, n_anchors, *_ = confs.shape
    present_head = PresentHead()
    z_present = present_head(confs)
    assert z_present.shape == (batch_size, n_anchors, 1)
