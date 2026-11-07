from pathlib import Path

from evals.datasets.cord import IMAGES_DIR, image_path


def test_an_exported_image_is_named_after_its_row_id() -> None:
    assert image_path("train-00019") == IMAGES_DIR / "train-00019.png"
    assert IMAGES_DIR == Path(__file__).parents[3] / "data" / "cord_images"
