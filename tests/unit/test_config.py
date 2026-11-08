from pathlib import Path

from midp.config import load_document_types, load_type_config

CONFIGS = Path(__file__).parents[2] / "configs"


def test_every_registered_type_has_a_config_for_that_type() -> None:
    types = load_document_types(CONFIGS / "document_types.yaml")

    assert [entry["name"] for entry in types] == ["receipt", "invoice", "delivery_note"]
    for entry in types:
        assert entry["config"].is_file()
        assert load_type_config(entry["config"])["document_type"] == entry["name"]
