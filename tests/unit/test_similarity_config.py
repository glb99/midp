from pathlib import Path

from midp.similarity import load_similarity_config

CONFIG = Path(__file__).parents[2] / "configs" / "similarity.yaml"


def test_the_similarity_settings_are_read_from_their_config() -> None:
    settings = load_similarity_config(CONFIG)

    assert settings == {"k": 3}
