import json
from pathlib import Path

import yaml

from evals.datasets.cord import to_reference

# A trimmed CORD row, with values copied from the first train example.
ROW = {
    "ground_truth": json.dumps(
        {
            "gt_parse": {
                "menu": [
                    {"nm": "Nasi Campur Bali", "cnt": "1 x", "price": "75,000"},
                    {"nm": "Tahu Telor Asin", "cnt": "1 x", "price": "40,000."},
                ],
                "sub_total": {
                    "subtotal_price": "1,346,000",
                    "service_price": "100,950",
                    "tax_price": "144,695",
                    "etc": "-45",
                },
                "total": {"total_price": "1,591,600"},
            }
        }
    )
}


def test_mapping_renames_cord_labels_and_parses_numbers() -> None:
    reference = to_reference(ROW)

    assert reference["lines"] == [
        {"description": "Nasi Campur Bali", "quantity": 1, "amount": 75000},
        {"description": "Tahu Telor Asin", "quantity": 1, "amount": 40000},
    ]
    assert reference["total"] == 1591600
    assert reference["rounding"] == -45


def test_config_and_mapping_name_the_same_fields() -> None:
    config = yaml.safe_load(
        (Path(__file__).parents[3] / "configs" / "receipts.yaml").read_text(encoding="utf-8")
    )
    config_names = {field["name"] for field in config["fields"]}

    assert config_names == set(to_reference(ROW))
