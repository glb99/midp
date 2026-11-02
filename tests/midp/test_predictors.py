from decimal import Decimal

from PIL import Image

from midp.predictors import MultimodalPredictor, parse_response

FIELDS = ["lines", "subtotal", "tax", "total"]


def test_multimodal_predictor_reads_the_model_answer() -> None:
    answer = (
        '{"lines": [{"description": "Nasi", "quantity": "1", "amount": "75000"}],'
        ' "subtotal": "75000", "tax": null, "total": "82500"}'
    )
    predictor = MultimodalPredictor(model=lambda image, prompt: answer, fields=FIELDS)

    result = predictor.predict(Image.new("RGB", (4, 4)))

    assert result["lines"] == [{"description": "Nasi", "quantity": 1, "amount": Decimal(75000)}]
    assert result["subtotal"] == Decimal(75000)
    assert result["tax"] is None
    assert result["total"] == Decimal(82500)


def test_parse_response_accepts_a_code_fence() -> None:
    fenced = '```json\n{"total": "5"}\n```'

    assert parse_response(fenced)["total"] == Decimal(5)
