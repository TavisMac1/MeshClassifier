import unittest

from service.classifier import parse_classification


class ParseClassificationTests(unittest.TestCase):
    def test_parses_plain_json(self):
        result = parse_classification(
            '{"name":"Coral Branch","category":"Nature","description":"A branching coral."}'
        )
        self.assertEqual(result["name"], "Coral Branch")
        self.assertEqual(result["category"], "Nature")

    def test_parses_json_inside_prose(self):
        result = parse_classification(
            'Result: {"name":"Rock","category":"Geology","description":"A stone."} Done.'
        )
        self.assertEqual(result["name"], "Rock")

    def test_parses_fenced_json(self):
        result = parse_classification(
            '```json\n{"name":"Fish","category":"Animal","description":"A fish model."}\n```'
        )
        self.assertEqual(result["category"], "Animal")

    def test_rejects_missing_fields(self):
        with self.assertRaisesRegex(ValueError, "description"):
            parse_classification('{"name":"Fish","category":"Animal"}')

    def test_rejects_non_json(self):
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            parse_classification("not JSON")


if __name__ == "__main__":
    unittest.main()
