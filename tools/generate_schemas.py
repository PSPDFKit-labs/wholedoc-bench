import json
from pathlib import Path

from wholedoc_bench.corpus.release import release_schema
from wholedoc_bench.schema import prediction_schema


def main() -> None:
    destination = Path(__file__).resolve().parents[1] / "schemas"
    destination.mkdir(exist_ok=True)
    for name, schema in {**release_schema(), **prediction_schema()}.items():
        schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema}
        (destination / name).write_bytes((json.dumps(schema, indent=2) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
