from src.agent_generator_langchain import generate_sql_from_explorer


class DummyExplorer:
    def retrieve(self, question):
        return {
            "tables": ["albums", "artists"],
            "reasoning": "Albums are linked to artists through artist_id.",
            "trace": [],
        }


class DummyGenerator:
    def __init__(self):
        self.calls = []

    def generate_sql_query(self, user_query, selected_tables, reasoning):
        self.calls.append((user_query, selected_tables, reasoning))
        return "SELECT * FROM albums JOIN artists ON albums.artist_id = artists.artist_id;"


def test_generate_sql_from_explorer_uses_explorer_result():
    explorer = DummyExplorer()
    generator = DummyGenerator()

    result = generate_sql_from_explorer(
        "List albums with their artist names.",
        explorer,
        generator,
    )

    assert result["tables"] == ["albums", "artists"]
    assert result["reasoning"] == "Albums are linked to artists through artist_id."
    assert result["sql"].startswith("SELECT")
    assert generator.calls[0][1] == ["albums", "artists"]
