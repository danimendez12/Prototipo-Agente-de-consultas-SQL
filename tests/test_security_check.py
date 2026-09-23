from src.services.security_check import evaluate_query


def test_allows_simple_select_query():
    ok, msg = evaluate_query("SELECT * FROM albums")
    assert ok is True
    assert msg == ""


def test_allows_cte_select_query():
    ok, msg = evaluate_query("WITH recent AS (SELECT * FROM albums WHERE album_id > 10) SELECT * FROM recent")
    assert ok is True
    assert msg == ""


def test_rejects_comment_based_injection():
    ok, msg = evaluate_query("SELECT * FROM users WHERE username = 'admin' -- comment")
    assert ok is False
    assert "injection" in msg.lower() or "comment" in msg.lower()


def test_rejects_boolean_injection_pattern():
    ok, msg = evaluate_query("SELECT * FROM users WHERE name = 'x' OR 1 = 1")
    assert ok is False
    assert "injection" in msg.lower() or "unsafe" in msg.lower()


def test_rejects_delete_statement():
    ok, msg = evaluate_query("DELETE FROM users")
    assert ok is False


def test_rejects_union_select_exfiltration():
    ok, msg = evaluate_query("SELECT * FROM users UNION SELECT * FROM admins")
    assert ok is False
