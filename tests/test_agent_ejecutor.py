import pytest

from src.agent_ejecutor import SecurityError, validate_readonly


def test_allows_simple_select_query():
    validate_readonly("SELECT * FROM Track")


def test_allows_cte_select_query():
    validate_readonly("WITH recent AS (SELECT * FROM Track WHERE TrackId > 10) SELECT * FROM recent")


def test_rejects_write_keywords_case_insensitive():
    with pytest.raises(SecurityError):
        validate_readonly("delete from Track")


def test_rejects_multiple_statements():
    with pytest.raises(SecurityError):
        validate_readonly("SELECT * FROM Track; DELETE FROM Track")


def test_rejects_empty_query():
    with pytest.raises(SecurityError):
        validate_readonly("   ")


def test_does_not_block_keyword_inside_string_literal():
    validate_readonly("SELECT * FROM Track WHERE Name = 'DELETE me'")
