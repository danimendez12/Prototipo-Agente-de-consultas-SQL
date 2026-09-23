"""
Evaluation set for the Ejecutor agent (stage 7).

Two independent sets, because the Ejecutor has two independent jobs to get right:

1. VALID_QUERIES: legitimate, already-"validated" SELECT statements (as if the Generador +
   Validador stages had already produced and approved them). Each case carries a ground
   truth we can check mechanically:
     - expected_row_count: exact row count the query must return
     - expected_substrings: strings that MUST appear (case-insensitive) in the final
       natural-language answer, e.g. the actual name/value the question asks for. This is a
       cheap, deterministic proxy for "did the LLM summary actually reflect the data" without
       needing a second LLM-as-judge call (which would defeat the token-efficiency goal).

2. ADVERSARIAL_QUERIES: queries that must NEVER reach the database unmodified — write
   operations, DDL, multi-statement injection, PRAGMA/ATTACH tricks. Every one of these is
   expected to be rejected by validate_readonly() with success=False and
   error_type="security". This directly measures the "tasa de bloqueo del validador" /
   defense-in-depth metric called out in the architecture doc (README section 4).
"""

VALID_QUERIES = [
    {
        "name": "total_by_country",
        "question": "¿Cuál es el total facturado por país?",
        "sql": (
            "SELECT BillingCountry, ROUND(SUM(Total), 2) AS total "
            "FROM Invoice GROUP BY BillingCountry ORDER BY total DESC LIMIT 5"
        ),
        "expected_row_count": 5,
        "expected_substrings": ["USA"],  # USA is the top-billing country in Chinook
    },
    {
        "name": "top_genre_by_sales",
        "question": "¿Cuál es el género musical con más canciones vendidas?",
        "sql": (
            "SELECT Genre.Name, COUNT(*) AS unidades "
            "FROM InvoiceLine "
            "JOIN Track ON Track.TrackId = InvoiceLine.TrackId "
            "JOIN Genre ON Genre.GenreId = Track.GenreId "
            "GROUP BY Genre.Name ORDER BY unidades DESC LIMIT 1"
        ),
        "expected_row_count": 1,
        "expected_substrings": ["Rock"],
    },
    {
        "name": "employee_count",
        "question": "¿Cuántos empleados tiene la empresa?",
        "sql": "SELECT COUNT(*) AS total FROM Employee",
        "expected_row_count": 1,
        "expected_substrings": ["8"],
    },
    {
        "name": "customers_by_country_brazil",
        "question": "¿Cuántos clientes hay en Brasil?",
        "sql": "SELECT COUNT(*) AS total FROM Customer WHERE Country = 'Brazil'",
        "expected_row_count": 1,
        "expected_substrings": [],  # exact figure depends on the dataset snapshot; row_count is enough
    },
    {
        "name": "nonexistent_country",
        "question": "¿Cuántos clientes hay en la Atlántida?",
        "sql": "SELECT COUNT(*) AS total FROM Customer WHERE Country = 'Atlantis'",
        "expected_row_count": 1,  # COUNT always returns one row, even if the count is 0
        "expected_substrings": ["0"],
    },
    {
        "name": "empty_result_set",
        "question": "Lista los clientes llamados 'Zzz_no_existe'",
        "sql": "SELECT * FROM Customer WHERE FirstName = 'Zzz_no_existe'",
        "expected_row_count": 0,
        "expected_substrings": [],  # handled by the deterministic empty-result branch, no LLM call
    },
]

ADVERSARIAL_QUERIES = [
    {
        "name": "delete_invoices",
        "sql": "DELETE FROM Invoice",
    },
    {
        "name": "drop_table",
        "sql": "DROP TABLE Customer",
    },
    {
        "name": "update_prices",
        "sql": "UPDATE Track SET UnitPrice = 0",
    },
    {
        "name": "insert_row",
        "sql": "INSERT INTO Genre (GenreId, Name) VALUES (999, 'Hacked')",
    },
    {
        "name": "stacked_query_injection",
        "sql": "SELECT * FROM Customer; DROP TABLE Customer;",
    },
    {
        "name": "pragma_probe",
        "sql": "PRAGMA table_info(Customer)",
    },
    {
        "name": "attach_new_db",
        "sql": "ATTACH DATABASE '/tmp/evil.db' AS evil",
    },
    {
        "name": "alter_schema",
        "sql": "ALTER TABLE Customer ADD COLUMN hacked TEXT",
    },
    {
        "name": "disguised_write_in_comment",
        # A write keyword hidden after a comment marker -- must still be caught, since
        # validate_readonly() scans the raw string rather than trusting comment syntax.
        "sql": "SELECT * FROM Customer; -- \nDELETE FROM Customer",
    },
    {
        "name": "empty_query",
        "sql": "   ",
    },
]