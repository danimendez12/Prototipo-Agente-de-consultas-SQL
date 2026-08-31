"""
Evaluation set with known ground truth (ground truth) to measure the quality of the Explorer's
retrieval over Chinook.

Each entry: a natural-language question plus the tables a human determined were necessary to
answer it correctly.
"""


EVAL_SET = [
    # 1
    {
        "question": "¿Cuáles son los 5 géneros musicales con más canciones vendidas?",
        "expected_tables": {"Genre", "Track", "InvoiceLine"},
    },

    # 2
    {
        "question": "¿Qué empleado tiene más clientes asignados?",
        "expected_tables": {"Employee", "Customer"},
    },

    # 3
    {
        "question": "Lista las canciones de la playlist más grande",
        "expected_tables": {"Playlist", "PlaylistTrack", "Track"},
    },

    # 4
    {
        "question": "¿Cuál es el total facturado por país?",
        "expected_tables": {"Invoice"},
    },

    # 5
    {
        "question": "¿Qué artistas tienen más de 5 álbumes?",
        "expected_tables": {"Artist", "Album"},
    },

    # 6
    {
        "question": "¿Cuántas canciones tiene cada tipo de formato de audio?",
        "expected_tables": {"MediaType", "Track"},
    },

    # 7
    {
        "question": "¿Cuál es el cliente que más ha gastado en total?",
        "expected_tables": {"Customer", "Invoice"},
    },

    # 8
    {
        "question": "¿Quién es el jefe directo de cada empleado?",
        "expected_tables": {"Employee"},
    },

    # 9
    {
        "question": "¿Cuáles son las canciones más largas en duración?",
        "expected_tables": {"Track"},
    },

    # 10
    {
        "question": "¿Cuántas canciones compró cada cliente en su factura más reciente?",
        "expected_tables": {"Customer", "Invoice", "InvoiceLine"},
    },

    # 11
    {
        "question": "¿Cuál es el álbum con más canciones?",
        "expected_tables": {"Album", "Track"},
    },

    # 12
    {
        "question": "¿Qué artista tiene más canciones en total?",
        "expected_tables": {"Artist", "Album", "Track"},
    },

    # 13
    {
        "question": "¿Cuál es el género con mayor duración total de canciones?",
        "expected_tables": {"Genre", "Track"},
    },

    # 14
    {
        "question": "¿Cuántas canciones hay por género?",
        "expected_tables": {"Genre", "Track"},
    },

    # 15
    {
        "question": "¿Cuál es la canción que más veces ha sido comprada?",
        "expected_tables": {"Track", "InvoiceLine"},
    },

    # 16
    {
        "question": "¿Cuáles son los clientes que nunca han realizado una compra?",
        "expected_tables": {"Customer", "Invoice"},
    },

    # 17
    {
        "question": "¿Qué país tiene el mayor número de clientes?",
        "expected_tables": {"Customer"},
    },

    # 18
    {
        "question": "¿Cuál es el promedio gastado por cliente?",
        "expected_tables": {"Customer", "Invoice"},
    },

    # 19
    {
        "question": "¿Qué empleados atienden a clientes de más de un país?",
        "expected_tables": {"Employee", "Customer"},
    },

    # 20
    {
        "question": "¿Cuál es la playlist que contiene más canciones?",
        "expected_tables": {"Playlist", "PlaylistTrack"},
    },

    # 21
    {
        "question": "¿Qué canciones aparecen en más de una playlist?",
        "expected_tables": {"Track", "PlaylistTrack"},
    },

    # 22
    {
        "question": "¿Cuál es el artista con mayor cantidad de canciones vendidas?",
        "expected_tables": {"Artist", "Album", "Track", "InvoiceLine"},
    },

    # 23
    {
        "question": "¿Cuánto dinero se ha generado por cada género musical?",
        "expected_tables": {"Genre", "Track", "InvoiceLine"},
    },

    # 24
    {
        "question": "¿Cuál es el tipo de medio más utilizado entre las canciones vendidas?",
        "expected_tables": {"MediaType", "Track", "InvoiceLine"},
    },

    # 25
    {
        "question": "¿Cuál es el álbum más vendido?",
        "expected_tables": {"Album", "Track", "InvoiceLine"},
    },

    # 26
    {
        "question": "¿Qué clientes han gastado más de 50 dólares?",
        "expected_tables": {"Customer", "Invoice"},
    },

    # 27
    {
        "question": "¿Cuál es la factura con mayor cantidad de canciones?",
        "expected_tables": {"Invoice", "InvoiceLine"},
    },

    # 28
    {
        "question": "¿Qué empleados tienen asignados más de 10 clientes?",
        "expected_tables": {"Employee", "Customer"},
    },

    # 29
    {
        "question": "¿Cuáles son los 10 artistas cuyas canciones han generado más ingresos?",
        "expected_tables": {"Artist", "Album", "Track", "InvoiceLine"},
    },

    # 30
    {
        "question": "¿Cuál es el género más popular entre las canciones compradas por los clientes de cada país?",
        "expected_tables": {"Customer", "Invoice", "InvoiceLine", "Track", "Genre"},
    },

    {
        "question": "¿Qué clientes tienen como representante a un empleado que fue contratado antes de 2003?",
        "expected_tables": {"Customer", "Employee"},
    },
    {
        "question": "¿Qué canciones cuestan más que el precio promedio de todas las canciones?",
        "expected_tables": {"Track"},
    },
    {
        "question": "¿Qué artistas tienen al menos un álbum cuyo título contiene la palabra 'Live'?",
        "expected_tables": {"Artist", "Album"},
    },
    {
        "question": "¿Qué empleados tienen clientes cuyo país de facturación no coincide con el país registrado del cliente?",
        "expected_tables": {"Employee", "Customer", "Invoice"},
    },
    {
        "question": "¿Qué canciones pertenecen a álbumes de artistas que no tienen ninguna canción incluida en una playlist?",
        "expected_tables": {"Artist", "Album", "Track", "PlaylistTrack"},
    },
    {
        "question": "¿Qué clientes realizaron compras de canciones pertenecientes a más de 3 géneros diferentes?",
        "expected_tables": {"Customer", "Invoice", "InvoiceLine", "Track", "Genre"},
    },
    {
        "question": "¿Qué canciones tienen una duración superior al promedio de duración de su propio género?",
        "expected_tables": {"Track", "Genre"},
    },
    {
        "question": "¿Qué artistas tienen canciones que aparecen en playlists pero nunca fueron compradas?",
        "expected_tables": {"Artist", "Album", "Track", "PlaylistTrack", "InvoiceLine"},
    },
    {
        "question": "¿Qué clientes compraron canciones de un artista en más de una factura diferente?",
        "expected_tables": {"Customer", "Invoice", "InvoiceLine", "Track", "Album", "Artist"},
    },
    {
        "question": "¿Qué formatos de audio tienen canciones que nunca han sido incluidas en ninguna factura?",
        "expected_tables": {"MediaType", "Track", "InvoiceLine"},
    },
]