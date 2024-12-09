import sqlite3

def create_connection(db_file):
    """Crea una conexión a la base de datos SQLite."""
    conn = sqlite3.connect(db_file)
    return conn

def create_tables(conn):
    """Crea las tablas necesarias en la base de datos."""
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS clientes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        region INTEGER,
        tenure INTEGER,
        age INTEGER,
        marital INTEGER,
        address INTEGER,
        income REAL,
        ed INTEGER,
        employ INTEGER,
        retire INTEGER,
        gender INTEGER,
        reside INTEGER
    )
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS recomendaciones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cliente_id INTEGER,
        servicio_recomendado TEXT,
        probabilidad REAL,
        FOREIGN KEY(cliente_id) REFERENCES clientes(id)
    )
    ''')
    
    conn.commit()

def insert_cliente(conn, cliente_data):
    """Inserta un nuevo cliente en la tabla clientes."""
    c = conn.cursor()
    c.execute("INSERT INTO clientes (region, tenure, age, marital, address, income, ed, employ, retire, gender, reside) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
              cliente_data)
    conn.commit()
    return c.lastrowid  # Devuelve el ID del cliente insertado

def insert_recomendacion(conn, cliente_id, servicio_recomendado):
    """Inserta una nueva recomendación en la tabla recomendaciones."""
    c = conn.cursor()
    c.execute("INSERT INTO recomendaciones (cliente_id, servicio_recomendado) VALUES (?, ?)",
              (cliente_id, servicio_recomendado))
    conn.commit()