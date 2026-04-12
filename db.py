import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
## Just data loading nothing more
# -----------------------------
# DB CONNECTION
# -----------------------------
conn = psycopg2.connect(
    dbname="integrated_project",
    user="postgres",
    password="root",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# -----------------------------
# COLUMN NAMES
# -----------------------------
cols = (
    ['unit', 'cycle'] +
    [f'op{i}' for i in range(1, 4)] +
    [f's{i}' for i in range(1, 22)]
)

# -----------------------------
# LOAD FUNCTION
# -----------------------------
def load_fd_to_db(fd_name):

    print(f"Loading {fd_name}...")

    # Read dataset
    df = pd.read_csv(f"train_{fd_name}.txt", sep=" ", header=None)
    df = df.dropna(axis=1)
    df.columns = cols

    records = []

    for _, row in df.iterrows():
        # Convert numpy types → Python types
        sensor_values = [float(x) for x in row[cols[2:]].values]

        record = (
            int(row['unit']),
            int(row['cycle']),
            *sensor_values,
            fd_name
        )

        records.append(record)

    # Build query dynamically
    query = f"""
    INSERT INTO engine_data
    (unit_id, cycle, {",".join(cols[2:])}, fd_type)
    VALUES (%s, %s, {",".join(["%s"]*(len(cols)-2))}, %s)
    """

    # Batch insert
    execute_batch(cursor, query, records, page_size=1000)

    conn.commit()
    print(f"{fd_name} loaded successfully ✅")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    for fd in ["FD002", "FD003", "FD004"]:
        load_fd_to_db(fd)

    print("All datasets loaded 🚀")

    cursor.close()
    conn.close()