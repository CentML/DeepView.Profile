import datetime
import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "deepview.sqlite3")

class DatabaseInterface:
    def __init__(self, database_name=DB_PATH) -> None:
        self.connection = sqlite3.connect(
            database_name, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES
        )
        self.create_energy_table()

    def create_energy_table(self) -> None:
        self.connection.cursor().execute("CREATE TABLE IF NOT EXISTS ENERGY ( \
            entry_point TEXT, \
            cpu_component REAL, \
            gpu_component REAL, \
            batch_size INT, \
            ts TIMESTAMP \
        );")


class EnergyTableInterface:
    def __init__(self, database_connection: sqlite3.Connection):
        self.database_connection: sqlite3.Connection = database_connection

    @staticmethod
    def is_valid_entry(entry: list) -> bool:
        '''
        Validates an entry in the Energy table by testing if the length is 3,
        and the types match the columns. Note that timestamp is not part of the entry.
        Returns True if it is valid, else False
        '''
        return len(entry) == 4 and type(entry[0]) == str and type(entry[1]) == float \
            and type(entry[2]) == float and type(entry[3]) == int

    @staticmethod
    def is_valid_entry_with_timestamp(entry: list) -> bool:
        '''
        Validates an entry in the Energy table by testing if the length is 4,
        and the types match the columns. Returns True if it is valid, else False
        '''
        return len(entry) == 5 and type(entry[0]) == str and type(entry[1]) == float \
            and type(entry[2]) == float and type(entry[3]) == int \
            and type(entry[4]) == datetime.datetime

    def add_entry(self, entry: list) -> bool:
        '''
        Validates an entry and then adds that entry into the Energy table. Note that
        current timestamp is added by this function. Returns False if the entry is
        not a valid format, or if the insertion failed. Else returns True
        '''
        if self.is_valid_entry(entry):
            try:
                entry.append(datetime.datetime.now())
                cursor = self.database_connection.cursor()
                cursor.execute("INSERT INTO ENERGY VALUES(?, ?, ?, ?, ?)", entry)
                self.database_connection.commit()
                return True
            except sqlite3.IntegrityError as e:
                print(e)
                return False
        else:
            return False

    def get_latest_n_entries_of_entry_point(self, n: int, entry_point: str) -> list:
        '''
        Gets the n latest entries of a given entry point
        '''
        params = [entry_point, n]
        cursor = self.database_connection.cursor()
        results = cursor.execute(
            "SELECT * FROM ENERGY WHERE entry_point=? ORDER BY ts DESC LIMIT ?;",
            params
        ).fetchall()
        return results
