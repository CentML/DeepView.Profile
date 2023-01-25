import os
import random
import skyline.db.database as database

class MockDatabaseInterface(database.DatabaseInterface):
    def __del__(self):
        if os.path.exists("test.sqlite"):
            os.remove("test.sqlite")


class TestSkylineDatabase:
    test_database: MockDatabaseInterface = MockDatabaseInterface("test.sqlite")
    energy_table_interface: database.EnergyTableInterface = database.EnergyTableInterface(test_database.connection)

    # Test if energy table is created
    def test_energy_table_is_created(self):
        query_result = self.test_database.connection.execute("SELECT name from sqlite_schema WHERE type='table' and name ='ENERGY';")
        query_result_list = query_result.fetchall()
        assert(len(query_result_list) > 0)

    # try adding invalid entry and test if it is added
    def test_invalid_entry_too_short(self):
        assert(self.energy_table_interface.is_valid_entry([]) == False)
    
    def test_invalid_entry_too_long(self):
        assert(self.energy_table_interface.is_valid_entry([1,2,3,4]) == False)

    def test_invalid_entry_wrong_types(self):
        assert(self.energy_table_interface.is_valid_entry([None, None, None, None, None]) == False)

    def test_adding_valid_entry(self):
        params = ["entry_point", random.random(), random.random()]
        self.energy_table_interface.add_entry(params)
        query_result = self.test_database.connection.execute("SELECT * FROM ENERGY;").fetchone()
        # params is passed in by reference so it have the timestamp in it
        assert(query_result == tuple(params))

    # add 10 valid entries and get top 3
    def test_get_latest_n_entries_of_entry_point(self):
        for _ in range(10):
            params = ["entry_point", random.random(), random.random()]
            self.energy_table_interface.add_entry(params)
        for _ in range(20):
            params = ["other_entry_point", random.random(), random.random()]
            self.energy_table_interface.add_entry(params)
        entries = []
        for _ in range(3):
            params = ["entry_point", random.random(), random.random()]
            entries.insert(0, params)
            self.energy_table_interface.add_entry(params)
        latest_n_entries = self.energy_table_interface.get_latest_n_entries_of_entry_point(3, "entry_point")
        entries = [tuple(entry) for entry in entries]
        assert(entries == latest_n_entries)
