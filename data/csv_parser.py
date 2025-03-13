import pandas as pd

class CSVParser:
    def __init__(self, file_path):
        """
        Initializes the CSVParser with the given file path.
        :param  file_path: Path to the CSV file.
        :type   file_path: str
        """
        try:
            self.df = pd.read_csv(file_path)
            self.headers = self.df.columns.tolist()
        except Exception as e:
            print(f"Error reading CSV file: {e}")

    def get_column(self, column_name):
        """
        Returns the data of the specified column.
        :param  column_name: Name of the column to retrieve.
        :type   column_name: str
        """
        try:
            return self.df[column_name].values
        except KeyError as e:
            print(f"Column not found: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    file_path = "data/data.csv"
    parser = CSVParser(file_path)
    print(parser.get_column("track_id"))
