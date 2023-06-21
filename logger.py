"""
This module contains on the one hand the standard log function, 
which writes the individual steps of the program into a log file 
and on the other hand functions, which store the results in an Excel file.
"""

import logging
import re
import os
import pandas as pd
import datetime
from pathlib import Path
import utils as ut


def logging_setup(
    module_name: str = "default", filename: str = "log.txt"
) -> logging.Logger:
    """
    Setup a logger for the specified module
    """
    if module_name == "__main__":
        clear_log_file(filename)
    # Set up logging to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Set up logging to file
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.INFO)

    # Create a logger and set the logging level
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    # Add the console and file handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # create a formatter and add it to the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    return logger


def clear_log_file(filename: str) -> None:
    """
    Empties log file
    if file doesnt exist, it will be created
    """
    with open(filename, "w") as log_file:
        pass
    return


# Log to excel
def save_pred(command: str, average: float, stddev: float, rand: float) -> None:
    """
    Saves Output to an Excel file with the help of the append_to_excel function.
    """
    parts = ut.active_abbrs()
    n = ut.get_n_from_config()
    today = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    df = pd.DataFrame(
        {
            "date": [today],
            "command": [command],
            "Average": [average],
            "Standard Deviation": [stddev],
            "random": [rand],
            "Number Instances": [n],
            "Features": [parts],
        }
    )

    filename = "Runs.xlsx"
    append_to_excel(df, filename)
    return


def append_to_excel(df: pd.DataFrame, filename: str):
    """
    Saves Output in Excel file.
    File will be created if it doesnt exist,
    otherwise the output will be appended
    """
    # Check if the file exists
    if not os.path.isfile(filename):
        # If it doesn't exist, create the Excel file and write the data
        df.to_excel(filename, index=False)
    else:
        # If the file exists, load the existing data, append the new data, and save
        df_existing = pd.read_excel(filename)
        df_total = pd.concat([df_existing, df])
        df_total.to_excel(filename, index=False)
    return
