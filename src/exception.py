import sys


def error_message_deatils(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    linenumber = exc_tb.tb_lineno
    error_message = f"Error occured in python script name [{filename}] line number [{linenumber}] error message[{str(error)}]"
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)
        self.error_message = error_message_deatils(error_message, error_details)

    def __str__(self):
        return self.error_message
