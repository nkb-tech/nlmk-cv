import re


def clean_str(s: str) -> str:
    """
    Cleans a string by replacing special characters with underscore _

    Args:
        s (str): a string needing special characters replaced

    Returns:
        (str): a string with special characters replaced by an underscore _
    """
    return re.sub(
        pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s
    )
