import re

def clean_error_message(error_message):
    # Step 1: Remove ANSI escape sequences (color codes, formatting codes)
    ansi_escape = re.compile(r'\x1b\[[0-9;]*[mK]')
    cleaned_message = ansi_escape.sub('', error_message)

    # Step 2: Remove unnecessary repeated lines or lines with just dashes and metadata
    cleaned_message = re.sub(r'[-]+ stderr [-]+\n', '', cleaned_message)  # Remove stderr marker
    cleaned_message = re.sub(r'[-]+\n', '', cleaned_message)  # Remove repeated dashes
    cleaned_message = re.sub(r'Cell In\[\d+\], line \d+\n', '', cleaned_message)  # Remove notebook line references

    # Step 3: Remove paths to internal library files (optional)
    cleaned_message = re.sub(r'File .*site-packages/.*\n', '', cleaned_message)  # Remove Python package tracebacks

    # Step 4: Remove excessive whitespace lines
    cleaned_message = re.sub(r'\n\s*\n', '\n', cleaned_message)  # Remove extra newlines

    return cleaned_message.strip()