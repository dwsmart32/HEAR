def sanitize_filename(name):
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")
