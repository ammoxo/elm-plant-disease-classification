def image_process_log(success, fails, invalid):
    print(f"{success} image processed with {fails} failures and {invalid} invalids...", end="\r", flush=True)