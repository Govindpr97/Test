## widget incremental path
dbutils.widgets.text("input_file_path", "", "Input File Path")
dbutils.widgets.dropdown("trigger_flag", "False", ["True", "False"], "Trigger Flag")
dbutils.widgets.text("alternative_file_path", "abfss://histo@storage.dfs.core.windows.net/monthly_data/2026/Jan/Forecast_Jan'26.csv", "Alternative File Path")

input_file_path = dbutils.widgets.get("input_file_path")
trigger_flag = dbutils.widgets.get("trigger_flag")
alternative_path = dbutils.widgets.get("alternative_file_path")

print("input_file_path:", input_file_path)
print("trigger_flag:", trigger_flag)
print("alternative_path:", alternative_path)

if trigger_flag == "True":
    def list_files_recursive(path):
        items = dbutils.fs.ls(path)
        files = []
        for item in items:
            if item.isDir():
                if "__unitystorage" not in item.path and "models" not in item.path:
                    files.extend(list_files_recursive(item.path))
            else:
                files.append({"path": item.path, "modificationTime": item.modificationTime})
        return files
    all_files = list_files_recursive(input_file_path)
    import pandas as pd
    df_files = pd.DataFrame(all_files)
    df_files['modificationTime'] = pd.to_datetime(df_files['modificationTime'], unit='ms')
    latest_file_path = df_files.loc[df_files['modificationTime'].idxmax(), 'path']
    full_path = latest_file_path
else:
    full_path = alternative_path

display(full_path)
