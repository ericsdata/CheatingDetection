
def recommended_sample_fraction(df):
     
    import psutil 

    df_size = df.memory_usage(deep=True).sum() 
    available_ram = psutil.virtual_memory().available 
    safe_limit = available_ram / 5 # ydata needs ~5× RAM 
    if df_size <= safe_limit: 
        return 1.0 # full dataset is safe 
     
    return safe_limit / df_size 