
import pandas

dataset_path = '/home/bernardoferreira/Documents/brown/projects/shell_knock_down/shells_small.csv'


df = pandas.read_csv(dataset_path)

shell_ids = set(df.loc[:, 'shell_id'])
n_shells = len(shell_ids)

print(f'n_shells = {n_shells}')

shells_data = []

for shell_id in shell_ids:
    
    df_shell_id = df.loc[df['shell_id'] == shell_id]
    
    print(df_shell_id)
    
    shell_data = {}
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    shell_defects = []
    
    for i, row in df_shell_id.iterrows():
        
        defect_attr = ('defect_id', 'theta', 'phi', 'delta', 'lambda')
        defect = {}
        for key in defect_attr:
            defect[key] = row[key]
        
        shell_defects.append(defect)
        
    shell_data['defects'] = shell_defects
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    global_attr = ('shell_id', 'knock_down', 'radius', 'nu', 'eta')
    
    for key in global_attr:
        shell_data[key] = df.iloc[0, df_shell_id.columns.get_loc(key)]
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    shells_data.append(shell_data)