
def item_element_country(df,item,element):
    '''function to take input of an item name, element name (e.g. Export/import value) and 
    output a formated file (aggregated by Reporter Countries) for downstream analysis'''
    
    data = df.loc[(df['Item']==item)&(df['Element']==element),:]
    print('%.f countries selected' %(data.shape[0]) )

    # Reshape data from wide to long by years 
    data_long = data.melt(['Reporter Countries'],years,'year','value')
    
    # Convert data to time series
    data_long['year'] = data_long['year'].map(lambda x: x.lstrip('Y')) # strip Y from year names for easy converting to ts
    data_long.year = pd.to_datetime(data_long.year)
    
    # Reshape data from long to wide, turn countries into columns
    data_wide = data_long.pivot(index='year',columns='Reporter Countries',values='value')

    return data_wide
