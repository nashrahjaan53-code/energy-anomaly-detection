##from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
##encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
##data = np.array(['Red', 'Blue', 'Green', 'Red']).reshape(-1, 1)
#print(data)
##encoded_data = encoder.fit_transform(data)
##print("\nEncoded Data:")
# import pandas as pd
# data = {
#     'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue'],
#     'Size': ['S', 'M', 'L', 'S', 'M']
# }
# df = pd.DataFrame(data)
# print("Original DataFrame:")
# print(df)
# encoded_df = pd.get_dummies(df, columns=['Color', 'Size'])
# print("\nOne-Hot Encoded DataFrame:")
# print(encoded_df)

import pandas as pd 
data = {
    'Fruit': ['Apple', 'Banana', 'Orange', 'Apple'],
    'Price': [1.2, 0.5, 0.8, 1.3]
}
df=pd.DataFrame(data)
print("Original DataFrame:")
print(df)
encoded_df = pd.get_dummies(df, columns=['Fruit']).astype(int)
print("\nOne-Hot Encoded DataFrame:")
print(encoded_df)