# import pandas as pd 
# data = {
#     'Name': ['Alice', 'Bob', 'Charlie'],
#     'Age': [25, 30, 35],
#     'City': ['NYC', 'LA', 'Chicago']
# }
# df = pd.DataFrame(data)
# print(df)
# import numpy as np
# numbers = np.array([1, 2, 3, 4, 5])
# print(numbers * 2)       
# print(numbers.mean())     
# print(numbers.sum()) 
# import random
# print (random.randint(1, 10))
# colours = ['red', 'green', 'blue', 'yellow']
# print(random.choice(colours))
# random.shuffle(colours)
# print(colours)
# print(random.random())

## all three together
# import pandas as pd
# import numpy as np
# import random
# data = {
#     'Product': ['A', 'B', 'C', 'D', 'E'],
#     'Sales': [random.randint(50, 200) for _ in range(5)],  # Random sales
#     'Price': np.array([10.99, 5.50, 15.75, 8.25, 20.00])   # NumPy array
# }
# df= pd.DataFrame(data)
# df['Revenue'] = df['Sales'] * df['Price']  # Calculate revenue
# print(df)