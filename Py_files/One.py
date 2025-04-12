import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import seaborn as sns

data = pd.read_csv('datasets/daily-total-female-births-in-cal.csv', 
                   delimiter=';',
                   parse_dates=['Date'])

data.set_index('Date', inplace=True)

data['Count'] = data['Count'].str.replace(',', '.', regex=True)
data['Count'] = data['Count'].astype(float)

result = adfuller(data['Count'])

print('Тест Дики-Фуллера:')
print(f'{result[0]}')
print(f'p-value: {result[1]}')
print('Критические значения:')
for key, value in result[4].items():
    print(f'   {key}: {value}')

if result[1] <= 0.05:
    print("Ряд стационарен (отвергаем нулевую гипотезу).")
else:
    print("Ряд нестационарен (не можем отвергнуть нулевую гипотезу).")

plt.style.use('dark_background')

plt.figure(figsize=(10, 6))
sns.lineplot(
    x=data.index,
    y=data['Count'],
    label='Количество рождений',
    color='cyan', 
    linewidth=2,
    marker='o',  
    markersize=4,
    markerfacecolor='white', 
    markeredgecolor='cyan'    
)

plt.fill_between(data.index, data['Count'], color='cyan', alpha=0.2) 

plt.title('Ежедневное количество женских рождений', fontsize=16, color='white') 
plt.ylabel('Количество рождений', fontsize=12, color='white')

plt.xticks(rotation=45, color='white') 
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(15))  

plt.legend(fontsize=10, labelcolor='white') 
plt.grid(True, linestyle='--', alpha=0.6, color='gray')

plt.tight_layout()

plt.savefig("female_births_plot.png", dpi=300, bbox_inches='tight')  
plt.close()
plt.show()