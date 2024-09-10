# empezando proyecto
# paso 1: hacer mi repositorio
# importando 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Paso 1: Crear los datos y guardar el archivo ventas_actualizadas.csv
data_updated = {
    'Fecha de Venta': [
        '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
        '2023-11-08', '2023-11-13', '2023-12-15', '2023-12-12', '2023-12-18',
        '2024-01-01', '2024-02-03', '2024-02-05', '2024-07-10', '2024-07-15',
        '2024-08-05', '2024-08-06', '2024-08-07', '2024-09-12', '2024-09-15'
    ],
    'Producto': [
        'Collar', 'Arete', 'Pulsera', 'Reloj', 'Anillo',
        'Brasalete', 'Llavero', 'Cartera', 'Correa', 'Bolso',   
        'Perfume', 'Tiara', 'Loción', 'Crema_Corporal', 'Shampoo',
        'Acondicionador', 'Exfoliante', 'Charms', 'Lentes', 'Ganchos'
    ],
    'Cantidad Vendida': [
        1250, 100, 400, 1, 180,
        165, 1, 800, 180, 299,
        185, 535, 1, 1, 95,
        95, 1, 80, 80, 25
    ],
    'Precio Unitario': [
        30.0, 25.0, 20.0, 50.0, 15.0,
        40.0, 45.99, 60.0, 20.0, 50.0,
        55.0, 60.0, 99.90, 99.90, 70.0,
        75.0, 75.50, 35.0, 45.0, 15.0
    ]
}

# Calcular el total de ventas y crear el DataFrame
data_updated['Total Venta'] = np.multiply(data_updated['Cantidad Vendida'], data_updated['Precio Unitario'])
df_updated = pd.DataFrame(data_updated)

# Guardar el DataFrame en un archivo CSV asegurando la correcta codificación y delimitador
df_updated.to_csv('C:/Users/HP/Documents/mi_tienda/ventas_actualizadas.csv', index=False, encoding='utf-8', sep=',')

print("El archivo 'ventas_actualizadas.csv' ha sido creado y guardado correctamente.")
