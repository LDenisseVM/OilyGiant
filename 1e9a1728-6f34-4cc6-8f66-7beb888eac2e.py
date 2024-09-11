#!/usr/bin/env python
# coding: utf-8

# # ¡Hola, Denisse!  
# 
# Mi nombre es Carlos Ortiz, soy code reviewer de TripleTen y voy a revisar el proyecto que acabas de desarrollar.
# 
# Cuando vea un error la primera vez, lo señalaré. Deberás encontrarlo y arreglarlo. La intención es que te prepares para un espacio real de trabajo. En un trabajo, el líder de tu equipo hará lo mismo. Si no puedes solucionar el error, te daré más información en la próxima ocasión. 
# 
# Encontrarás mis comentarios más abajo - **por favor, no los muevas, no los modifiques ni los borres**.
# 
# ¿Cómo lo voy a hacer? Voy a leer detenidamente cada una de las implementaciones que has llevado a cabo para cumplir con lo solicitado. Verás los comentarios de esta forma:
# 
# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Si todo está perfecto.
# </div>
# 
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Si tu código está bien pero se puede mejorar o hay algún detalle que le hace falta.
# </div>
# 
# 
# <div class="alert alert-block alert-danger">
#     
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
#     
# Si de pronto hace falta algo o existe algún problema con tu código o conclusiones.
# </div>
# 
# 
# Puedes responderme de esta forma: 
# 
# 
# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class="tocSkip"></a>
# </div>
# ¡Empecemos!

# # Descripción del proyecto:
# Para la extracción de petróleo de la empresa OilyGiant, encontraremos los mejores lugares donde abrir 200 pozos nuevos de petróleo.
# 
# Indice:
# Importación de librerías 
# Carga de datos.
# Creación de un modelo para predecir el volumen de reservas en pozos nuevos.
# Elección de los pozos petrolíferos que tienen los valores estimados más altos.
# Cálculo de riegos
# Elección de la región con el beneficio total más alto para los pozos petrolíferos seleccionados.

# # Importación de librerías:

# In[155]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


# # Carga de datos

# In[156]:


data0= pd.read_csv('/datasets/geo_data_0.csv')


# In[157]:


data0.head()


# In[158]:


data0.info()


# In[159]:


data1= pd.read_csv('/datasets/geo_data_1.csv')


# In[160]:


data1.head()


# In[161]:


data1.info()


# In[162]:


data2= pd.read_csv('/datasets/geo_data_2.csv')


# In[163]:


data2.head()


# In[164]:


data2.info()


# Compararemos las columnas de los distintos datasets con histogramas

# In[165]:


data0['f0'].hist(bins=10, alpha=.4)
data1['f0'].hist(bins=10, alpha=.4)
data2['f0'].hist(bins=10, alpha=.4)


# In[166]:


data0['f1'].hist(bins=10, alpha=.4)
data1['f1'].hist(bins=10, alpha=.4)
data2['f1'].hist(bins=10, alpha=.4)


# In[167]:


data0['f2'].hist(bins=10, alpha=.4)
data1['f2'].hist(bins=10, alpha=.4)
data2['f2'].hist(bins=10, alpha=.4)


# Analicemos las relaciones entre las columnas en las distintas regiones 

# In[168]:


pd.plotting.scatter_matrix(data0, figsize=(15,10))


# In[169]:


pd.plotting.scatter_matrix(data1, figsize=(15,10))


# In[170]:


pd.plotting.scatter_matrix(data2, figsize=(15,10))


# Vemos muy distintas relaciones entre columnas, dependiendo de la región

# # Preparación de los datos:

# In[171]:


def split_data(data):
    
    data=data.drop(['id'], axis=1)
    
    features= data.drop(['product'], axis=1)
    
    scaler=StandardScaler()
    scaler.fit(features)
    features= scaler.transform(features)
    target=data['product']
    
    features_train, features_valid, target_train, target_valid= train_test_split(features, target, test_size=.25, random_state=(12345))

    
    return features_train, features_valid, target_train, target_valid


# In[172]:


features_train_0, features_valid_0, target_train_0, target_valid_0 = split_data(data0)


# In[173]:


features_train_1, features_valid_1, target_train_1, target_valid_1 = split_data(data1)


# In[174]:


features_train_2, features_valid_2, target_train_2, target_valid_2 = split_data(data2)


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Buen trabajo.
# </div>

# # Creación de modelo, predicción promedio, y RECM

# In[175]:


def model_pred(features_train, features_valid, target_train, target_valid):
    
    model=LinearRegression()
    
    model.fit(features_train, target_train)
    predictions=model.predict(features_valid)
    
    average_predictions= predictions.mean()
    
    RECM= mean_squared_error(target_valid, predictions, squared=False)
    
    return model, predictions, RECM, average_predictions


# <div class="alert alert-block alert-info">
# <b>Me parece que es: mean_squared_error(x, y, squared=False)</b> <a class="tocSkip"></a>
# </div>

# In[176]:


model_0, predictions_0, RECM_0, average_predictions_0= model_pred(features_train_0, features_valid_0, target_train_0, target_valid_0)


# In[177]:


model_1, predictions_1, RECM_1, average_predictions_1= model_pred(features_train_1, features_valid_1, target_train_1, target_valid_1)


# In[178]:


model_2, predictions_2, RECM_2, average_predictions_2= model_pred(features_train_2, features_valid_2, target_train_2, target_valid_2)


# In[215]:


print(predictions_0)
print(RECM_0)
print(average_predictions_0)


# In[216]:


print(predictions_1)
print(RECM_1)
print(average_predictions_1)


# In[217]:


print(predictions_2)
print(RECM_2)
print(average_predictions_2)


# In[ ]:





# Para calcular las unidades mínimas que cada pozo debería tener para evitar pérdidas:
#     Dada la inversión de 100 millones por 200 pozos petrolíferos, de media un pozo petrolífero debe producir al menos un valor de 500,000 dólares en unidades para evitar pérdidas (esto es equivalente a 111.1 unidades)

# In[182]:


100000000/200

pozos = 200
presupuesto_total= 100000000
USD_por_unidad = 4500

presupuesto_por_pozo = presupuesto_total/pozos

unidades_mínimas = presupuesto_por_pozo/USD_por_unidad

unidades_mínimas


# In[183]:


averages=[average_predictions_0, average_predictions_1, average_predictions_2, unidades_mínimas]


# In[184]:


etiquetas=['average_predictions_0', 'average_predictions_1', 'average_predictions_2', 'unidades mínimas']
plt.bar(etiquetas, averages)
plt.xticks(rotation=45)
plt.title('Average predictions')


# # Top 200 predictions

# In[185]:


def top_pred(predictions):
    top_predictions= pd.Series(predictions)

    top_predictions=top_predictions.sort_values(ascending=False)
    top_predictions=top_predictions.head(200)
    return top_predictions


# In[186]:


top_predictions_0= top_pred(predictions_0)
top_predictions_1= top_pred(predictions_1)
top_predictions_2= top_pred(predictions_2)


# In[187]:


top_predictions_0.mean()


# In[188]:


top_predictions_1.mean()


# In[189]:


top_predictions_2.mean()


# In[ ]:





# In[190]:


top_averages=[top_predictions_0.mean(), top_predictions_1.mean(), top_predictions_2.mean(), 111.1]


# In[191]:


etiquetas=['average_top_predictions_0', 'average_top_predictions_1', 'average_top_predictions_2', 'unidades mínimas']
plt.bar(etiquetas, top_averages)
plt.xticks(rotation=45)
plt.title('Average top predictions')


# In[ ]:





# Vamos a sacar de target_valid, los valores reales del top 200 de cada una de las regiones, según las predicciones 

# In[192]:


def top_target(top_predictions, target_valid):
    target_valid = target_valid.reset_index(drop=True)
    top_target=target_valid.iloc[top_predictions.index]
    return top_target


# In[193]:


top_target_0= top_target(top_predictions_0, target_valid_0)
top_target_0


# In[194]:


top_target_1= top_target(top_predictions_1, target_valid_1)
top_target_1


# In[195]:


top_target_2= top_target(top_predictions_2, target_valid_2)
top_target_2


# In[ ]:





# # Calculamos la ganancia según las predicciones con el top 200 de pozos 

# In[196]:


budget=100000000
unit= 4500
def ganancias(top_predictions):
    suma= top_predictions.sum()
    ganancia= suma*unit-budget
    
    return ganancia


# In[197]:


ganancia_predictions_0=ganancias(top_predictions_0)
ganancia_predictions_0


# In[198]:


ganancia_predictions_1=ganancias(top_predictions_1)
ganancia_predictions_1


# In[199]:


ganancia_predictions_2=ganancias(top_predictions_2)
ganancia_predictions_2


# In[200]:


ganancias_predictions_bar=[ganancia_predictions_0, ganancia_predictions_1, ganancia_predictions_2]

etiquetas=['ganancia_predictions_0', 'ganancia_predictions_1', 'ganancia_predictions_2']
plt.bar(etiquetas, ganancias_predictions_bar)
plt.xticks(rotation=45)


# Vemos que la región con la ganancia más alta es la región 0

# In[201]:


state = np.random.RandomState(12345)


# Intervalo de confianza:
# 
# Distribución de los beneficios:
# 
# Beneficio promedio:

# Intervalo de confianza para cada pozo:

# In[202]:


state = np.random.RandomState(12345)


# In[203]:


def loss(values):
    
    loss_count = sum(1 for profit in values if profit < 0)

    probability_of_loss = loss_count / len(values)

    percentage_of_loss = probability_of_loss * 100
    return percentage_of_loss


# <div class="alert alert-block alert-info">
# <b>No estoy segura si quería que quitara esta función, pero es con la que calculo el riesgo de pérdidas, por eso no la he eliminado</b> <a class="tocSkip"></a>
# </div>

# Dado que obtuvimos 0 riesgo de perdidas en todas las regiones, vamos a hacer un bootstrapping con 500 datos de las predicciones totales, para ver el risego real de las 3 regiones

# In[204]:


def ganancias_bootstrapping_all(target, predictions):

    target=target.reset_index(drop=True)
    values = []
    for i in range(1000):
        
        subsample = pd.Series(predictions).sample(n=500, replace=True)
        subsample_top= subsample.sort_values(ascending=False).head(200)
        subtarget= target[subsample_top.index]
        ganancia= ganancias(subtarget)
        values.append(ganancia)

    values = pd.Series(values)
    mean= values.mean()
    
    lower = values.quantile(.025)
    upper = values.quantile(.975)
    intervalo_de_confianza= lower, upper
    return values, intervalo_de_confianza, mean


# In[205]:


values_all_0, intervalo_de_confianza_all_0, mean_all_0= ganancias_bootstrapping_all(target_valid_0, predictions_0)
intervalo_de_confianza_all_0


# In[206]:


values_all_1, intervalo_de_confianza_all_1, mean_all_1= ganancias_bootstrapping_all(target_valid_1, predictions_1)
intervalo_de_confianza_all_1


# In[207]:


values_all_2, intervalo_de_confianza_all_2, mean_all_2= ganancias_bootstrapping_all(target_valid_2, predictions_2)
intervalo_de_confianza_all_2


# In[208]:


mean_all_0


# In[209]:


mean_all_1


# In[210]:


mean_all_2


# In[211]:


ganancias_all_bar=[mean_all_0, mean_all_1, mean_all_2]

etiquetas=['mean_all_0', 'mean_all_1', 'mean_all_2']
plt.bar(etiquetas, ganancias_all_bar)
plt.xticks(rotation=45)
plt.title('Average profit of all predictions')


# In[212]:


loss(values_all_0)


# In[213]:


loss(values_all_1)


# In[214]:


loss(values_all_2)


# Conclusión: La región 1 es la única que tiene el riesgo de pérdidas menor a 2.5, y por lo tanto con el menor riesgo, además de que la ganancia promedio es la más lata, por lo que recomendamos la región 1

# <div class="alert alert-block alert-danger">
#     
# # Comentarios generales
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Hola, Denisse. Debemos trabajar en algunos puntos para poder aprobar tu proyecto. He dejado comentarios a lo largo del documento para ello.
# </div>

# <div class="alert alert-block alert-success">
#     
# # Comentarios generales
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Todo corregido. Has aprobado un nuevo proyecto. ¡Felicitaciones!
# </div>

# In[ ]:




