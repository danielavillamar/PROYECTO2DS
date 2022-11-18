import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import ast
import numpy as np
import svm
import recomendacionkmeans as km
st.title('SNAPLIST🥶🥶')
st.subheader('Snaplist es una aplicacion que te ayuda a encontrar tus canciones favoritas 🧐\n Las canciones se generan de manera programatica dependiendo de tu gusto🤯\n ')
st.write("Hemos creado 3 diferentes metodos para tus recomendaciones🤓:")
st.markdown("- Cosine Similarity")
st.markdown("- Sigmoid SVM")
st.markdown("- K Means")

st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    list-style-position: inside;
}
</style>
''', unsafe_allow_html=True)

st.caption('El algoritmo Cosine-Similarity utiliza tu Cancion para recomendarte 😱!!!')
playlist= st.text_input('',placeholder='Ingresa tu Cancion')
cosine_similarity=st.checkbox('Genera Usando Cosine Similarity')
if playlist != "":
    if cosine_similarity:
    

        df = svm.svm(playlist, svm.cosine)
        
        
        
        st.subheader('Recomendacion 🔥🔥🔥')
        st.write('Recomendacion para: ', playlist)
        left_column, right_column = st.columns(2)
        if st.checkbox("Mostrar Comparacion"):
            with left_column:   
                'Comparacion de tus Recomendaciones'
                num_feat = st.selectbox(
            'Escoge una feature', df.select_dtypes('number').columns)    
                fig = px.bar(df,x=df['song_title'],y = num_feat)
                
                st.plotly_chart(fig,use_container_width=False)
        with right_column:   
           'Tus recomendaciones 🔥'
           if st.checkbox("Mostrar Data completa"):
                st.dataframe(df)
            
           st.dataframe(df['song_title']    )
else:
    if cosine_similarity:
        st.text("Ingresa tu playlist😕")
######################################################################################################
st.caption('El algoritmo Sigmoid SVM con una cancion te recomienda mas de 10 canciones similares 🤖')
cancion= st.text_input('',placeholder='Ingresa Una cancion')
sigmoid=st.checkbox('Genera Usando Sigmoid SVM')
if cancion != "":
    if sigmoid:
    

        df = svm.svm(cancion, svm.sig_kernel)
        
        
        
        st.subheader('Recomendacion 🔥🔥🔥')
        st.write('Recomendacion para: ', cancion)
        left_column, right_column = st.columns(2)
        if st.checkbox("Mostrar Grafica"):
            with left_column:   
                'Comparacion de tus Recomendaciones'
                num_feat = st.selectbox(
            'Escoge una feature', df.select_dtypes('number').columns)    
                fig = px.bar(df,x=df['song_title'],y = num_feat)
                
                st.plotly_chart(fig,use_container_width=False)
        with right_column:   
           'Tus recomendaciones 🔥'
           if st.checkbox("Mostrar Data completa"):
                st.dataframe(df)
        st.dataframe(df['song_title']    )
else:
    if sigmoid:
        st.text("Ingresa tu cancion 😕")
######################################################################################################
st.caption('El algoritmo K means solicita un json ya que utiliza las entries de el API de Spotify 🥴')
lista= st.text_area('',placeholder='Ingresa el JSON')
kmeans=st.checkbox('Genera Usando K means')
if lista != "":
    if kmeans:
        st.subheader('Tu Recomendacion con K Means 😱')
        st.markdown('He aqui el proceso del algoritmo 🤓')
        if st.checkbox('Dale aqui para los detalles 🧐'):
            st.caption("Este cluster es utilizando los datos de los generos, y como se puede observar, el algoritmo detecta una recomendacion de 8 cluster esto significa que para el algoritmo lo optimo es una playlist de 8 canciones")
            st.caption("Esto no es suficiente para ti por eso quisimos probar con otras columnas de relevancia")
            fig=km.scatter
            st.plotly_chart(fig, use_container_width=True)
            st.caption("El sigiente K Means es con los titulos de las canciones, junto con otros atributos como el nivel acustico y la duracion")
            st.caption("Esto ya es mas aceptable, 18 canciones son razonablemente una playlist 🤔")
            fig=km.scatter_songs
            st.plotly_chart(fig, use_container_width=True)
        st.subheader('Lo que te recomienda K Means 🔥')
        b=lista.replace("[","")
        b=b.replace("]","")
        result = ast.literal_eval(b)
        dat=km.data
        df= km.recommend_songs(list(result),dat)
        left_column, right_column = st.columns(2)
        if st.checkbox("Mostrar Grafica"):
            with left_column:   
                'Comparacion de tus Recomendaciones'
                num_feat = st.selectbox(
            'Escoge una feature', df.select_dtypes('number').columns)    
                fig = px.bar(df,x=df['name'],y = num_feat)
                
                st.plotly_chart(fig,use_container_width=True)
        with right_column:   
           'Tus recomendaciones 🔥'
           if st.checkbox("Mas detalle"):
            st.dataframe(df)
           st.dataframe(df['name']    )

        
else:
    if kmeans:
        st.text("Ingresa el JSON 😕")
        st.write("Sino conoces el formato JSON ve a : [JSON FORMAT](https://en.wikipedia.org/wiki/JSON)")