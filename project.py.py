# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:43:23 2025

@author: Admin
"""

import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

uploaded_file=st.file_uploader("Upload your csv file",type=["csv"])

if uploaded_file is not None:
    WDMC=pd.read_csv(uploaded_file,index_col=0)
    WDM = WDMC.copy()
                                                            
    en=LabelEncoder()
    categorical_cols= WDM.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        WDM[col]=en.fit_transform(WDM[col])
          
    st.title("World Development Measures Insights")
    st.write("### Preview of the dataset:")
    st.write(WDM.head())
    
    st.write("Select features for clustering:")
    selected_features=st.multiselect("Choose at least 2 features:" ,WDM.columns )
    
    if len(selected_features)>=2:
        X=WDM[selected_features]
        ss=StandardScaler()
        X_scaled=ss.fit_transform(X)
        
        n = st.slider("Select the variance to be captured:",2,min(len(selected_features),5),1)
        pca=PCA(n_components= n,random_state=4)
        X_comp=pca.fit_transform(X_scaled)
        
        k=st.slider("Select number of clusters:",2,10,2)    
        
        kmeans=KMeans(n_clusters = k,random_state=4)
        clusters=kmeans.fit_predict(X_comp)
        WDMC["Clusters"]=clusters 
        WDM["Clusters"]=clusters
               
        WDM_pca=pd.DataFrame(X_comp)
        WDM_pca["Clusters"]=clusters
        
        st.write("### Clustering Results")
        
        fig,ax=plt.subplots()
        sns.scatterplot(x=WDM_pca.iloc[:,0], y = WDM_pca.iloc[:,1],hue=WDM_pca["Clusters"],palette="viridis",ax=ax)
        ax.set_title("Clusters distribution")
        st.pyplot(fig) 
    
        st.write("### Cluster Data")  
        st.write(WDM.head())         
        
        st.write("### Explained Variance by PCA components")
        st.bar_chart(pca.explained_variance_ratio_)   
        
        c=st.selectbox("Select the cluster group.",[i for i in range(k)])
        st.write(WDMC[WDMC["Clusters"]==c]["Country"].unique())
        
    else:
        st.warning("Please select at least two features for clustering.")
            
            
	

