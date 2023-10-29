import streamlit as st
import prepro
import matplotlib.pyplot as plt
import seaborn as sns
import sentiment
from PIL import Image
import similarity

image = Image.open("D:\ML\Whatapp Chat Analyser\data\logo.png")
st.sidebar.image(image)

st.sidebar.title("WAppTastic")
st.sidebar.write("Note :Whatsapp Chat in 24 hours time format only supported !!")

uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = prepro.make_dataframe(data)



# Fetch unique users

user_list = df['user'].unique().tolist()
user_list.remove('group notification')
user_list.sort() 
user_list.insert(0,"Overall")

selected_user = st.sidebar.selectbox('Show Analysis wrt',user_list)

if st.sidebar.button("Show Analysis"):

    # Stats Area
    num_msg , words , num_media , num_links = prepro.stats(selected_user,df)
    
    st.title("Top Statitics")
    col1,col2,col3,col4 = st.columns(4)
    
    with col1:
        st.header("Total Messages")
        st.title(num_msg)
    with col2:
        st.header("Total Words")
        st.title(words)
    with col3:
        st.header("Media Shared")
        st.title(num_media)
    with col4:
        st.header("Links Shared")
        st.title(num_links)
        
    # Monthly Timeline

    timeline = prepro.monthly_timeline(selected_user,df)
    fig,ax = plt.subplots()
    ax.plot(timeline['time'],timeline['message'])
    plt.xticks(rotation = 'vertical')
    st.title("Monthly Timeline")
    st.pyplot(fig)

    # Daily Timeline

    d_timeline = prepro.daily_timeline(selected_user,df)
    fig,ax = plt.subplots()
    ax.plot(d_timeline['date'],d_timeline['message'],color = 'black')
    plt.xticks(rotation = 'vertical')
    st.title("Daily Timeline")
    st.pyplot(fig)
    
    # Finding the busiest user in the group
    if selected_user == 'Overall':

        st.title("Most Busy User")
        X,new_df = prepro.most_active_user(df)
        fig,ax = plt.subplots()

        col1,col2 = st.columns(2)

        with col1:
            ax.bar(X.index,X.values,color = 'red')
            plt.xticks(rotation= 'vertical')
            st.pyplot(fig)
        with col2:
            st.dataframe(new_df)

    # Word Cloud

    st.title("Word Cloud")
    # df_wc = prepro.word_cloud(selected_user,df)

    # fig,ax = plt.subplots()
    # ax.imshow(df_wc)
    # st.pyplot(fig)

    # Most Common Words
    most_common_words = prepro.most_common_words(selected_user,df)

    fig,ax = plt.subplots()
    ax.barh(most_common_words[0],most_common_words[1])
    plt.xticks(rotation = 'vertical')
    st.title("Most Common Words")
    st.pyplot(fig)
    

    # Emoji

    most_common_emoji = prepro.most_common_emoji(selected_user,df)
    st.title("Emoji Analysis")
    col1,col2 = st.columns(2)

    with col1:
        st.dataframe(most_common_emoji)
    with col2:
        fig,ax = plt.subplots()
        ax.pie(most_common_emoji['Frequency'].head(10),labels = most_common_emoji['Emoji'].head(10),autopct = '%0.2f')
        st.pyplot(fig)

    # Activity Map

    st.title("Activity Map")

    d1 = prepro.day_active(selected_user,df)
    d2 = prepro.month_active(selected_user,df)

    col1,col2 = st.columns(2)

    with col1:
        fig,ax = plt.subplots()
        ax.bar(d1['Day'],d1['Messages'],color = 'green')
        plt.xticks(rotation = 'vertical')
        st.title("Most Busy Day")
        st.pyplot(fig)

    with col2:
        fig,ax = plt.subplots()
        ax.bar(d2['Month'],d2['Messages'],color = 'yellow')
        plt.xticks(rotation = 'vertical')
        st.title("Most Busy Month")
        st.pyplot(fig)

    # HeatMap
    
    act_heatmap = prepro.activity_heatmap(selected_user,df)
    st.title("Weekly Activity Map")
    fig,ax = plt.subplots()
    ax = sns.heatmap(act_heatmap)
    st.pyplot(fig)

    # Sentiment Analysis

    user_score = sentiment.sentiment_analysis(df)

    col1,col2 = st.columns(2)
    with col1:
        st.title("Complete Sentiment Analysis")
        st.dataframe(user_score)
    with col2:
        
        labels,sizes = sentiment.plot_sentiment(selected_user,user_score)
        sizes = [x*100 for x in sizes]
        fig,ax = plt.subplots()
        ax.bar(labels,sizes,color = 'cyan')
        plt.xlabel("Sentiment")
        plt.ylabel("Percentage")
        st.title(f"Sentiment distribution for {selected_user}")
        st.pyplot(fig)


    # User- User Similarity 

    cos_sim = similarity.creating_similarity(df)

    st.title("User-User Similarity Heat Map")
    fig,ax = plt.subplots(figsize = (20,8))
    ax = sns.heatmap(cos_sim, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5)
    plt.xlabel('User Names')
    plt.ylabel('User Names')
    st.pyplot(fig)

    if selected_user != "Overall":

        col1,col2 = st.columns(2)

        with col1:
            st.title(f"Users most similar to {selected_user}")
            sim = similarity.get_user_user_similarity(cos_sim,selected_user)
            st.dataframe(sim)
        with col2:
            fig,ax = plt.subplots()
            ax.bar(sim['User'],sim["Percentage Similarity"],color = 'red')
            plt.xticks(rotation = 'vertical')
            st.title(f"Percentage Similarity with {selected_user}")
            st.pyplot(fig)





    # Similarity
    # Embedding