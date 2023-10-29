import re
import pandas as pd
from datetime import datetime
from urlextract import URLExtract

import emojis
from collections import Counter
from wordcloud import WordCloud
import seaborn as sns

def make_dataframe(data):
    """
    Reads the contents of a WhatsApp chat file and converts it into a pandas DataFrame with relevant columns.
    
    Parameters:
        f (file): A file object containing the WhatsApp chat data.
    
    Returns:
        pandas.DataFrame: A DataFrame containing the chat data with the following columns:
            - 'user' (str): The name of the user who sent the message.
            - 'message' (str): The message text.
            - 'day' (int): The day of the month when the message was sent.
            - 'month' (int): The month when the message was sent.
            - 'year' (int): The year when the message was sent.
            - 'hours' (int): The hour when the message was sent.
            - 'minutes' (int): The minute when the message was sent.
    """

    
    # Define a regular expression pattern to extract the message date/time and split the data into messages
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    
    # Create a DataFrame with the message dates/times and message text
    df = pd.DataFrame({'message_date': dates,'user_message': messages})
    
    # Convert the message date/time to a datetime object and rename the column
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %H:%M', errors='ignore')
    df.rename(columns={'message_date': 'date'}, inplace=True)
    
    # Split the message text into user and message columns and handle group notifications
    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if len(entry) != 1:
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group notification')
            messages.append(entry[0])
    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)
    
    # Extract the date components (day, month, year, hours, minutes) from the message date and add them as columns
    day = []; month = []; year = []; hours = [] ; minutes = [] ;
    for date in df['date']:
        date_format = '%d/%m/%y, %H:%M'
        dt = datetime.strptime(date[:-3], date_format)
        d = dt.day
        mo = dt.month
        y = dt.year
        h = dt.hour
        mi = dt.minute
        day.append(d)
        month.append(mo)
        year.append(y)
        hours.append(h)
        minutes.append(mi)
    df['day'] = day
    df['month'] = month 
    df['year'] = year
    df['hours'] = hours
    df['minutes'] = minutes
    df.drop(columns=['date'], inplace=True)

    return df

###################################################################################
###################################################################################

extractor = URLExtract()

def stats(selected_user,df):

  if selected_user != 'Overall':
    df = df[df['user'] == selected_user]

  # Total no. of messeges
  num_msg = df.shape[0]

  # Total no. of words
  words = []
  for message in df['message']:
    words.extend(message.split())

  # Total media Shared
  num_media = df[df['message'] == '<Media omitted>\n'].shape[0]

  # Total Links Shared
  links = []
  for message in df['message']:
    links.extend(extractor.find_urls(message))

  return num_msg, len(words), num_media, len(links)

##################################################################################
##################################################################################

def most_active_user(df):
  X = df['user'].value_counts().head()
  new_df = round((df['user'].value_counts()/df.shape[0])*100,2).reset_index().rename(columns = {'index' : 'Name','user':'Percentage'})
  return X,new_df

##################################################################################
##################################################################################

def word_cloud(selected_user,df):
  if selected_user != 'Overall':
    df = df[df['user']] == selected_user
  
  temp = df[df['user']!='group notification']
  temp = temp[temp['message'] != '<Media omitted>\n']

  wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white ')
  wc_img = wc.generate(temp['message'].str.cat(sep= " "))

  return wc_img

#############################################################################################
######################################################################################

   

def most_common_words(selected_user,df):
  if selected_user != 'Overall' :
    df = df[df['user'] == selected_user ]

  temp = df[df['user']!='group notification']
  temp = temp[temp['message'] != '<Media omitted>\n']

  f1 = open('D:\ML\Whatapp Chat Analyser\data\Hinglish_stop.txt','r')
  stop_words = f1.read()

  words = []

  for message in temp['message']:
    for word in message.lower().split() :
      if word not in stop_words:
        words.append(word)
  
  most_common_words_df = pd.DataFrame(Counter(words).most_common(25))

  return most_common_words_df

##################################################################################
##################################################################################

def most_common_emoji(selected_user,df):
  if selected_user != 'Overall' :
    df = df[df['user'] == selected_user ]
  
  emoticon = []
  for message in df['message']:
    emoticon.extend(emojis.get(message))

  emoji_df = pd.DataFrame(Counter(emoticon).most_common(len(Counter(emoticon)))).rename(columns={0:'Emoji',1:'Frequency'})

  return emoji_df

##################################################################################
##################################################################################

# Monthly Timeline
def monthly_timeline(selected_user,df):
  if selected_user != 'Overall' :
    df = df[df['user'] == selected_user ]
  
  timeline = df.groupby(['year','month']).count()['message'].reset_index()

  time = []
  for i in range(timeline.shape[0]):
    time.append(str(timeline['month'][i]) + '-' + str(timeline['year'][i]))

  timeline['time'] = time

  return timeline

##################################################################################
##################################################################################

# Daily Timeline
def daily_timeline(selected_user,df):
  if selected_user != 'Overall' :
    df = df[df['user'] == selected_user ]

  df['date'] = df.apply(lambda row: f"{row['day']}-{row['month']}-{row['year']}", axis=1)

  daily_time = df.groupby(['date']).count()['message'].reset_index()

  return daily_time

##################################################################################
##################################################################################

def month_active(selected_user,df):
  df['date'] = df.apply(lambda row: f"{row['day']}-{row['month']}-{row['year']}", axis=1)
  df['date'] = pd.to_datetime(df.date, format='%d-%m-%Y')
  df['Month_name'] = df['date'].dt.month_name()

  monthwise_data = df['Month_name'].value_counts().reset_index().rename(columns = {'index':'Month','Month_name':'Messages'})

  return monthwise_data

##################################################################################
##################################################################################

def day_active(selected_user,df):
  df['date'] = df.apply(lambda row: f"{row['day']}-{row['month']}-{row['year']}", axis=1)
  df['date'] = pd.to_datetime(df.date, format='%d-%m-%Y')
  df['Day_name'] = df['date'].dt.day_name()

  daywise_data = df['Day_name'].value_counts().reset_index().rename(columns = {'index':'Day','Day_name':'Messages'})

  return daywise_data

######################################################################################

def activity_heatmap(selected_user,df):

  if selected_user != 'Overall':
    df =df[df['user'] == selected_user]

  period = []
  for hour in df['hours']:
    if hour == 23:
      period.append(str(hour)+'-'+str('00'))
    elif hour == 0:
      period.append(str('00')+'-'+str(hour+1))
    else:
      period.append(str(hour)+'-'+str(hour+1))

  df['period'] = period

  act_heatmap = df.pivot_table(index = 'Day_name',columns = 'period',values = 'message',aggfunc = 'count').fillna(0)

  return act_heatmap