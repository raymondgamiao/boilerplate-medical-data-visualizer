import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 Import the data from medical_examination.csv and assign it to the df variable
df = pd.read_csv('medical_examination.csv')

# 2 Create the overweight column in the df variable
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = np.where(df['BMI'] > 25, 1, 0)
df.drop(columns=['BMI'], inplace=True)

# 3Normalize data by making 0 always good and 1 always bad. 
# If the value of cholesterol or gluc is 1, set the value to 0. 
# If the value is more than 1, set the value to 1.
df['cholesterol'] = np.where(df['cholesterol'] > 1, 1, 0)
df['gluc'] = np.where(df['gluc'] > 1, 1, 0)

# 4 Draw the Categorical Plot in the draw_cat_plot function
def draw_cat_plot():
    # 5 Create a DataFrame for the cat plot using pd.melt with values 
    # from cholesterol, gluc, smoke, alco, active, and overweight in the df_cat variable.
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6 Group and reformat the data in df_cat to split it by cardio. Show the counts of each feature. 
    # You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='count')
   
    # 7 Convert the data into long format and create a chart that shows the value counts of the 
    # categorical features using the following method provided by the seaborn library import : sns.catplot()
    cat_plot = sns.catplot(
        x='variable',
        y='count',
        hue='value',
        col='cardio',
        data=df_cat,
        kind='bar',
        height=5,
        aspect=1
    )
    cat_plot.set_axis_labels('variable', 'total')
    cat_plot.set_titles('Cardio: {col_name}')

    # 8 Get the figure for the output and store it in the fig variable
    fig = cat_plot.fig
    plt.tight_layout()

    # 9 Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig

# 10
def draw_heat_map():
    # 11 Clean the data in the df_heat variable by filtering out the following patient segments that represent incorrect data:
    # (Keep the correct data with (df['height'] >= df['height'].quantile(0.025)))
    
    # height is less than the 2.5th percentile 
    height_low = df['height'].quantile(0.025)

    # height is more than the 97.5th percentile
    height_high = df['height'].quantile(0.975)

    # weight is less than the 2.5th percentile
    weight_low = df['weight'].quantile(0.025)

    # weight is more than the 97.5th percentile
    weight_high = df['weight'].quantile(0.975)

    # Filter the DataFrame
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &  
        (df['height'] >= height_low) &
        (df['height'] <= height_high) &
        (df['weight'] >= weight_low) &
        (df['weight'] <= weight_high)
    ]

    # 12 Calculate the correlation matrix and store it in the corr variable
    corr = df_heat.corr()

    # 13 Generate a mask for the upper triangle and store it in the mask variable
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 15 Plot the correlation matrix using the method provided by the seaborn library import: sns.heatmap()
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax)

    ax.set_title("Correlation Matrix")

    # 16 Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
