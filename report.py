# %% setup
import sys
import pandas as pd # 2.2.3
import numpy as np # 2.1.2
import scipy as sp # 1.14.1
from scipy.stats import iqr, skew, kurtosis, ttest_ind, pearsonr, f_oneway, norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib as mpl # 3.9.2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns # 0.13.2
from statistics import mode

pd.options.mode.copy_on_write = True # introduced due to pandas warning to migrate to C.o.W.

# print module versions
print(f"""
    Python version: {sys.version.split()[0]}
    
    --- Imported module versions ---
    
    Pandas version: {pd.__version__}
    NumPy version: {np.__version__}
    SciPy version: {sp.__version__}
    statsmodels version: {sm.__version__}
    Matplotlib version: {mpl.__version__}
    Seaborn version: {sns.__version__}
    """)
# %% read in csv
def read_data(fname):
    df = pd.read_csv(fname)
    return df

df = read_data("movie_metadata.csv")

# %% initial inspection of data

print(df.head(10))

# %%
print(f"Rows: {df.shape[0]} \nColumns: {len(df.columns)}") # rows 5043, columns 28

# %%
print(df.info()) # many columns have nulls

# %%
print(df.sort_index(axis = 1).describe())
print(mode(df["imdb_score"])) # for summary table
print(mode(df["title_year"])) # for summary table

# for summary table
for col in df.sort_index(axis = 1).columns:
    print(f"{col}: {len(df[col].unique())}")

# %%
print(df.sort_index(axis = 1).columns)

# As this dataset is a list of movies, each row should refer to a unique movie. To ensure this, we need to 
# remove duplicate rows. 
# 
# Each movie will have its own title ("movie_title"), creation year ("title_year") and director 
# ("director_name"). These columns will be used to identify duplicates. 
# 
# Using all columns to identify duplicates could potentially lead to duplicate entries of movies. For example, 
# if a movie had two entries, but only differed by the value in the "movie_facebook_likes" column (e.g. due to 
# accidentally recording the facebook likes twice), this would not be considered a duplicate if we used every 
# column. Isolating just those columns which we believe act as unique identifiers avoid this problem - however, 
# we are assuming that there isn't a scenario where two different movies with the same name, made in the same 
# year by two different directors with the same name. We could implement a check for this, but such a scenario is
# assumed to be very unlikely.

# %% duplicates

# Identify rows that were duplicated, stored for (potential) future reference
duplicated_rows = df[df.duplicated(subset = ["movie_title", "title_year", "director_name"], keep="first")][["movie_title", "title_year", "director_name"]]

duplicated_rows.shape[0] # 124 duplicates!

# remove duplicates while retaining the first occurrence
df.drop_duplicates(["movie_title", "title_year", "director_name"], keep = "first", inplace = True)

# Each row should now refer to a unique movie!

# %% "non-movie" rows

# Some rows appear to not be movies (e.g. TV shows), although there is no definitive way to identify these. From
# inspecting the data, it seems that "non-movies" have no director listed in the data set. This means we can 
# remove rows with no director

# One way we can remove "non-movies" is to remove rows which have no director - because every movie has a director!
non_director_rows = df[df["director_name"].isna()]

print(f"Removing {len(non_director_rows)} rows with no known director")
df = df[~df["director_name"].isna()]

# %% "very low" review count

# movies with very few imdb reviews are likely not reliable enough data points to interpret (e.g. the ratings 
# maybe be grossly inflated above what they "should" be). inspecting the data can give us an idea of what "very 
# low" might be.

sns.histplot(
    data = df, 
    x = "num_voted_users",
    kde = True,
    label = f"Skew: {round(df["num_voted_users"].skew(), 3)}"
)
plt.legend(
    loc = "lower center",
    fontsize = 8
)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show() # show first plot
plt.close() # close first plot so second plot can be shown

# %%

# there is a strong positive skew
df["num_voted_users"].skew() # 4.04!

# skew is unsurprising given that the this seems to follow a log normal distribution. We can therefore take the # logarithm of the x-axis to (potentially!) see a more Gaussian-like distribution.
#
# given that logarithmic scales cannot show counts = 0, we need to check how many rows have num_voted_users == 0
# 
# check min value first
df["num_voted_users"].min() # 5, as such we have no zero-counts (log scale is okay to use)

sns.histplot(
    data = df, 
    x = "num_voted_users",
    log_scale = (True, False), # y = log(x)
    kde = True,
    label = f"Skew:{round(np.log10(df["num_voted_users"]).skew(), 3)}"
)
plt.legend(
    loc = "lower center",
    fontsize = 8
)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
plt.close()

# log-scale for x-axis shows sizable negative skew
np.log10(df["num_voted_users"]).skew() # -0.985

# mean average and stdev: num_voted_users
print(df["num_voted_users"].mean()) # mean = 83,720.98
print(df["num_voted_users"].std()) # 3sigma = 139379.77
print(df["num_voted_users"].mean() - df["num_voted_users"].std()) # -55658.79

# Can't use standard deviations to remove small values since data is heavily skewed (not Gaussian!). As such, I 
# am deciding to remove films with less than 1000 (10^3) reviews as, judging by eye, it seems that this places 
# the mean approximately in the middle of the distribution (i.e. it's a "good enough" minimum review count). 
# 
# To be clear, if we had a strong negative skew (or strong positive for the log-scale), we **would not** remove 
# these since more reviews means more accurate estimates for the true population IMDB score.

sns.histplot(
    data = df[df["num_voted_users"] > 1000], 
    x = "num_voted_users",
    log_scale = (True, False), # y = log(x)
    kde = True,
    label = f"Skew: {round(np.log10(df[df["num_voted_users"] > 1000]["num_voted_users"]).skew(), 3)}"
)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
plt.close()

# skew is now "acceptable"
np.log10(df[df["num_voted_users"] > 1000]["num_voted_users"]).skew() # -0.242

# looks more Gaussian (still log scale, of course!), but not perfect at all. However, the main point is that 
# every movie has at least N = 1000 as a sample size for its IMDB score which is an acceptable sample size 
# (Central Limit Theorem) to have our IMDB scores approximate their true population values.

# %% combine these plots for display in report

fig, ax = plt.subplots(1, 1, figsize = (8, 6))

sns.histplot(
    data = df, 
    x = "num_voted_users",
    label = f"Skew: {round(df["num_voted_users"].skew(), 3)}",
    color = "purple",
    alpha = 0.5,
    bins = np.linspace(0, 2000000, 9) # creates 8 bins with 250,000 width, between 0 and 2,000,000
)

plt.margins(x = 0)
ax.ticklabel_format(style = "plain") 

ax.get_xaxis().set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
plt.xticks(np.arange(0, 2250000, step = 250000), rotation = 20, fontsize = 8)
plt.xlabel("Number of Reviewers")

plt.yticks(np.arange(0, 5000, step = 500))

plt.legend(fontsize = 10)
plt.savefig("report_pdf/images/review_hist.png")
plt.show()
plt.close()

fig, axes = plt.subplots(1, 2, figsize = (12, 7))

sns.histplot(
    data = df, 
    x = "num_voted_users",
    log_scale = (True, False),
    label = f"Skew: {round(np.log10(df["num_voted_users"]).skew(), 3)}",
    color = "purple",
    alpha = 0.5,
    bins = 50,
    ax = axes[0]
)
axes[0].set_title(
    "All Movies", 
    fontsize = 16,
    fontweight = "bold"
)
axes[0].set_xlabel("Number of Reviewers", fontsize = 12)
axes[0].set_ylabel("Count", fontsize = 12)
axes[0].set_yticks(np.arange(0, 400, step = 50))
axes[0].set_xlim(
    df["num_voted_users"].min(), # needs a min > 0 since it's a log-scale
    df["num_voted_users"].max()
) 
axes[0].legend(
    loc = "upper left",
    fontsize = 12
)

sns.histplot(
    data = df, 
    x = "num_voted_users",
    log_scale = (True, False),
    label = f"$Skew_{{> 1000}}$: {round(np.log10(df[df["num_voted_users"] > 1000]["num_voted_users"]).skew(), 3)}", 
    color = "#d3a3d0",
    alpha = 0.5,
    bins = 50,
    ax = axes[1]
)
axes[1].set_title(
    "Movies with > 1000 Reviews", 
    fontsize = 16,
    fontweight = "bold"
)
axes[1].set_xlabel("Number of Reviewers\n$\mathbf{Note:}$ x-axis begins at $10^{3}$", fontsize = 12)
axes[1].set_ylabel("Count", fontsize = 12)
axes[1].set_yticks(np.arange(0, 400, step = 50))
axes[1].set_xlim(1000, df["num_voted_users"].max()) # truncates x-axis to remove bins < 1000
axes[1].legend(
    loc = "upper left",
    fontsize = 12
)

plt.savefig("report_pdf/images/review_hist_log.png")
plt.tight_layout()
plt.show()

    # %% remove rows with less than 1000 imdb reviews
print(df[df["num_voted_users"] < 1000].shape[0])
df = df[df["num_voted_users"] > 1000]

# %%
df_required_results = df # to be used for 1. and 2. of the required results from the report's brief

# further cleaning is specifically to address the main research question. 

# %% method section: summary statistics

pd.DataFrame(
    data = {
        "N" : [f"{df.shape[0]}"],
        "Year range" : [f"{df["title_year"].min():.0f}-{df["title_year"].max():.0f}"]
    }
)


# %% Required Results 1: top 10 movies according to IMDB score

# grab "movie_title", "imdb_score" and "num_voted_users" columns from df_required_results. Sort, in descending 
# order, by imdb_score, then just show the first 10 rows. 
print(df_required_results[["movie_title", "title_year", "imdb_score", "num_voted_users"]].sort_values("imdb_score", ascending = False).head(10))

# printing table in latex to copy-and-paste for the report pdf
required_result_1_table = df_required_results[["movie_title", "title_year", "imdb_score", "num_voted_users"]].sort_values(["imdb_score", "num_voted_users"], ascending = False).head(10)

required_result_1_table["Rank"] = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"]

required_result_1_table = required_result_1_table[["Rank", "movie_title", "title_year", "imdb_score", "num_voted_users"]]

required_result_1_table = required_result_1_table.rename(columns = {
    
    "movie_title" : "Movie Title",
    "title_year": "Year Released",
    "imdb_score" : "IMDb Score",
    "num_voted_users" : "Number of User Reviews"

})

required_result_1_table["Number of User Reviews"] = required_result_1_table["Number of User Reviews"].apply(lambda x: f"{x:,}")

required_result_1_table["Year Released"] = required_result_1_table["Year Released"].apply(lambda x: f"{x:.0f}")

print(required_result_1_table.to_latex(
    escape = False, 
    index = False, 
    column_format = "p{0.75cm}p{5cm}ccc", 
    float_format = "{:0.1f}".format
))

# %% Required Results 2: top 5 individual genres (e.g., action, romance, horror, thriller) with the most number 
# of movies, the distribution of their IMDb scores and some summary statistics

# since values in "genres" column are lists of multiple genres, delimited by "|", we need to split these values
genres_list = list(df_required_results["genres"].str.split("|"))

# now, values in this column are python lists of each genre. We now have an array of lists!

# since adding two lists just concatenates them, summing all the sublists, starting with an empty list, will 
# unnest the genres_list. converting to a set will then remove duplicates, since sets cannot have duplicates
genres = list(set(sum(genres_list, [])))
df_genre_counts = pd.DataFrame(genres, columns = ["genre"])

# loop through genres set and count the number of movies for each genre
genre_counts = [] # initialise empty "genre counts" list

for genre in genres: # for each genre in out list of unique genres
    count = df_required_results['genres'].apply(lambda x: genre in x).sum() # count the number of rows containing that genre in the "genres" column
    genre_counts.append(count) # append that count to the "genre counts" list

df_genre_counts["count"] = genre_counts # assign this "genre counts" list to a new "count" column

# show top 5 genres with most movies
print(df_genre_counts.sort_values("count", ascending = False).head(5))

# printing table in latex to copy-and-paste for the report pdf
required_result_2_table = df_genre_counts.sort_values("count", ascending = False).head(5)

required_result_2_table = required_result_2_table.rename(columns = {
    
    "genre" : "Genre",
    "count" : "Number of Movies"

})

required_result_2_table["Number of Movies"] = required_result_2_table["Number of Movies"].apply(lambda x: f"{x:,}")

print(required_result_2_table.to_latex(
    escape = False, 
    index = False, 
    column_format = "p{5cm}c",
    float_format = "{:0.1f}".format
))

# %% plot distributions for top 5 genres

# create list of top 5 genres
top_5_genres = list(df_genre_counts.sort_values("count", ascending = False).head(5)["genre"].unique())

# plot imdb_score distribution for each top 5 genre

fig = plt.figure(figsize = (12, 12))
gs = fig.add_gridspec(3, 2) 
plt.subplots_adjust(hspace = 0.5)

axes = [
    fig.add_subplot(gs[0, 0]), 
    fig.add_subplot(gs[0, 1]),  
    fig.add_subplot(gs[1, 0]), 
    fig.add_subplot(gs[1, 1]), 
    fig.add_subplot(gs[2, 0]) 
]

for index, genre in enumerate(top_5_genres):
    
    ax = axes[index]
    data = df_required_results[df_required_results["genres"].str.contains(genre, case = False)]
    
    
    sns.histplot(
        data = data,
        x = "imdb_score",
        ax = ax,
        bins = np.arange(1, 11, step = 1),
        color = sns.color_palette("tab10", len(genres))[index], # uses color from palette based on loop index
        label = f"IQR: {round(iqr(data["imdb_score"]), 2)}\nskew: {round(skew(data["imdb_score"]), 2)}\nkurtosis: {round(kurtosis(data["imdb_score"]), 2)}"
    )
    
    ax.axvline(
        np.mean(data["imdb_score"]), 
        color = "black", 
        alpha = 1,
        linestyle = "--", 
        linewidth = 2, 
        label = fr"$\bar{{x}}$: {np.mean(data["imdb_score"]):.2f} $\pm$ {np.std(data["imdb_score"]):.2f}$\sigma$" + f"\n(N = {data.shape[0]})\n"
    )
    
    ax.set_xlim(1, 10)
    ax.set_xticks(np.arange(1, 11, step = 1))
    
    ax.set_ylim(0, 1000)
    
    ax.legend(loc='upper left', fontsize = 11)
    
    ax.set_title(
        label = f"{genre} (N = {len(data["imdb_score"])})", 
        fontsize = 18,
        fontweight = "bold",
        color = sns.color_palette("tab10", len(genres))[index]
    )

plt.savefig("report_pdf/images/top5histplots.png")
plt.tight_layout()
plt.show()

# %% main research question: explore missing data

def missing_data_summary(dataframe):
    
    row_count = dataframe.shape[0] # this gets the row count for df
    non_null_counts = dataframe.notnull().sum() # finds the non-null count for each column
 
    null_counts = row_count - non_null_counts # calculates the null counts for each column
    null_counts_percentage = round((null_counts / row_count) * 100, 2) # express null counts as percentages

    missing_data_summary_df = pd.DataFrame({
        
        "column_name": dataframe.columns, # first col is just the column names of the dataframe
        "null_count": null_counts.values, # second col is the null counts from the "null_counts" dictionary
        "null_percentage": null_counts_percentage.values # third col is the percentages 
        
    }).sort_values(
        "null_count", # sorts by null_count
        ascending = False, # sorts in descending order
        ignore_index = True # ignore column index (optional)
    )

    return missing_data_summary_df # print missing data table

missing_data_summary(df).head(5)

print(missing_data_summary(df).head(5).rename(columns = {
    
    "column_name" : "Column Name",
    "null_count" : "Nulls",
    "null_percentage" : "Percentage of Total Rows"

}).to_latex(
    escape = True, 
    index = False, 
    float_format = "{:0.1f}".format
))

# top 5 includes "gross" and "budget" which may be problematic for answering our research question (i.e. finding 
# the profit of each movie may be useful)
# 
# we want to determine whether the production company should make a romance movie of a horror movie. As such
# we should see if these nulls incorporate a substantial proportion of our target genres (romance & horror)
# 
# For example, it could be that all the null "gross" rows constitute all the romance movies in our data. As such
# removing them would remove all the romance films from our analysis - not helpful!

# %% create subset dfs for "romance" and "horror" films
# 
# first, we need to check for any movies which might fall into both "horror" and "romance" genres before 
# sub-setting them and remove/deal with them depending on how many there are. For instance, we can't speak of the # set of all "horror" movies as being independent of "romance" movies if some movies fall into both genres. 

print(f"Number of movies that are both 'horror' and 'romance': {df[df["genres"].str.contains("romance", case = False) & df["genres"].str.contains("horror", case = False)].shape[0]}")

# since there are only 17 such films in the data set, I will exclude these from the analysis, rather than 
# spending time trying to find alternative means to place them into the "correct" genre category. These films
# could have important information/implications, so this could be a weakness of this analysis (e.g. Bram Stoker's 
# Dracula (1992) is classed as both horror and romance, but is a fairly high-profile movie having Coppola as 
# director and actors like Anthony Hopkins, Keanu Reeves and Gary Oldman - removing this movie from analysis 
# might be problematic!)

df = df[~(df["genres"].str.contains("romance", case = False) & df["genres"].str.contains("horror", case = False))]

# %%
df_romance = df[df["genres"].str.contains("romance", case = False)] # using str.contains(string, case=False) to find any occurrences of "romance", regardless of case, in the "genres" column
df_horror = df[df["genres"].str.contains("horror", case = False)]

subset_dfs = {
    "df_horror": df_horror, 
    "df_romance": df_romance
    }

# this filtering assumes the values in the "genres" column have been encoded correctly (assigning genre to a 
# movie is arguably subjective) 
# %% check null counts

for name, sub_df in subset_dfs.items():
    print(f"-------------------- {name} --------------------\nN: {sub_df.shape[0]}\n{missing_data_summary(sub_df)}\n")
    print(missing_data_summary(sub_df).head(5).rename(columns = {
    
    "column_name" : "Column Name",
    "null_count" : "Nulls",
    "null_percentage" : "Percentage of Total Rows"

    }).to_latex(
        escape = True, 
        index = False, 
        float_format = "{:0.1f}".format
    ))
# %% remove "gross" and "budget" nulls 

# we can see that null count profiles of these subset dataframes are similar to the overall dataframe. However, 
# df_romance has over 25% of nulls for its "gross" column. Removing these nulls still leaves us with ~850 data 
# points, but this is a notable omission of data.
# 
# since "gross" and "budget" have been identified as important variables for our research question, null values
# for these will be excluded hereafter from this analysis 

removed_rows_nulls = []
for name, sub_df in subset_dfs.items():
    print(f"Number of rows in {name} before exclusion of nulls: {sub_df.shape[0]}")
    before = sub_df.shape[0]
    sub_df.dropna(subset = ["gross", "budget"], inplace = True)
    print(f"Number of rows in {name} after exclusion of nulls: {sub_df.shape[0]}\n")
    after = sub_df.shape[0]
    print(f"Difference: {before - after}")
    removed_rows_nulls.append(before - after)
    
print(sum(removed_rows_nulls))
# %% from this, we can determine that removing the nulls from the original dataframe would not lead to us to 
# omitting the entirety of the genre categories we are interested in. 

df.dropna(subset = ["gross", "budget"], inplace = True)

# %% calculate "profit" column

# profit tells us which movies were commercially successful. Positive profit is "successful", negative (or 0) is 
# "unsuccessful" (0 is unsuccessful because of the opportunity cost - i.e. despite not losing any money, we 
# could have made a profitable movie instead). 
# 
# A weakness of this data is that it doesn't take inflation into account. Given that movies may take years to 
# complete and budgets can be very large, inflation matters. Additionally, the movies themselves have creations 
# dates ranging over 100 years which also means inflation has an impact when comparing across time.

df["profit"] = df["gross"] - df["budget"]

for sub_df in subset_dfs.values():
    sub_df["profit"] = sub_df["gross"] - sub_df["budget"]
    
# %% check nulls again

for name, sub_df in subset_dfs.items():
    print(f"-------------------- {name} --------------------\n{missing_data_summary(sub_df)}\n")
    
# at this point, null values still exist in our data set, but they currently exist in columns that we are not 
# (currently) planning to use. 

# %% EDA
# 
# 1. checking higher IMDb score = higher profit
#
# Whilst the senior leadership team assume higher IMDb score means higher profit, we can check to see if this 
# is reasonable.  
def corr_plot(x, y, dataframe):

    correlation = round(pearsonr(dataframe[x], dataframe[y])[0], 4)
    significance = pearsonr(dataframe[x], dataframe[y])[1]
    dof = dataframe.dropna().shape[0] - 2
    
    fig = plt.subplots(1, 1, figsize = (8, 4))
    
    sns.scatterplot(data = dataframe, x = x, y = y, label = f"--- Pearson Correlation ---\n\nCoefficient: {correlation}\n\nP-value: {significance:.2E}\n\ndof: {dof:.2f}")
    
    plt.xlabel("IMDb Score", fontsize = 14)
    plt.xticks(np.arange(1,11,step=1), fontsize = 14)
    
    plt.ylabel("Profit (Inflation-adjusted)", fontsize = 14)
    plt.yticks(fontsize = 14)
    
    plt.legend(
        fontsize = 12
    )
    plt.tight_layout()
    for name, val in globals().items():
        if val is dataframe:
            plt.savefig(f"report_pdf/images/corr_{name}.png")
    plt.show()

print(corr_plot("imdb_score", "profit", df))

# interestingly, there seems to be very little correlation between imdb_score and profit. This might be caused by
# the large outliers.

# %% inspecting outliers from scatterplot

print(df[df["profit"] == df["profit"].min()].iloc[0]) # the movie corresponding to the lowest point on the 
# scatterplot

# we can see that "The Host (2006)" grossed 2,201,412 with a budget of 12,215,500,000... 
# 
# Given that "The Host (2006)" is a South Korean film, these numbers may reflect differences in currency. From 
# inspecting the imdb page for the movie (https://www.imdb.com/title/tt0468492/?ref_=fn_t#:~:text=credits%20at%20IMDbPro-,Box%20office,-Edit), it would appear that "gross" refers to "Gross 
# revenue US & Canada", whilst "budget" is expressed in the movie's country-of-origin's currency.
# 
# This, combined with the issue of inflation, makes interpretation difficult. We could adjust for inflation, 
# exchange rate and scrape the correct "Gross worldwide" values for each of our ~5000 movies. Instead, I am going
# to opt for only using movies made in the USA and then adjust "gross" and "budget" for inflation.

# %% check sample size of US movies
print(df.groupby("country").size().reset_index(name='counts').sort_values(by='counts', ascending = False))

# USA = 2918
# 
# As such, these still constitute a large proportion of our data. Excluding the rest will mean our conclusions
# only apply to USA viewers, but we will have much more accurate measures of profit for each movie 
# %% Back to cleaning: filter out non-US movies
df_usa = df[df["country"].str.contains("USA", case = False)]

print(df.shape[0] - df_usa.shape[0]) # removed 778
print(df_usa.shape[0]) # N = 2918
# %% adjust "gross" and "budget" for inflation 

# using CPI data from U.S. Bureau of labour statistics, and the formula for adjusting nominal values to 2023 US 
# dollars

cpi_data = pd.read_csv("cpi_data.csv") # read cpi_data csv
cpi_dict = cpi_data.set_index('Year')['Annual'].to_dict() # convert the cpi_data to a dictionary

# inflation formula:
#
# 2023 inflation-adjusted price = original price * (2023 CPI / CPI of original price's year)

# take each "gross" value and multiply it by the ratio between cpi 2023 and the cpi of the year the movie was 
# released

df_usa['gross_inf_adj'] = df_usa.apply(lambda row: row['gross'] * (cpi_dict[2023] / cpi_dict[row['title_year']]), axis=1) 

# take each "budget" value and multiply it by the ratio between cpi 2023 and the cpi of the year the movie was 
# released

df_usa['budget_inf_adj'] = df_usa.apply(lambda row: row['budget'] * (cpi_dict[2023] / cpi_dict[row['title_year']]), axis=1)

# subtract budget_inf_adj from gross_inf_adj to calculate profit_inf_adj

df_usa["profit_inf_adj"] = df_usa['gross_inf_adj'] - df_usa['budget_inf_adj']

# %% check correlation again after adjustments

print(corr_plot("imdb_score", "profit_inf_adj", df_usa))

# a relatively weak correlation, probably due to a large amount of movies that (approximately) break even. The 
# correlation is positive, which is in-line with our assumption and the most profitable films seem to be rated 
# between 7-10. Whilst profit is not guaranteed with a higher imdb score, higher profits are essentially only 
# obtained by higher rated movies. Thus, we can still aim to make a movie with the highest possible IMDb score

# %% EDA
# 
# 2. change in imdb scores of romance and horror movies over time
#
# something else to consider when comparing romance and horror movies is that ratings may be changing over time for 
# each genre. For example, one genre might outperform on average but be on a downward trajectory. 

df_usa_romance = df_usa[df_usa["genres"].str.contains("romance", case = False)] 
df_usa_horror = df_usa[df_usa["genres"].str.contains("horror", case = False)]

fig, ax = plt.subplots(2, 1, figsize = (12, 6))

sns.scatterplot(
    data = df_usa_romance,
    x = "title_year",
    y = "imdb_score",
    color = "red",
    ax = ax[0]
)

sns.scatterplot(
    data = df_usa_horror,
    x = "title_year",
    y = "imdb_score",
    color = "green",
    ax = ax[0]
)

ax[0].set_xlabel("Title Year", fontsize = 14)
ax[0].set_xticks(np.arange(1920, 2030, step = 10))

ax[0].set_ylabel("IMDb Score", fontsize = 14)
ax[0].set_yticks(np.arange(1, 11, step = 1))

ax[0].set_title(
    "1916-2016",
    fontsize = 18
)

# there doesn't appear to be any obvious difference in the trends of each genre, nor is it clear what the trend even 
# is. IMDb scores seems fairly uniform between 4 and 8
#
# the quantity of movies produced each year seems to increase substantially after ~1980, we could filter to just look 
# after this year to see if a trend emerges

sns.scatterplot(
    data = df_usa_romance[df_usa_romance["title_year"] >= 1980],
    x = "title_year",
    y = "imdb_score",
    color = "red",
    ax = ax[1]
)

sns.scatterplot(
    data = df_usa_horror[df_usa_horror["title_year"] >= 1980],
    x = "title_year",
    y = "imdb_score",
    color = "green",
    ax = ax[1]
)

ax[1].set_xlabel("Title Year", fontsize = 14)
ax[1].set_xticks(np.arange(1980, 2030, step = 10))

ax[1].set_ylabel("IMDb Score", fontsize = 14)
ax[1].set_yticks(np.arange(1, 11, step = 1))

ax[1].set_title(
    "Post-1980",
    fontsize = 18
)
plt.tight_layout()
plt.savefig("report_pdf/images/trend.png")
plt.show()

# Again, there doesn't seem to be any difference in trends, nor any notable increase/decrease over time. This could be 
# a product of how imdb scores are always "present" ratings, rather than reflections of how a movies was perceived at 
# the time of release.
# 
# as such, I am deciding not to filter out "older" movies

# %% resulting sample size

print(df_usa.shape[0]) # 2918

print(((5043 - df_usa.shape[0])/5043)) # 42% reduction

# %% hypothesis test (romance IMDb vs horror IMDb)

# Gaussian distribution assumption

# Create the P-P plot
fig, ax = plt.subplots(1, 2, figsize = (8, 4))

# Loop over each DataFrame to create a P-P plot
for index, dataframe in enumerate([df_usa_horror, df_usa_romance]):
    
    names = ["Horror", "Romance"]
    # Sort the data in the 'data_column'
    dataframe_sorted = np.sort(dataframe["imdb_score"])
    
    # Calculate the empirical cumulative probabilities
    sample_cdf = np.arange(1, len(dataframe_sorted) + 1) / len(dataframe_sorted)
    
    # Calculate the theoretical cumulative probabilities for a normal distribution
    theoretical_cdf = norm.cdf(dataframe_sorted, loc=np.mean(dataframe_sorted), scale=np.std(dataframe_sorted))
    
    # Plot the P-P plot on the i-th subplot
    ax[index].plot(
        theoretical_cdf, 
        sample_cdf, 
        marker = "o", 
        linestyle = "--", 
        color = "black", 
        markersize = 4
    )
    ax[index].plot([0, 1], [0, 1], color = "red", linestyle = "--")
    ax[index].set_title(f"{names[index]}")
    ax[index].set_xlabel("Theoretical CDF")
    ax[index].set_ylabel("Empirical CDF")
    ax[index].grid()

plt.tight_layout()
plt.savefig("report_pdf/images/ppplot.png")
plt.show()

# %% KDE plots and Welch's t-test

sns.kdeplot(
    data = df_usa_romance,
    x = "imdb_score",
    color = "red",
    fill = True,
    alpha = 0.2
)

sns.kdeplot(
    data = df_usa_horror, 
    x = "imdb_score",
    color = "green",
    fill = True,
    alpha = 0.2
)

plt.axvline(
    np.mean(df_usa_romance["imdb_score"]), 
    color = "red", 
    alpha = 0.8,
    linestyle = "--", 
    linewidth = 2.5, 
    label = fr"$\bar{{x}}_{{IMDb}}$ Romance: {np.mean(df_usa_romance["imdb_score"]):.2f} $\pm$ {np.std(df_usa_romance["imdb_score"]):.2f}$\sigma$" + f"\n(N = {df_usa_romance.shape[0]})\n"
)

plt.axvline(
    np.mean(df_usa_horror["imdb_score"]), 
    color = "green", 
    alpha = 0.8,
    linestyle = "--", 
    linewidth = 2.5,
    label = fr"$\bar{{x}}_{{IMDb}}$ Horror: {np.mean(df_usa_horror["imdb_score"]):.2f} $\pm$ {np.std(df_usa_horror["imdb_score"]):.2f}$\sigma$" + f"\n(N = {df_usa_horror.shape[0]})"
)

plt.xlabel("IMDb Score")
plt.xticks(np.arange(1, 11, step = 1))

plt.ylabel("Density")
plt.yticks(np.arange(0, 0.6, step = 0.1))

plt.legend(loc = "upper left", fontsize = 9)
plt.savefig("report_pdf/images/Hyp1_KDE.png")
plt.show()

ttest_result = ttest_ind(
    df_usa_romance["imdb_score"], 
    df_usa_horror["imdb_score"], 
    equal_var = False,
    alternative = "two-sided"
)

print(f"Results from Welch's t-test: \n\nT-statistic: {ttest_result.statistic:.4f} \nP-value: {ttest_result.pvalue:.4f} \n\nConfidence Interval: ({ttest_result.confidence_interval(confidence_level = 0.95)[0]:.4f}, {ttest_result.confidence_interval(confidence_level = 0.95)[1]:.4f})\n\nDegrees of Freedom: {ttest_result.df:.4f}")
if ttest_result.pvalue < 0.05:
    print("P-value is below 0.05 significance threshold, reject null hypothesis.")
else:
    print("P-value is above 0.05 significance threshold, failed to reject null hypothesis.")

# significant difference in average IMDb score between romance and horror movies. KDE plot shows that romance
# movies tend to have significantly higher imdb scores than horror movies

# %% directors one-way ANOVA

# filter df_usa_romance to just show "director_name" and "imdb_score" columns. Group by each director name and aggregate all their imdb scores
df_director_anova = df_usa_romance[["director_name", "imdb_score"]].groupby("director_name")["imdb_score"].agg(list).reset_index() 

# count the number of unique imdb ratings each director has
df_director_anova["no_of_imdbs"] = df_director_anova["imdb_score"].apply(len)

# filter out directors with < 2 imdb scores (can't calculate variance with 1 imdb score)
df_director_anova = df_director_anova[df_director_anova["no_of_imdbs"] >= 2]

# flatten the df_director_anova back out again
df_director_anova_flattened = []
for _, row in df_director_anova.iterrows():
    for imdb_score in row["imdb_score"]:
        df_director_anova_flattened.append({"director_name": row["director_name"], "imdb_score": imdb_score})

df_director_anova_final = pd.DataFrame(df_director_anova_flattened)

# grab each director's set of imdb scores
directors = [df_director_anova_final[df_director_anova_final["director_name"] == name]["imdb_score"] for name in df_director_anova_final["director_name"].unique()]

# %% perform one-way ANOVA 

f_stat, p_val = f_oneway(*directors)

print(f"F-statistic: {f_stat:.2f}\np-value: {p_val:.2f}\nDegrees of Freedom: ({len(df_director_anova) - 1:.2f}, {sum(df_director_anova["no_of_imdbs"]) - len(df_director_anova)})")

# %%
# Perform Tukey's HSD test
tukey_result = pairwise_tukeyhsd(df_director_anova_final["imdb_score"], df_director_anova_final["director_name"], alpha = 0.05)

summary = tukey_result.summary()
print(summary)

# %% Add directors who show significant differences to a set

significant_names = set()
for row in summary.data[1:]:  # Skip the header row
    if row[6] == True:
        significant_names.add(row[0])
        significant_names.add(row[1])
        
print("Significant directors:", significant_names) # set ensures only unique values (director names) are retained
# %% filter to just those directors in the "sigificant_names" set

df_directors_tukey = df_director_anova_final[df_director_anova_final["director_name"].isin(significant_names)]

# %% (messy) means plot to visualise one way ANOVA

# calculate means and standard errors for each group
means = df_directors_tukey.groupby("director_name")["imdb_score"].mean()
stderr = df_directors_tukey.groupby("director_name")["imdb_score"].std() / np.sqrt(df_directors_tukey.groupby("director_name")["imdb_score"].count())

# sort the means and standard errors from largest to smallest
sorted_means = means.sort_values(ascending=False)
sorted_stderr = stderr[sorted_means.index]

# prepare data for plotting
sorted_names = sorted_means.index

plt.figure(figsize=(12, 8))
plt.errorbar(
    sorted_names, # x-axis: each director's name
    sorted_means, # y-axis: each director's average IMDb score
    yerr = sorted_stderr, # adds error bars to each director's average IMDb score
    fmt = "o",                
    color = "blue",          
    markersize = 6,           
    ecolor = "black",        
    capsize = 5,               
    linestyle = "None",       
    markeredgewidth = 2       
)

plt.axhline(
    y = 6,
    color = "red",
    linestyle = "-."
)

plt.axhline(
    y = means.loc["Martin Brest"] + stderr.loc["Martin Brest"],
    color = "green",
    linestyle = "--"
)

plt.title("")
plt.xlabel("Director", fontsize = 20, fontweight = "bold")
plt.xticks(rotation = 45, fontsize = 16)
plt.ylabel("Average IMDb Score", fontsize = 20, fontweight = "bold")
plt.yticks(np.arange(1, 11, step = 1))
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("report_pdf/images/director_tukey.png")
plt.show()
# %% actor ANOVA

# filter df_usa_romance to just show "actor_1_name" and "imdb_score" columns. Group by each actor name and aggregate all their imdb scores
df_actor_anova = df_usa_romance[["actor_1_name", "imdb_score"]].groupby("actor_1_name")["imdb_score"].agg(list).reset_index() 

# count the number of unique imdb ratings each actor has
df_actor_anova["no_of_imdbs"] = df_actor_anova["imdb_score"].apply(len)

# filter out actors with < 2 imdb scores (can't calculate variance with 1 imdb score)
df_actor_anova = df_actor_anova[df_actor_anova["no_of_imdbs"] >= 2]

# flatten the df_actor_anova back out again
df_actor_anova_flattened = []
for _, row in df_actor_anova.iterrows():
    for imdb_score in row["imdb_score"]:
        df_actor_anova_flattened.append({"actor_1_name": row["actor_1_name"], "imdb_score": imdb_score})

df_actor_anova_final = pd.DataFrame(df_actor_anova_flattened)

# grab each actor's set of imdb scores
actors = [df_actor_anova_final[df_actor_anova_final["actor_1_name"] == name]["imdb_score"] for name in df_actor_anova_final["actor_1_name"].unique()]

# %% perform one-way ANOVA 

f_stat, p_val = f_oneway(*actors)

print(f"F-statistic: {f_stat:.2f}\np-value: {p_val:.2f}\nDegrees of Freedom: ({len(df_actor_anova) - 1:.2f}, {sum(df_actor_anova["no_of_imdbs"]) - len(df_actor_anova)})")

# %%
# Perform Tukey's HSD test
tukey_result = pairwise_tukeyhsd(df_actor_anova_final["imdb_score"], df_actor_anova_final["actor_1_name"], alpha = 0.05)

summary = tukey_result.summary()
print(summary)

# %% Add directors who show significant differences to a set

significant_names = set()
for row in summary.data[1:]:  # Skip the header row
    if row[6] == True:
        significant_names.add(row[0])
        significant_names.add(row[1])
        
print("Significant actors:", significant_names)
# %% filter to just those directors in the "sigificant_names" set
df_actors_tukey = df_actor_anova_final[df_actor_anova_final["actor_1_name"].isin(significant_names)]

# %% (messy) means plot to visualise one way ANOVA

# calculate means and standard errors for each group
means = df_actors_tukey.groupby("actor_1_name")["imdb_score"].mean()
stderr = df_actors_tukey.groupby("actor_1_name")["imdb_score"].std() / np.sqrt(df_actors_tukey.groupby("actor_1_name")["imdb_score"].count())

# sort the means and standard errors from largest to smallest
sorted_means = means.sort_values(ascending=False)
sorted_stderr = stderr[sorted_means.index]

# Prepare data for plotting
sorted_names = sorted_means.index

plt.figure(figsize=(12, 8))
plt.errorbar(
    sorted_names, 
    sorted_means, 
    yerr = sorted_stderr, 
    fmt = "o",                
    color = "blue",          
    markersize = 6,           
    ecolor = "black",        
    capsize = 5,               
    linestyle = "None",       
    markeredgewidth = 2      
)

plt.title("")
plt.xlabel("Actor", fontsize = 20, fontweight = "bold")
plt.xticks(rotation = 45, fontsize = 16)
plt.ylabel("Average IMDb Score", fontsize = 20, fontweight = "bold")
plt.yticks(np.arange(1, 11, step = 1))
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("report_pdf/images/actor_tukey.png")
plt.show()

# %%
print(df_actor_anova_final[(df_actor_anova_final["actor_1_name"] == "Kate Winslet") | (df_actor_anova_final["actor_1_name"] == "Ryan Gosling")].groupby("actor_1_name")["imdb_score"].mean().sort_values(ascending = False))
# %%
