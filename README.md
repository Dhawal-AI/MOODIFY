# MOODIFY
- [x] Week 1
- [x] Week 2
- [x] Week 3
- [x] Week 4
- [x] Week 5-8 Implementation
## WEEK 1
Made a basic virus simulator using the python we already knew plus some newer library functions. Also learnt Matplotlib to make plots and also learnt using Jupyter notebooks efficiently. 
In this task, I start by creating an 100 x 150 matrix with one of it's entries as 1.<br />
WHILE CODING I FOLLOWED THE FOLLOWING ORDERS<br />
In each iteration 8 random entries are swapped with 8 other entries,so first I decided to code for the swap algorithm using arrays and this took the most debugging and help<br />
The nearest neighbours of an infected entry (1) have a 0.25 probability of getting infected,for this I just used random.choice<br />
The second nearest neighbours of an infected entry have a 0.08 probability of getting infected<br />
The iterations go on until all the entries of the matrix become 1<br />
<br />
The result obtained is plotted in 2 graphs:<br />
<br />
This graph depicts the number of ones in the matrix corresponding to each iteration.<br />
This graph depicts the number of ones added in each iteration . The peak obtained for this output was 838<br />
The graphs have been put up as plots1.png file and plots2.png file.
## WEEK 2
We learnt in depth about plots and plotting the data into various diff types of plots to understand what was going on with the data. Also learnt data cleaning and making it finer and better. Learning bar graphs arent it and sometimes violin plots can be handy was fun.
## WEEK 3
A detailed report was made on the google ML standardisation course we did this week was already made as an assignment and can be checked out in the assignment 3 folder(also uploaded as a README). It was broadly about reducing the losses in the cost function.
## WEEK 4
Started with ML principles and learnt upto logistic regression and linear regression.<br />
This week we had two problems for assignments<br />
Problem 1 was without using any standard models or pre exisitng codes, to make an entire pricing model and this took a little longer than expected. I think i took this to week 3 and completed it with the most difficulty. The plots in this were easy.<br />
Problem 2 was much easier as we could alreayd use library and pre existing functions which were inbuilt in Keras and Tensorflow. hence making training and Test set wasnt an issue. Plots were also basic and simpler. <br />
Also basic principles of Data frames and making CSV files were also learnt which came in very handy in the later half of the project.
## WEEK 5-8
After the learning was done, we were finally divided into teams and my preference was MUSIC API. Since it had only initial load hence I only had one partner as Tanirika Roy. In the subsequent weeks, we first worked on making a pipeline for continuous fetching of songs and made a test set of 600 songs based on mood. We also hand picked the 16 features for our model ranging from danceability, energy, tone, pitch,duration etc.<br /> 
After picking out the features, we prepared a code to first generate the access token for the user and then another code to fetch the songs with their feature data using the SONG IDS. This was cumbersome and we struggled a bit with the dataframes as we couldnt make a list but later append command came to the rescue.<br />
We Fetched songs dataset with features such that they can be further classified based on mood. Ensure a data pipeline that fetches song details along with features <br />
We scrapped through tonnes of API calls and integrated them with SPOTIPY(THE PYTHON SPOTFIY LIB) and then tried sorting the date.<br />
Found a dataset or api that contains songs and the moods associated with them.<br />
Came up with an algorithm where in we had the song features and sampled out songs from a large chunk of song database that matched best with the obtained features. This was a very crucial part since these are the songs that would be suggested to the user.<br />
For the final step and here we faced so many issues with the WEB API and the coding integration with python in general but somehow managed was the USER AUTH PART. We had to:<br />
Authenticate User <br />
Find User’s song history <br />
Fetch song features <br />
Make a playlist in spotify <br />
We kinda succeeded with some bugs here and there and that was MOODIFY for us. The rest was taken care by the other teams.


