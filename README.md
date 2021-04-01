# BASIC VIRUS SIMULATOR
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
