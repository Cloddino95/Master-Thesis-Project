# TODO:
"""
1) "p > 0.05 when comparing accuracy rates with a t-test" --> I need to implement then the t-test for each study

"""

# TODO FROM RUI:
"""
1) In the psychometric study the **Y variable** to be forecasted is: (look at pag 10)
 a) the mean of the ratings from THE 9 DIMENSIONS dataset? (NOO! it overfits)
 b) the mean of the ratings from THE 300 DIMENSIONS dataset? (since it is from -110 to 100, i scaled it to -1 to 7 and the result does not change)

2) how do I perform the singular values forecast for the 300 dimensions dataset? 
3) do i have to do cross validation for the psychometric study with the same settings of the rating one? I am afraid it overfit with nplit=10 and repeat=1000
4) do I have to do the full analysis performed by Bathia?
5) Emotion Words analyses (pag 14) they have 14k words and want to assess the rating of these with the model created, however i can create the vector representation of these words but i do not have the ratings so what i use as y variable? for the 300 dimensions dataset i used the mean of the ratings give by the participants, but i do not have the ratings of the 14k words

    """