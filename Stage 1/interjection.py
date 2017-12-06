# Feature idea and list by Madiha. Code from Discourse Scoring by Kevin Swanberg.

def interjection_score(tweets):

    with open('interjection_list.txt', 'r') as file:
        interjection_list = file.read()
        interjection_list = interjection_list.split('\n')

        interjection_score_list = []

    for tweet in tweets:
        tweet = tweet.split()
        count = 0
        interjection_count = 0
        for word in tweet:
            if word in interjection_list:
                interjection_count += 1
            count += 1

        if interjection_count == 0:
            interjection_score = 0
        else:
            interjection_score = (interjection_count / count)

        interjection_score_list.append(interjection_score)

    return interjection_score_list

