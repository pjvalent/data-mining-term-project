from matplotlib.pyplot import get
import twitter
import json
import datetime
import time 


def oauth_login():
    CONSUMER_KEY = 'fjRSNYpRexhGGEBRZ6wDtR46W'
    CONSUMER_SECRET = 'TsN5f7qN6SHUFliBptqqu4FFh3YLx6tisldkvFx5C065HiwaoE'
    OAUTH_TOKEN = '804135069988306947-WoNcUYrqIJsfcdKegraukBv3oF4hsKA'
    OAUTH_TOKEN_SECRET = 'x3O77abe4vDaQ40ad8h5SforRGoLY7G9kBjOiRuTBrnEj'

    auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                               CONSUMER_KEY, CONSUMER_SECRET)
    
    twitter_api = twitter.Twitter(auth=auth)
    return twitter_api



def get_tweets_before_invasion():

    parsed = []
    tweets = []

    with open("prev_ukraine_raw.json", "r") as f:
        for line in f:
            parsed.append(json.loads(line))
    for i in range(0, len(parsed)):
        print(json.dumps(parsed[i], indent=4, sort_keys=True))

    for i in range(0, len(parsed)):
        tweets.append(parsed[i]["content"])

    for i in range(0, len(tweets)):
        f = open("ukraine_tweets_before_invasion.txt", "a")
        f.write(tweets[i])
    
    f.close()

def get_tweets_after_invasion():
    
    parsed = []
    tweets = []

    with open("post_ukraine_raw.json", "r") as f:
        for line in f:
            parsed.append(json.loads(line))
    for i in range(0, len(parsed)):
        print(json.dumps(parsed[i], indent=4, sort_keys=True))

    for i in range(0, len(parsed)):
        tweets.append(parsed[i]["content"])

    for i in range(0, len(tweets)):
        f = open("ukraine_tweets_after_invasion.txt", "a")
        f.write(tweets[i])
    
    f.close()


if __name__ == "__main__":
    api = oauth_login()
    get_tweets_before_invasion()
    get_tweets_after_invasion()
