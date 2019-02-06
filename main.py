from collections import Counter, defaultdict
import json
from random import random
import nltk

# need to parse JSON to file that is usable
# concatenate all tweets into one file
def readTweetsFromJSONIntoStr(fileName, order, addPadding):
    outputStr = ""
    parsedJSON = json.load(open(fileName))

    for jsonObj in parsedJSON:
        # don't count retweets, quotes, or replies. 
        if 'retweeted_status' in jsonObj or 'quoted_status' in jsonObj or ('in_reply_to_user_id' in jsonObj and jsonObj['in_reply_to_user_id'] != None):
            continue
        
        textKey = ""
        if 'full_text' in jsonObj: textKey = 'full_text'
        elif 'text' in jsonObj: textKey = 'text'
        
        # add ~~~ padding for text generation if necessary
        if addPadding:
            outputStr += ("~" * order) 
        # add tweet text to output
        outputStr += replaceLineFeeds(jsonObj[textKey]) + " "
            
    return outputStr 

# loops through files and compiles all tweets into
def readTweetsFromListOfFiles(fileList, createFileName, order, addPadding):
    fileStr = ""
    for fileName in fileList:
        fileStr += readTweetsFromJSONIntoStr(fileName, order, addPadding)

    # once you have all the tweets together, write them out to a new file
    with open(createFileName, 'w', encoding="utf-8") as f:
        f.write(fileStr)

# just replace line feeds with spaces.
def replaceLineFeeds(text):
    return text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").replace("\n\n", " ")

# creates a dictionary with keys consisting of all the unique n character combinations where the value is an array of tuples. 
# The tuples are <letter, decimal> where the letter is the next letter in the character sequence and the decimal is the probability that that letter will come next in the sequence
def trainLanguageModel(fname, order=4):
    data = open(fname, encoding="utf-8").read()
    lm = defaultdict(Counter)
    pad = "~" * order
    data = pad + data
    for i in range(len(data)-order):
        history, char = data[i:i+order], data[i+order]
        lm[history][char]+=1
    def normalize(counter):
        s = float(sum(counter.values()))
        return [(c,cnt/s) for c,cnt in counter.items()]
    outlm = {hist:normalize(chars) for hist, chars in lm.items()}
    return outlm

# this simply takes the model, looks at the last n characters and then generates the next letter based on the distibution.
# it's not completely random - this is weighted based on the frequency in the text
def generateLetter(lm, history, order):
        history = history[-order:]
        dist = lm[history]
        x = random()
        for c,v in dist:
            x = x - v
            if x <= 0: return c

# takes a model, the number of characters the distribution is based on, and a number of characters to generate
# just creates a history and then generates letter by letter, updating the history, until we reach the desired number of letters
def generateText(lm, order, nletters=1000):
    history = "~" * order
    out = []
    for i in range(nletters):
        c = generateLetter(lm, history, order)
        history = history[-order:] + c

        # finish tweet if we have reached an end marker
        if (c == "~" and i > order):
            break
        out.append(c)
    return "".join(out)

# takes model and generates
def generateTweets(model, numCharsToLookBack, numCharsToGenerate, tweetFileName, numTweetsToGenerate):
    for x in range(numTweetsToGenerate):
        tweet = generateText(model, numCharsToLookBack, numCharsToGenerate)

    # save them to a file
    with open(tweetFileName, 'a+', encoding='utf-8') as f:
        f.write(tweet + "\r\n#####################################\r\n")
    print(tweet)
    print("\r\n#####################################\r\n")

numCharsToLookBack = 8
numCharsToGenerate = 240
numTweetsToGenerate = 15
createFileName = 'alltweets2012to2018NoPadding.txt'
tweetFileName = 'trumpTweets2012to2018.txt'
fileList = ['master_2012.json', 'master_2013.json', 'master_2014.json', 'master_2015.json', 'master_2016.json', 'master_2017.json', 'master_2018.json']
# fileList = ['master_2018.json']

# readTweetsFromListOfFiles(fileList, createFileName, numCharsToLookBack, False)

# model = trainLanguageModel(createFileName, numCharsToLookBack)
# generateTweets(model, numCharsToLookBack, numCharsToGenerate, tweetFileName, numTweetsToGenerate)
allTweets = ""
with open(createFileName, 'r', encoding='utf-8') as f:
    allTweets = f.read()

from nltk.collocations import *
ignoredWords = nltk.corpus.stopwords.words('english')
tokens = nltk.word_tokenize(allTweets)
finder = BigramCollocationFinder.from_words(tokens, window_size=5)
finder.apply_freq_filter(10)
finder.apply_word_filter(lambda w: w.lower() in ignoredWords)
finder.apply_ngram_filter(lambda w1, w2: 'dems' not in (w1.lower(), w2.lower()))
print(finder.nbest(nltk.collocations.BigramAssocMeasures().pmi, 10))
