import nltk
import string

# used for looping through folders/files
from os import listdir
from os.path import isfile, join

#print results
from tabulate import tabulate

#Calc tfidf and cosine similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


######################################################## folder/file
def returnFileContents( filePath ):
    ifile = open( filePath, "r" )
    fileContent = ifile.read()
    ifile.close()
    return fileContent

########################################################  processing functions


def returnListOfFilePaths( folderPath ):
    fileInfo = []
    listOfFileNames = [ fileName for fileName in listdir(folderPath) if isfile(join(folderPath, fileName)) ]
    listOfFilePaths = [ join(folderPath, fileName) for fileName in listdir(folderPath) if isfile(join(folderPath, fileName)) ]
    fileInfo.append(listOfFileNames)
    fileInfo.append(listOfFilePaths)
    return fileInfo


def removeStopWordsFromTokenized( tokenizedContents ):
    stop_word_set = set( nltk.corpus.stopwords.words("english") )
    filteredContents = [ word for word in tokenizedContents if word not in stop_word_set ]
    return filteredContents


def performPorterStemmingOnContents( contentsRaw ):
    porterStemmer = nltk.stem.PorterStemmer()
    filteredContents = [ porterStemmer.stem(word) for word in contentsRaw ]
    return filteredContents


def removePunctuationFromTokenized( contentsRaw ):
    excludePuncuation = set( string.punctuation )
    filteredContents = [ word for word in contentsRaw if word not in excludePuncuation ]
    return filteredContents


def convertItemsToLower( contentsRaw ):
    filteredContents = [ term.lower() for term in contentsRaw ]
    return filteredContents

###############################################################################





def processData( rawContents ):
    rawContents = nltk.tokenize.word_tokenize( rawContents )        # word tokenize
    rawContents = convertItemsToLower( rawContents )                # convert to lowercase
    rawContents = removeStopWordsFromTokenized( rawContents )       # remove stop words
    rawContents = removePunctuationFromTokenized( rawContents )     # remove punctuation ??
    rawContents = performPorterStemmingOnContents( rawContents )    # stemming

    processedContents = rawContents
    return processedContents


def create_docContentDict( filePathList ):
    rawContentDict = {}
    for document in filePathList:
        text = returnFileContents(document)
        rawContentDict[document] = text
    return rawContentDict



######################################################### TFIDF

# this is really ugly right now
def printTFIDF( matrixInfo ):
    for i in matrixInfo:
        term = i[0]
        freq = i[1].toarray()[0]
        print(i[0], "\t:", freq[0], freq[1], freq[2], freq[3], freq[4], sep='\t\t')


def print_TFIDF_dict( term, values, fileNames ):
    values = values.transpose()

    # create a list of dict [ { 'filename': value, filename2 : value, .... }, ... ]
    termValueDict_List = []
    # loop through each set of five values per term
    for vals in values:
        keyInfo = {}
        j = 0
        # loop through each value in the set of five values
        for i in vals:
            keyInfo[ fileNames[j] ] = i
            j = j+1
        termValueDict_List.append(keyInfo)

    testZip = zip(term, termValueDict_List)
    testZip_list = list(testZip)

    #print(testZip_list)
    print(tabulate(testZip_list))


def print_TFIDF( term, values, fileNames ):
    values = values.transpose()
    testZip = zip(term, values)
    testZip_list = list(testZip)

    #print(testZip_list)
    print(tabulate(testZip_list))

#use this to build a list --> use tabulate to print
def print_TFIDF_custom( term, values, fileNames ):
    values = values.transpose()
    numValues = len(values[0])
    #print(fileNames[3])
    print('                ', end="")
    for n in range(len(fileNames)):
        print('{0:18}'.format(fileNames[n]), end="")
    print()
    for i in range(len(term)):
        print('{0:8}'.format(term[i]), end='\t|  ')
        for j in range(numValues):
            print( '{0:.12f}'.format(values[i][j]), end='   ' )
        print()



######################################################### COSINE SIMILARITY

#should modify this to build matrix then print from matrix form
def calcCosineSimilarity( tfs, fileNames ):
    #print(cosine_similarity(tfs[0], tfs[1]))
    print("\n\n\n========COSINE SIMILARITY====================================================================\n")
    numFiles = len(fileNames)
    names = []
    print('                   ', end="")
    for i in range(numFiles):
        if i == 0:
            for k in range(numFiles):
                print(fileNames[k], end='   ')
            print()

        print(fileNames[i], end='   ')
        for n in range(numFiles):
            #print(fileNames[n], end='\t')
            matrixValue = cosine_similarity(tfs[i], tfs[n])
            numValue = matrixValue[0][0]
            #print(numValue, end='\t')
            names.append(fileNames[n])
            print(" {0:.8f}".format(numValue), end='         ')
            #(cosine_similarity(tfs[i], tfs[n]))[0][0]

        print()
    print("\n\n=============================================================================================\n")

#############################################################################


def main():
    baseFolderPath = "../hw1_datasets/"

    fileInfo = returnListOfFilePaths( baseFolderPath )
    fileNames = fileInfo[0]
    filePathList = fileInfo[1]


    rawContentDict = create_docContentDict( filePathList )

    tfidf = TfidfVectorizer(tokenizer=processData)
    tfs = tfidf.fit_transform( rawContentDict.values() )
    tfs_Values = tfs.toarray()
    tfs_Term = tfidf.get_feature_names()

    #print_TFIDF_dict( tfs_Term, tfs_Values, fileNames )
    #print_TFIDF( tfs_Term, tfs_Values, fileNames )
    print_TFIDF_custom( tfs_Term, tfs_Values, fileNames )


    calcCosineSimilarity( tfs, fileNames )



if __name__ == '__main__':
    main()







# perform file preprocessing
# tokenize
# remove stopwords
# remove punctuation
# perform stemming
# def returnPreProcessedFileContents( individualFile ):
#     rawContents = returnFileContents( individualFile )              # raw file contents
#
#     rawContents = nltk.tokenize.word_tokenize( individualFile )     # word tokenize
#     rawContents = convertItemsToLower( rawContents )                # convert to lowercase
#     rawContents = removeStopWordsFromTokenized( rawContents )       # remove stop words
#     rawContents = removePunctuationFromTokenized( rawContents )     # remove punctuation??
#     rawContents = performPorterStemmingOnContents( rawContents )    # stemming
#     processedContents = rawContents
#     return processedContents



#print( tabulate(testZip, headers=['term', 'freq']) )
#print( tabulate(tfs.toarray()) )
