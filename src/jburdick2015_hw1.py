import nltk
import string

# used for looping through folders/files
from os import listdir
from os.path import isfile, join

#print results -- could be used for formatting
#from tabulate import tabulate

#Calc tfidf and cosine similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


######################################################## folder/file
def returnFileContents( filePath ):
    ifile = open( filePath, "r" )
    fileContent = ifile.read()
    ifile.close()
    return fileContent


def write_results( filePath, contents, title ):
    oFile = open(filePath, 'a')
    contentLength = len(contents)
    titleLine = "------------------------------------ " + title + " --> contents length: " + str(contentLength) + "\n"
    oFile.write(titleLine)
    body = str(contents) + "\n"
    oFile.write(body)
    oFile.write("-------------------------------------------------------------\n\n")
    oFile.close()
    #print("done")

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
    #manual punctuation to remove
    doubleSingleQuote = '\'\''
    doubleDash = '--'
    doubleTick = '``'

    excludePuncuation.add(doubleSingleQuote)
    excludePuncuation.add(doubleDash)
    excludePuncuation.add(doubleTick)

    filteredContents = [ word for word in contentsRaw if word not in excludePuncuation ]
    return filteredContents


def convertItemsToLower( contentsRaw ):
    filteredContents = [ term.lower() for term in contentsRaw ]
    return filteredContents

###############################################################################

def processData( rawContents ):
    resultsDirectory = "../results/"
    filePath = resultsDirectory + "process_text.txt"

    rawContents = nltk.tokenize.word_tokenize( rawContents )        # word tokenize
    write_results( filePath, rawContents, "word_tokenize" )
    rawContents = removeStopWordsFromTokenized( rawContents )       # remove stop words
    write_results( filePath, rawContents, "removeStopWordsFromTokenized" )
    rawContents = performPorterStemmingOnContents( rawContents )    # stemming
    write_results( filePath, rawContents, "performPorterStemmingOnContents" )
    rawContents = removePunctuationFromTokenized( rawContents )     # remove punctuation ??
    write_results( filePath, rawContents, "removePunctuationFromTokenized" )
    rawContents = convertItemsToLower( rawContents )                # convert to lowercase
    write_results( filePath, rawContents, "convertItemsToLower" )

    processedContents = rawContents
    return processedContents


def create_docContentDict( filePathList ):
    rawContentDict = {}
    for document in filePathList:
        text = returnFileContents(document)
        rawContentDict[document] = text
    return rawContentDict

######################################################### TFIDF

#print TFIDF values in 'table' format
def print_TFIDF_custom( term, values, fileNames ):
    values = values.transpose() # files along 'x-axis', terms along 'y-axis'
    numValues = len(values[0])
    print('                ', end="")   #bank space for formatting output
    for n in range(len(fileNames)):
        print('{0:18}'.format(fileNames[n]), end="")    #file names
    print()
    for i in range(len(term)):
        print('{0:8}'.format(term[i]), end='\t|  ')     #the term
        for j in range(numValues):
            print( '{0:.12f}'.format(values[i][j]), end='   ' ) #the value, corresponding to the file name, for the term
        print()

#write TFIDF values in 'table' format
def write_TFIDF_custom( term, values, fileNames ):
    filePath = "../results/tfid.txt"
    outFile = open(filePath, 'a')
    title = "TFIDF\n"
    outFile.write(title)
    values = values.transpose() # files along 'x-axis', terms along 'y-axis'
    numValues = len(values[0])
    outFile.write('               \t')   #bank space for formatting output
    for n in range(len(fileNames)):
        outFile.write('{0:18}'.format(fileNames[n]))    #file names
    outFile.write("\n")
    for i in range(len(term)):
        outFile.write('{0:15}'.format(term[i]))     #the term
        outFile.write('\t|  ')
        for j in range(numValues):
            outFile.write( '{0:.12f}'.format(values[i][j]) ) #the value, corresponding to the file name, for the term
            outFile.write('   ')
        outFile.write("\n")

    outFile.close()

######################################################### COSINE SIMILARITY

#should modify this to build matrix then print from matrix form
def print_CosineSimilarity( tfs, fileNames ):
    #print(cosine_similarity(tfs[0], tfs[1]))
    print("\n\n\n========COSINE SIMILARITY====================================================================\n")
    numFiles = len(fileNames)
    names = []
    print('                   ', end="")    #formatting
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

def write_CosineSimilarity( tfs, fileNames ):
    filePath = "../results/cosine_similarity.txt"
    outFile = open(filePath, 'a')
    title = "COSINE SIMILARITY\n"
    outFile.write(title)
    numFiles = len(fileNames)
    names = []
    outFile.write('                   ')
    for i in range(numFiles):
        if i == 0:
            for k in range(numFiles):
                outFile.write(fileNames[k])
                outFile.write('   ')
            outFile.write("\n")
        outFile.write(fileNames[i])
        outFile.write('   ')

        for n in range(numFiles):
            matrixValue = cosine_similarity(tfs[i], tfs[n])
            numValue = matrixValue[0][0]
            names.append(fileNames[n])
            outFile.write('{0:.8f}'.format(numValue))
            outFile.write('         ')
            #(cosine_similarity(tfs[i], tfs[n]))[0][0]

        outFile.write("\n")

    outFile.close()

#############################################################################


def main():
    baseFolderPath = "../inputdata/"

    fileNames,filePathList = returnListOfFilePaths( baseFolderPath )

    rawContentDict = create_docContentDict( filePathList )

    tfidf = TfidfVectorizer(tokenizer=processData, stop_words='english')
    tfs = tfidf.fit_transform( rawContentDict.values() )

    tfs_Values = tfs.toarray()
    tfs_Term = tfidf.get_feature_names()

    #print_TFIDF_custom( tfs_Term, tfs_Values, fileNames )  #print results to terminal
    write_TFIDF_custom( tfs_Term, tfs_Values, fileNames )   #write results to file

    #print_CosineSimilarity( tfs, fileNames )
    write_CosineSimilarity( tfs, fileNames )


if __name__ == '__main__':
    main()
