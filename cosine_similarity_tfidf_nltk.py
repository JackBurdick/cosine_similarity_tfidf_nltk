import nltk
import string

# used for looping through folders/files
from os import listdir
from os.path import isfile, join

#Calc tfidf and cosine similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# All text entries to compare will appear here
BASE_INPUT_DIR = "./inputdata/"

def returnListOfFilePaths(folderPath):
    fileInfo = []
    listOfFileNames = [fileName for fileName in listdir(folderPath) if isfile(join(folderPath, fileName))]
    listOfFilePaths = [join(folderPath, fileName) for fileName in listdir(folderPath) if isfile(join(folderPath, fileName))]
    fileInfo.append(listOfFileNames)
    fileInfo.append(listOfFilePaths)
    return fileInfo

def create_docContentDict(filePaths):
    rawContentDict = {}
    for filePath in filePaths:
        with open(filePath, "r") as ifile:
            fileContent = ifile.read()
        rawContentDict[filePath] = fileContent
    return rawContentDict

def tokenizeContent(contentsRaw):
    tokenized = nltk.tokenize.word_tokenize(contentsRaw)
    return tokenized

def removeStopWordsFromTokenized(contentsTokenized):
    stop_word_set = set(nltk.corpus.stopwords.words("english"))
    filteredContents = [word for word in contentsTokenized if word not in stop_word_set]
    return filteredContents

def performPorterStemmingOnContents(contentsTokenized):
    porterStemmer = nltk.stem.PorterStemmer()
    filteredContents = [porterStemmer.stem(word) for word in contentsTokenized]
    return filteredContents

def removePunctuationFromTokenized(contentsTokenized):
    excludePuncuation = set(string.punctuation)
    
    # manually add additional punctuation to remove
    doubleSingleQuote = '\'\''
    doubleDash = '--'
    doubleTick = '``'

    excludePuncuation.add(doubleSingleQuote)
    excludePuncuation.add(doubleDash)
    excludePuncuation.add(doubleTick)

    filteredContents = [word for word in contentsTokenized if word not in excludePuncuation]
    return filteredContents

def convertItemsToLower(contentsRaw):
    filteredContents = [term.lower() for term in contentsRaw]
    return filteredContents

# process data without writing inspection file information to file
def processData(rawContents):
    cleaned = tokenizeContent(rawContents)
    cleaned = removeStopWordsFromTokenized(cleaned)
    cleaned = performPorterStemmingOnContents(cleaned)    
    cleaned = removePunctuationFromTokenized(cleaned)
    cleaned = convertItemsToLower(cleaned)
    return cleaned

# print TFIDF values in 'table' format
def print_TFIDF_for_all(term, values, fileNames):
    values = values.transpose() # files along 'x-axis', terms along 'y-axis'
    numValues = len(values[0])
    print('                ', end="")   #bank space for formatting output
    for n in range(len(fileNames)):
        print('{0:18}'.format(fileNames[n]), end="")    #file names
    print()
    for i in range(len(term)):
        print('{0:8}'.format(term[i]), end='\t|  ')     #the term
        for j in range(numValues):
            print('{0:.12f}'.format(values[i][j]), end='   ') #the value, corresponding to the file name, for the term
        print()

# write TFIDF values in 'table' format
def write_TFIDF_for_all(term, values, fileNames):
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
            outFile.write('{0:.12f}'.format(values[i][j])) #the value, corresponding to the file name, for the term
            outFile.write('   ')
        outFile.write("\n")

    outFile.close()

# TODO: modify this to build matrix then print from matrix form
def calc_and_print_CosineSimilarity_for_all(tfs, fileNames):
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

def calc_and_write_CosineSimilarity_for_all(tfs, fileNames):
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

def main(printResults=True):
    baseFolderPath = "./inputdata/"

    fileNames, filePathList = returnListOfFilePaths(baseFolderPath)

    rawContentDict = create_docContentDict(filePathList)

    # calculate tfidf
    tfidf = TfidfVectorizer(tokenizer=processData, stop_words='english')
    tfs = tfidf.fit_transform(rawContentDict.values())
    tfs_Values = tfs.toarray()
    tfs_Term = tfidf.get_feature_names()
    
    if printResults:
        # print results
        print_TFIDF_for_all(tfs_Term, tfs_Values, fileNames)
        calc_and_print_CosineSimilarity_for_all(tfs, fileNames)
    else:
        # write results to file
        write_TFIDF_for_all(tfs_Term, tfs_Values, fileNames)   
        calc_and_write_CosineSimilarity_for_all(tfs, fileNames)

if __name__ == "__main__":
    main()