import numpy as np
from textwrap import wrap

blosum62 = {
    ('W', 'F'): 1, ('L', 'R'): -2, ('S', 'P'): -1, ('V', 'T'): 0,
    ('Q', 'Q'): 5, ('N', 'A'): -2, ('Z', 'Y'): -2, ('W', 'R'): -3,
    ('Q', 'A'): -1, ('S', 'D'): 0, ('H', 'H'): 8, ('S', 'H'): -1,
    ('H', 'D'): -1, ('L', 'N'): -3, ('W', 'A'): -3, ('Y', 'M'): -1,
    ('G', 'R'): -2, ('Y', 'I'): -1, ('Y', 'E'): -2, ('B', 'Y'): -3,
    ('Y', 'A'): -2, ('V', 'D'): -3, ('B', 'S'): 0, ('Y', 'Y'): 7,
    ('G', 'N'): 0, ('E', 'C'): -4, ('Y', 'Q'): -1, ('Z', 'Z'): 4,
    ('V', 'A'): 0, ('C', 'C'): 9, ('M', 'R'): -1, ('V', 'E'): -2,
    ('T', 'N'): 0, ('P', 'P'): 7, ('V', 'I'): 3, ('V', 'S'): -2,
    ('Z', 'P'): -1, ('V', 'M'): 1, ('T', 'F'): -2, ('V', 'Q'): -2,
    ('K', 'K'): 5, ('P', 'D'): -1, ('I', 'H'): -3, ('I', 'D'): -3,
    ('T', 'R'): -1, ('P', 'L'): -3, ('K', 'G'): -2, ('M', 'N'): -2,
    ('P', 'H'): -2, ('F', 'Q'): -3, ('Z', 'G'): -2, ('X', 'L'): -1,
    ('T', 'M'): -1, ('Z', 'C'): -3, ('X', 'H'): -1, ('D', 'R'): -2,
    ('B', 'W'): -4, ('X', 'D'): -1, ('Z', 'K'): 1, ('F', 'A'): -2,
    ('Z', 'W'): -3, ('F', 'E'): -3, ('D', 'N'): 1, ('B', 'K'): 0,
    ('X', 'X'): -1, ('F', 'I'): 0, ('B', 'G'): -1, ('X', 'T'): 0,
    ('F', 'M'): 0, ('B', 'C'): -3, ('Z', 'I'): -3, ('Z', 'V'): -2,
    ('S', 'S'): 4, ('L', 'Q'): -2, ('W', 'E'): -3, ('Q', 'R'): 1,
    ('N', 'N'): 6, ('W', 'M'): -1, ('Q', 'C'): -3, ('W', 'I'): -3,
    ('S', 'C'): -1, ('L', 'A'): -1, ('S', 'G'): 0, ('L', 'E'): -3,
    ('W', 'Q'): -2, ('H', 'G'): -2, ('S', 'K'): 0, ('Q', 'N'): 0,
    ('N', 'R'): 0, ('H', 'C'): -3, ('Y', 'N'): -2, ('G', 'Q'): -2,
    ('Y', 'F'): 3, ('C', 'A'): 0, ('V', 'L'): 1, ('G', 'E'): -2,
    ('G', 'A'): 0, ('K', 'R'): 2, ('E', 'D'): 2, ('Y', 'R'): -2,
    ('M', 'Q'): 0, ('T', 'I'): -1, ('C', 'D'): -3, ('V', 'F'): -1,
    ('T', 'A'): 0, ('T', 'P'): -1, ('B', 'P'): -2, ('T', 'E'): -1,
    ('V', 'N'): -3, ('P', 'G'): -2, ('M', 'A'): -1, ('K', 'H'): -1,
    ('V', 'R'): -3, ('P', 'C'): -3, ('M', 'E'): -2, ('K', 'L'): -2,
    ('V', 'V'): 4, ('M', 'I'): 1, ('T', 'Q'): -1, ('I', 'G'): -4,
    ('P', 'K'): -1, ('M', 'M'): 5, ('K', 'D'): -1, ('I', 'C'): -1,
    ('Z', 'D'): 1, ('F', 'R'): -3, ('X', 'K'): -1, ('Q', 'D'): 0,
    ('X', 'G'): -1, ('Z', 'L'): -3, ('X', 'C'): -2, ('Z', 'H'): 0,
    ('B', 'L'): -4, ('B', 'H'): 0, ('F', 'F'): 6, ('X', 'W'): -2,
    ('B', 'D'): 4, ('D', 'A'): -2, ('S', 'L'): -2, ('X', 'S'): 0,
    ('F', 'N'): -3, ('S', 'R'): -1, ('W', 'D'): -4, ('V', 'Y'): -1,
    ('W', 'L'): -2, ('H', 'R'): 0, ('W', 'H'): -2, ('H', 'N'): 1,
    ('W', 'T'): -2, ('T', 'T'): 5, ('S', 'F'): -2, ('W', 'P'): -4,
    ('L', 'D'): -4, ('B', 'I'): -3, ('L', 'H'): -3, ('S', 'N'): 1,
    ('B', 'T'): -1, ('L', 'L'): 4, ('Y', 'K'): -2, ('E', 'Q'): 2,
    ('Y', 'G'): -3, ('Z', 'S'): 0, ('Y', 'C'): -2, ('G', 'D'): -1,
    ('B', 'V'): -3, ('E', 'A'): -1, ('Y', 'W'): 2, ('E', 'E'): 5,
    ('Y', 'S'): -2, ('C', 'N'): -3, ('V', 'C'): -1, ('T', 'H'): -2,
    ('P', 'R'): -2, ('V', 'G'): -3, ('T', 'L'): -1, ('V', 'K'): -2,
    ('K', 'Q'): 1, ('R', 'A'): -1, ('I', 'R'): -3, ('T', 'D'): -1,
    ('P', 'F'): -4, ('I', 'N'): -3, ('K', 'I'): -3, ('M', 'D'): -3,
    ('V', 'W'): -3, ('W', 'W'): 11, ('M', 'H'): -2, ('P', 'N'): -2,
    ('K', 'A'): -1, ('M', 'L'): 2, ('K', 'E'): 1, ('Z', 'E'): 4,
    ('X', 'N'): -1, ('Z', 'A'): -1, ('Z', 'M'): -1, ('X', 'F'): -1,
    ('K', 'C'): -3, ('B', 'Q'): 0, ('X', 'B'): -1, ('B', 'M'): -3,
    ('F', 'C'): -2, ('Z', 'Q'): 3, ('X', 'Z'): -1, ('F', 'G'): -3,
    ('B', 'E'): 1, ('X', 'V'): -1, ('F', 'K'): -3, ('B', 'A'): -2,
    ('X', 'R'): -1, ('D', 'D'): 6, ('W', 'G'): -2, ('Z', 'F'): -3,
    ('S', 'Q'): 0, ('W', 'C'): -2, ('W', 'K'): -3, ('H', 'Q'): 0,
    ('L', 'C'): -1, ('W', 'N'): -4, ('S', 'A'): 1, ('L', 'G'): -4,
    ('W', 'S'): -3, ('S', 'E'): 0, ('H', 'E'): 0, ('S', 'I'): -2,
    ('H', 'A'): -2, ('S', 'M'): -1, ('Y', 'L'): -1, ('Y', 'H'): 2,
    ('Y', 'D'): -3, ('E', 'R'): 0, ('X', 'P'): -2, ('G', 'G'): 6,
    ('G', 'C'): -3, ('E', 'N'): 0, ('Y', 'T'): -2, ('Y', 'P'): -3,
    ('T', 'K'): -1, ('A', 'A'): 4, ('P', 'Q'): -1, ('T', 'C'): -1,
    ('V', 'H'): -3, ('T', 'G'): -2, ('I', 'Q'): -3, ('Z', 'T'): -1,
    ('C', 'R'): -3, ('V', 'P'): -2, ('P', 'E'): -1, ('M', 'C'): -1,
    ('K', 'N'): 0, ('I', 'I'): 4, ('P', 'A'): -1, ('M', 'G'): -3,
    ('T', 'S'): 1, ('I', 'E'): -3, ('P', 'M'): -2, ('M', 'K'): -1,
    ('I', 'A'): -1, ('P', 'I'): -3, ('R', 'R'): 5, ('X', 'M'): -1,
    ('L', 'I'): 2, ('X', 'I'): -1, ('Z', 'B'): 1, ('X', 'E'): -1,
    ('Z', 'N'): 0, ('X', 'A'): 0, ('B', 'R'): -1, ('B', 'N'): 3,
    ('F', 'D'): -3, ('X', 'Y'): -1, ('Z', 'R'): 0, ('F', 'H'): -1,
    ('B', 'F'): -3, ('F', 'L'): 0, ('X', 'Q'): -1, ('B', 'B'): 4
}
newSequence = []
Sequence=input("enter the sequence")
WordLength=int(input("enter the Word Length"))
# convert list to string
def Convert(str):
    newlist=[]
    newlist[:0]=str
    return newlist

def listToString(s):
    str1 = ""
    for ele in s:
        str1 += ele
    return str1

#step 1 Removing low complexity

def cancel_repeated(sequence, min, max, index):
    if max + 1 >= len(sequence) + 1 or max + 1 == len(sequence) + 1:
        if min + 1 != len(sequence):
            newSequence.insert(index, sequence[min])
            cancel_repeated(sequence, min + 1, max + 1, index + 1)
    elif sequence[min] == sequence[max] and sequence[min + 1] == sequence[max + 1]:
        cancel_repeated(sequence, min + 2, max + 2, index)
    elif sequence[max] == sequence[max - 2]:
        cancel_repeated(sequence, min + 2, max + 2, index)
    elif sequence[min] == sequence[max] and sequence[min + 1] != sequence[max + 1]:
        cancel_repeated(sequence, min + 1, max + 2, index)
    else:
        newSequence.insert(index, sequence[min])
        cancel_repeated(sequence, min + 1, max + 1, index + 1)
    return newSequence

# step 2 , Make a W-lette

#seq = cancel_repeated(Sequence, 0, 2, 0)
print("New Sequence ")
#print(seq)

def QuerySequence (seq):
    words = []
    for i in range(0, len(seq)):
        str = ""
        for j in range(i, i + 3):
            if (j == len(seq)):
                break
            else:
                str += seq[j]
            if (len(str) == 3):
                words.append(str)
    return words


#step 3 , all possible matches

aminoacids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',  'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def NeighborhoodWords(word, aminoacids):
    str1 = ""
    Neighbors = ""
    possibles = []

    for i in range(0, len(word)):
        x =[]
        x.append(word[i])
        for j in range(0, WordLength):
            for k in range(0, len(aminoacids)):
                str1 = word[i]
                Neighbors = list(str1)
                Neighbors[j] = aminoacids[k]
                str1 = listToString(Neighbors)
                x.append(str1)
        possibles.append(x)

    return possibles

word=[]
word = QuerySequence(Sequence)
print("The Words")
r = NeighborhoodWords(word, aminoacids)
print(r)



Threshold=input("Enter the Threshold")
# Step3 and 4
#Calculating the score
def CreatingSeed(r, word):
    Seeds = []
    T = []
    for m in range(len(word)):
        for n in range(1,60):
            T = []
            score = 0
            for i in range(WordLength):
                name1 = word[m][i], r[m][n][i]
                name2 = r[m][n][i], word[m][i]
                if name1 in blosum62:
                    score = blosum62[word[m][i], r[m][n][i]] + score
                if name2 in blosum62:
                    score = blosum62[r[m][n][i], word[m][i]] + score
        if score >= int(Threshold):
            T.append(word[m])
            T.append(score)
            Seeds.append(T)
    return Seeds
s = CreatingSeed(r, word)
print("The Seeds is")
print(s)

#Step5
Database = [
 "PQGMMKSFFLVVTILALTLPFLGAQEQNQEQPIRCEKDERFFSDKIAKYIPIQYVLSRYPSYGLNYYQQKPVALINNQFLPYPYYAKPAAVRSPAQILQWQVLSNTVPAKSCQAQPTTMARHPHPHLSFMAIPPKKNQDKTEIPTINTIASGEPTSTPTTEAVESTVATLEDSPEVIESPPEINTVQVTSTAV",
 "MKLFWLLFTIGFCWAQYSSNTQQGRTSIVHLFEWRWVDIALECERYLAPKGFGGVQVSPPNENVAIHNPFRPWWERYQPVSYKLCTRSGNEDEFRNMVTRCNNVGVRIYVDAVINHMCGNAVSAGTSSTCGSYFNPGSRDFPAVPYSGWDFNDGKCKTGSGDIENYNDATQVRDCRLSGLLDLALGKDYVRSKIAEYMNHLIDIGVAGFRIDASKHMWPGDIKAILDKLHNLNSNWFPEGSKPFIYQEVIDLGGEPIKSSDYFGNGRVTEFKYGAKLGTVIRKWNGEKMSYLKNWGEGWGFMPSDRALVFVDNHDNQRGHGAGGASILTFWDARLYKMAVGFMLAHPYGFTRVMSSYRWPRYFENGKDVNDWVGPPNDNGVTKEVTINPDTTCGNDWVCEHRWRQIRNMVNFRNVVDGQPFTNWYDNGSNQVAFGRGNRGFIVFNNDDWTFSLTLQTGLPAGTYCDVISGDKINGNCTGIKIYVSDDGKAHFSISNSAED",
 "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELSKDIGSESTEDQAMEDIKQMEAESISSSEEIVPNSVEQKHIQKEDVPSERYLGYLEQLLRLKKYKVPQLEIVPNSAEERLHSMKEGIHAQQKEPMIGVNQELAYFYPELFRQFYQLDAYPSGAWYYVPLGTQYTDAPSFSDIPNPIGSENSEKTTMPLW" ,
 "MKFFIFTCLLAVALAKNTMEHVSSSEESIISQETYKQEKNMAINPSKENLCSTFCKEVVRNANEEEYSIGSSSEESAEVATEEVKITVDDKHYQKALNEINQFYQKFPQYLQYLYQGPIVLNPWDQVKRNAVPITPTLNREQLSTSEENSKKTVDMESTEVFTKKTKLTEEEKNRLNFLKKISQRYQKFALPQYLKTVYQHQKAMKPWIQPKTKVIPYVRYL" ,
 ]

def WordHits(Seeds):
    L = []
    for i in range(0, len(Seeds)):
        for x in range(0, len(Database)):
            for z in range(0, len(Database[x]) - WordLength):
                W = ""
                DbL = []
                for y in range(z, z + WordLength):
                    W = W + Database[x][y]
                if Seeds[i][0] == W:
                    DbL.append(W)
                    DbL.append(Seeds[i][1])
                    L.append(DbL)
    return L

l = WordHits(s)
print("Words Hits at")
print(l)
HSPThreshold=input("Enter the HSP Threshold")
HSP=[]
#step 6

for i in range(len(l)):
    for x in range(len(r)):
        for z in range(1,len(r[x])):
           if l[i][0]==r[x][z]:
               for n in range(len(Sequence) - WordLength + 1):
                   gaber = ""
                   Dw=""
                   sc=0
                   hsp=[]
                   WHILE = True
                   for m in range(n,WordLength):
                       gaber = gaber + Sequence[m]
                       if gaber==r[x][0]:
                           while WHILE==True:
                               if n - 1 >= 0 and n + WordLength + 1<len(Sequence):
                                   for d in range(len(Database)):
                                       for w in range(len(gaber)):
                                           Dw=Dw+Database[d]
                                           if Dw==r[x][1] and d-1>=0and d+ WordLength + 1<len(Database):
                                               Dw= Database[d-1]+Dw+ Database[d+ WordLength + 1]
                                               gaber = Sequence[n - 1] + gaber + Sequence[n + WordLength + 1]
                                               sc = blosum62[Database[d-1], Sequence[n - 1]] + l[i][1]
                                               sc = blosum62[Database[d + WordLength + 1], Sequence[n + WordLength + 1]] + sc
                                               if sc<HSPThreshold:
                                                   hsp.append(gaber)
                                                   hsp.append(sc)
                                                   HSP.append(hsp)
                                                   WHILE=False
                               else:
                                   hsp.append(r[x][z])
                                   hsp.append(l[i][1])
                                   HSP.append(hsp)
                                   WHILE = False

print(HSP)



