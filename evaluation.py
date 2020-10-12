import sys
import ast

def main():
    if len(sys.argv) < 3:
        print( "Usage: python POS-S.py <tagger output> <reference file>")
        exit(1)

    tag_out = open(sys.argv[1], "r", encoding='utf-8-sig')
    user_sentences = tag_out.readlines()
    tag_out.close()

    infile = open(sys.argv[2], "r", encoding='utf-8-sig')
    correct_sentences = infile.readlines()
    infile.close()

    num_correct = 0
    total = 0
    for i in range(len(user_sentences)):
      user = user_sentences[i].split('\n')[0]
      correct = correct_sentences[i].split('\n')[0]
      if user == correct:
        num_correct += 1
      total += 1
        
    score = float(num_correct) / total * 100

    print("Percent correct tags:", score)

if __name__=='__main__':
    main()