from sacrebleu.metrics import BLEU, CHRF
import argparse
import os
import jieba
import fire

# python evaluation.py --method BLEU --ref ./translate_result/GLM/... --result ./translate_result/GLM/...
def read_txt(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return lines

def main(method:str, ref:str, result:str):
    result = read_txt(result)
    ref = read_txt(ref)
    #分词
    result = [i.strip() for i in result]
    ref = [i.strip() for i in ref]
    for i in range(len(ref)):
        if result[i]:
            result[i] = ' '.join(jieba.lcut(result[i]))
        if ref[i]:
            ref[i] = ' '.join(jieba.lcut(ref[i]))
    #评价
    if(method == 'BLEU'):
        bleu = BLEU()
        score = bleu.corpus_score(result, [ref]).score
        print(score)
    elif(method == 'CHRF'):
        chrf = CHRF()
        score = chrf.corpus_score(result, [ref]).score
        print(score)
        
if __name__ == "__main__":
    fire.Fire(main)