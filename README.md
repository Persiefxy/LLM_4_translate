# LLM_4_translate
A demo of English to Chinese translation using three LLMs(ChatGLM, LlaMa, Chinese_LlaMa) is implemented here, in addition to comparing the translation ability of these three LLMs using two benchmarks(BLEU, CHRF).

## Translate
Run
```
python translate.py --Model LlaMa --Dataset ./dataset/wmt22en_zh/en_zh.json --Output result --Ref ref --Output_dir ./translate_result/
```
--Model denotes the model selected(LlaMa, GLM, Chinese_LlaMa).

--Dataset indicates the location of the dataset, the data needs to be saved in a json file based on the Chinese and English parts.

--Output is the name of the file where the translation results are stored.

--Ref is the name of the file where the reference is stored.

--Output_dir is the folder where Output and Reference are located.

## Evaluate
```
python evaluation.py --method BLEU --ref ./translate_result/GLM/... --result ./translate_result/GLM/...
```
--method denotes the chosen method of evaluation.

--ref is the the file where the reference is stored.

--result is the file where the translation results are stored.

## Result
* Result on WMT20

| Method| BLEU | CHRF |
| ------ | ------ | ------ |
| LlaMa2-7b| 5.52  | 12.42  |
| Chinese_LlaMa2-7b  | 7.99  | 12.35  |
| ChatGLM-6b  | 17.06  | 21.84  |

* Result on WMT22

| Method| BLEU | CHRF |
| ------ | ------ | ------ |
| LlaMa2-7b|  4.56 | 14.14  |
| Chinese_LlaMa2-7b  |  6.01 | 11.98  |
| ChatGLM-6b  |19.33   |  29.19 |
