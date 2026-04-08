import math
from collections import defaultdict

def bleu_score(candidate, reference, max_n):
    """
    Compute the BLEU score for a candidate translation.
    """
    c =  len(candidate)
    r = len(reference)

    if c == 0:
        return 0
    
    precisions = defaultdict(list)
    
    for i in range(1, max_n+1):
        for j in range(0, len(candidate)):
            if j + i <= len(candidate):
                precisions[i].append(candidate[j: j+i] == reference[j: j+i])
    
    BP = 1 if len(candidate) >= len(reference) else math.exp(min(0, 1 - (r/c)))
    
    bleu_score = BP * math.exp(sum([math.log(sum(v)/len(v)) if len(v) > 0 else 0 for k, v in precisions.items()]) / max_n)

    return bleu_score