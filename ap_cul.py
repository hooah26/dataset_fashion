import csv
import numpy as np

def AP(img_iou, Thres = 0.5):
    TP = np.zeros(len(img_iou))
    FP = np.zeros(len(img_iou))

    # 전체 ground truth box의 수
    # Recall값 분모
    npos = len(img_iou)

    for i in range(len(img_iou)):
        if float(img_iou[i]) >= Thres:
            #class가 일치하면 TP 불일치하면 FP
            TP[i] = 1
        else:
            FP[i] = 1

    acc_FP = np.cumsum(FP)
    acc_TP = np.cumsum(TP)

    rec = acc_TP / npos
    prec = np.divide(acc_TP, (acc_FP + acc_TP))

    [ap, mpre, mrec, _] = ElevenPointInterpolatedAP(rec, prec)

    r = {
        'AP': round(ap, 5),

    }

    return r


def ElevenPointInterpolatedAP(rec, prec):
    mrec = [e for e in rec]
    mpre = [e for e in prec]

    # recallValues = [1.0, 0.9, ..., 0.0]
    recallValues = np.linspace(0, 1, 11)
    recallValues = list(recallValues[::-1])
    rhoInterp, recallValid = [], []

    for r in recallValues:
        # r : recall값의 구간
        # argGreaterRecalls : r보다 큰 값의 index
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0
        # print(r, argGreaterRecalls)

        # precision 값 중에서 r 구간의 recall 값에 해당하는 최댓값
        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min():])

        recallValid.append(r)
        rhoInterp.append(pmax)

    ap = sum(rhoInterp) / 11

    return [ap, rhoInterp, recallValues, None]


f = open("Shapeless_eval_log.csv", "r")
rdr = csv.reader(f)
iou =[]
for line in rdr:
    # print(line)
    iou.append(line[1])


# print(iou[1:])
print("AP50 : " + str(AP(iou[1:],50.0)))
print("AP75 : " + str(AP(iou[1:],75.0)))
print("AP90 : " + str(AP(iou[1:],90.0)))

f.close()