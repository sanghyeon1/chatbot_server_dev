import cv2


config_file = '../chatbot_django_proj/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)  # 데이터셋으로부터 모델 생성.
# 파라미터 : 학습된 가중치(weight)를 포함하는 model, config file : 네트워크 구성이 포함된 텍스트파일

classLabels = []
file_name = '../Person_Num_Recognize/Lables.txt'

# Labels 파일에 포함된 데이터이름을 불러옴.
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(320, 320)  # Set input size for frame. (width, height)
model.setInputScale(1.0 / 127.5)  # Set scalefactor value for frame. (double scale)
model.setInputMean((127.5, 127.5, 127.5))  # 프레임의 평균값을 설정합니다.
model.setInputSwapRB(True)  # Set flag swapRB for frame. (bool swapRB)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open Camera")

font_scale = 1
font = cv2.FONT_HERSHEY_PLAIN  # small size sans-serif font : 글꼴을 설정함.


def person_num_return(c):
    return c - 1


while True:
    ret, frame = cap.read()
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)
    # confThreshold : 신뢰도를 기준으로 상자를 필터링하는 데 사용되는 임계값입니다.
    # detect() returns (classIds, confidences, boxes)

    c = 1
    if len(ClassIndex) != 0:

        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            # zip() : 파라미터들의 데이터를 하나씩 짝지어줌. 여기서는 각각 ClassInd, conf, boxes에 할당됨.
            # flatten : 2차원 배열을 1차원 배열로 수정.

            if ClassInd == 1:
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                # img(frame), boxes(왼쪽 위 모서리, 오른쪽 아래 모서리), 색, 두께
                cv2.putText(frame, classLabels[ClassInd - 1] + f'{c}', (boxes[0] + 10, boxes[1] + 40), font,
                            fontScale=font_scale, color=(0, 255, 0), thickness=2)
                c += 1

    cv2.putText(frame, f'Total Persons : {c - 1}', (20, 430), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow('object Detection Tutorial', frame)

    person_num_return(c - 1)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
    #break

cap.release()
cv2.destroyAllWindows()
