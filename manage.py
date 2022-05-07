from flask import Flask, request, jsonify

manage = Flask(__name__)


@manage.route("/")
def hello():
    return "Hello goorm!"


@manage.route("/Person", methods=['POST'])
def Person():
    req = request.get_json()
    Person_Number = req["action"]["detailParams"]["Person_Number"]["value"]  # json파일 읽기

    answer = Person_Number

    # 답변 텍스트 설정
    res = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": answer
                    }
                }
            ]
        }
    }

    # 답변 전송
    return jsonify(res)


if __name__ == "__main__":
    manage.run(host='0.0.0.0', port=5000, threaded=True)