#!/usr/bin/env python3
# Author: fanqiang
# create date: 2020/12/10
# Content: 钉钉报警
# desc:
import time
import hmac
import hashlib
import base64
import urllib.parse
import requests


class DingDingWarn:
    """
    钉钉警报类
    """

    def __init__(self, token, secret) -> None:
        super().__init__()
        self.__token = token
        self.__secret = secret

    def sign(self, timestamp):
        secret_enc = self.__secret.encode('utf-8')
        string_to_sign = '{}\n{}'.format(timestamp, self.__secret)
        string_to_sign_enc = string_to_sign.encode('utf-8')
        hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        return sign

    def send_warning_msg(self, title, text):
        """
        发送报警
        :param title:
        :param text:
        :return:
        """
        timestamp = str(round(time.time() * 1000))
        url = "https://oapi.dingtalk.com/robot/send?access_token=" + self.__token + "&timestamp=" + timestamp + "&sign=" + self.sign(timestamp);
        print(url)
        response = requests.post(url, json={"msgtype": "text","text": {"content": title + ' ' + text}},headers={'Content-Type':'application/json'})
        print(response)

# if __name__ == '__main__':
#     ddw = DingDingWarn('bb0e401f6e0578be32c40f7f7a350ac529f0a7981c9cdc9ad3bbe0417a7d67d5', 'SEC9a44c1ba3056cf10a44402a0b727519ada085b08adb926cc3801742c92a73405')
#     ddw.send_warning_msg('测试','大家不要慌2')