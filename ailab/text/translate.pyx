#!/usr/bin/env python
#-*- coding: UTF-8 -*-

class Translate:
    def __init__(self):
        from google.cloud import translate
        self.translate_client = translate.Client()

    def __getitem__(self, text, target="en"):
        return self.translate_client.translate(text,target_language=target)["translatedText"]


if __name__ == "__main__":
    g = Translate()
    print(g.tran("受注伝票を登録後に品目コードの変更を行えますか"))

